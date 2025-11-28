import argparse
import os
import random
import time
import warnings
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# Import các module
from dataloader.video_dataloader import train_data_loader, test_data_loader
from trainer import Trainer
from utils.builders import build_model, get_class_info

# --- CẤU HÌNH ĐƯỜNG DẪN CHECKPOINT ---
# Lưu ý: Kiểm tra kỹ đường dẫn này có tồn tại không
STAGE1_CHECKPOINT_PATH = '/kaggle/input/raer-thucnghiemchinh/outputs/test-[11-28]-[10:01]/model_best.pth'

def setup_environment(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    return args

def training_stage2():
    parser = argparse.ArgumentParser()
    # --- CẤU HÌNH MODEL (PHẢI GIỐNG HỆT STAGE 1) ---
    parser.add_argument('--dataset', type=str, default='RAER')
    parser.add_argument('--root-dir', type=str, default='/kaggle/input/raer-enhanced/RAER')
    parser.add_argument('--train-annotation', type=str, default='/kaggle/working/CLIP-CAER/annotation/train.txt')
    parser.add_argument('--test-annotation', type=str, default='/kaggle/working/CLIP-CAER/annotation/test.txt')
    parser.add_argument('--clip-path', type=str, default='ViT-B/32')
    parser.add_argument('--bounding-box-face', type=str, default='/kaggle/working/CLIP-CAER/bounding_box/face.json')
    parser.add_argument('--bounding-box-body', type=str, default='/kaggle/working/CLIP-CAER/bounding_box/body.json')
    
    # Text & Context settings
    parser.add_argument('--text-type', type=str, default='class_descriptor')
    parser.add_argument('--contexts-number', type=int, default=8)
    parser.add_argument('--class-token-position', type=str, default='end')
    parser.add_argument('--class-specific-contexts', type=str, default='True') 
    parser.add_argument('--load_and_tune_prompt_learner', type=str, default='True')
    
    # Model params
    parser.add_argument('--temporal-layers', type=int, default=4)
    parser.add_argument('--num-segments', type=int, default=16)
    parser.add_argument('--duration', type=int, default=1)
    parser.add_argument('--image-size', type=int, default=224)
    
    # Stage 2 params
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-path', type=str, default='/kaggle/working/CLIP-CAER/outputs/finetune_result')
    
    args = parser.parse_args()
    args = setup_environment(args)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    print("\n" + "="*40)
    print("BẮT ĐẦU GIAI ĐOẠN 2: FINE-TUNE PROMPTS (Re-balancing)")
    print(f"Loading weights from: {STAGE1_CHECKPOINT_PATH}")
    print("="*40)

    # 1. Build Model
    print("=> Building model...")
    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text)
    model = model.to(args.device)

    # 2. Load Weights từ Stage 1
    if os.path.isfile(STAGE1_CHECKPOINT_PATH):
        checkpoint = torch.load(STAGE1_CHECKPOINT_PATH, map_location=args.device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        print("=> Loaded checkpoint successfully.")
    else:
        print(f"=> [ERROR] File not found: {STAGE1_CHECKPOINT_PATH}")
        return

    # 3. Đóng băng Visual, Mở khóa Prompt Learner
    print("=> Configuring Trainable Parameters...")
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        # Unfreeze Prompt Learner để cân bằng lại Class Bias
        if "prompt_learner" in name: 
            param.requires_grad = True
            trainable_count += 1
            print(f" -> Unfrozen: {name}")
        else:
            param.requires_grad = False
            frozen_count += 1
            
    print(f"=> Complete. Trainable: {trainable_count}, Frozen: {frozen_count}")

    # 4. Optimizer (Chỉ chứa tham số unfreeze)
    optimizer_ft = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.0001, # LR nhỏ
        momentum=0.9, 
        weight_decay=1e-4
    )

    # 5. Dữ liệu chuẩn (Shuffle=True, Không Sampler)
    print("=> Building Standard DataLoader...")
    
    # Train Loader
    train_dataset = train_data_loader(
        list_file=args.train_annotation,
        num_segments=args.num_segments,
        duration=args.duration,
        image_size=args.image_size,
        dataset_name=args.dataset,
        bounding_box_face=args.bounding_box_face,
        bounding_box_body=args.bounding_box_body
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,       # Shuffle=True để học phân phối thực tế
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Val Loader - Sử dụng hàm helper bên dưới
    _, val_loader = build_dataloaders(args)

    # 6. Loss chuẩn (Không trọng số)
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    trainer = Trainer(model, criterion, optimizer_ft, None, args.device, os.path.join(args.output_path, 'log_ft.txt'))

    # 7. Chạy 5 Epochs
    print("=> Starting Fine-tuning...")
    for epoch in range(5):
        print(f"\n--- Fine-tune Epoch {epoch} ---")
        trainer.train_epoch(train_loader, epoch)
        val_war, val_uar, _, _ = trainer.validate(val_loader, f"FT-{epoch}")
        print(f"Result: WAR={val_war:.2f}%, UAR={val_uar:.2f}%")
        
        # Save
        torch.save({
            'state_dict': model.state_dict(),
            'best_acc': val_uar,
        }, os.path.join(args.output_path, f'model_ft_epoch{epoch}.pth'))

# Helper function sửa lỗi: Gọi trực tiếp test_data_loader mà không cần định nghĩa transform
def build_dataloaders(args):
    # test_data_loader tự xử lý transform nội bộ
    val_dataset = test_data_loader(
        list_file=args.test_annotation,
        num_segments=args.num_segments,
        duration=args.duration,
        image_size=args.image_size,
        bounding_box_face=args.bounding_box_face,
        bounding_box_body=args.bounding_box_body
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return None, val_loader

if __name__ == '__main__':
    training_stage2()