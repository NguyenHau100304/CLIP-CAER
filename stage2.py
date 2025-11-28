import argparse
import os
import random
import time
import warnings
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# Import các module từ source của bạn
from models.Generate_Model import GenerateModel
from dataloader.video_dataloader import train_data_loader, test_data_loader
from trainer import Trainer
from utils.builders import build_model, get_class_info
from utils.utils import RecorderMeter

# --- KHAI BÁO CẤU HÌNH TRỰC TIẾP Ở ĐÂY ĐỂ DỄ CHỈNH ---
# Hãy thay đổi đường dẫn này thành đường dẫn thực tế chứa file model_best.pth của Stage 1
STAGE1_CHECKPOINT_PATH = '/kaggle/input/raer-thucnghiemchinh/outputs/test-[11-28]-[10:01]/model_best.pth' 
# (Lưu ý: Bạn cần vào folder outputs để copy đúng tên folder ngày giờ)

def setup_environment(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    return args

def training_stage2():
    # 1. Cấu hình Argument (Copy từ main cũ để model build giống hệt)
    parser = argparse.ArgumentParser()
    # --- CÁC THAM SỐ CẤU HÌNH MODEL (PHẢI KHỚP TUYỆT ĐỐI VỚI STAGE 1) ---
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
    
    # --- ĐIỂM SỬA QUAN TRỌNG TẠI ĐÂY ---
    parser.add_argument('--class-specific-contexts', type=str, default='True') # Phải là String 'True'
    parser.add_argument('--load_and_tune_prompt_learner', type=str, default='True') # Phải là String 'True'
    # -----------------------------------

    parser.add_argument('--temporal-layers', type=int, default=4)
    parser.add_argument('--num-segments', type=int, default=16)
    parser.add_argument('--duration', type=int, default=1)
    parser.add_argument('--image-size', type=int, default=224)
    
    # Param chạy Stage 2
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
    print("BẮT ĐẦU GIAI ĐOẠN 2: FINE-TUNE CLASSIFIER (CRT)")
    print(f"Loading weights from: {STAGE1_CHECKPOINT_PATH}")
    print("="*40)

    # 2. Build Model
    print("=> Building model...")
    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text)
    model = model.to(args.device)

    # 3. Load Trọng số từ Stage 1
    if os.path.isfile(STAGE1_CHECKPOINT_PATH):
        checkpoint = torch.load(STAGE1_CHECKPOINT_PATH, map_location=args.device)
        # Xử lý nếu state_dict có prefix 'module.'
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("=> Loaded checkpoint successfully.")
    else:
        print(f"=> [ERROR] Không tìm thấy file: {STAGE1_CHECKPOINT_PATH}")
        return

    # 4. Freeze Backbone
    print("=> Freezing backbone layers...")
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if "classifier" in name: 
            param.requires_grad = True
            trainable_count += 1
            print(f" -> Unfrozen (Trainable): {name}")
        else:
            param.requires_grad = False
            frozen_count += 1
    print(f"=> Freezing Complete. Trainable layers: {trainable_count}, Frozen layers: {frozen_count}")

    # 5. Dataset & DataLoader (Quan trọng: Shuffle=True, Không Sampler)
    print("=> Building Standard DataLoader (Natural Distribution)...")
    train_dataset_ft = train_data_loader(
        list_file=args.train_annotation,
        num_segments=args.num_segments,
        duration=args.duration,
        image_size=args.image_size,
        dataset_name=args.dataset,
        bounding_box_face=args.bounding_box_face,
        bounding_box_body=args.bounding_box_body
    )
    
    train_loader_ft = DataLoader(
        train_dataset_ft,
        batch_size=args.batch_size,
        shuffle=True,       # <--- QUAN TRỌNG: Shuffle để học phân phối thực
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Load Val Loader
    test_dataset = test_data_loader(
        list_file=args.test_annotation,
        num_segments=args.num_segments,
        duration=args.duration,
        image_size=args.image_size,
        bounding_box_face=args.bounding_box_face,
        bounding_box_body=args.bounding_box_body
    )
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # 6. Optimizer & Loss
    # Learning Rate nhỏ (1e-3 hoặc 1e-4) để tinh chỉnh nhẹ nhàng
    optimizer_ft = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    
    criterion_ft = nn.CrossEntropyLoss().to(args.device) # Loss mặc định, không trọng số
    
    # 7. Training Loop (5 Epochs)
    # Truyền scheduler=None vì chạy ngắn không cần decay
    trainer_ft = Trainer(model, criterion_ft, optimizer_ft, None, args.device, os.path.join(args.output_path, 'log_ft.txt'))

    print("=> Starting Fine-tuning...")
    for epoch in range(5):
        print(f"\n--- Fine-tune Epoch {epoch} ---")
        # Train
        trainer_ft.train_epoch(train_loader_ft, epoch)
        
        # Validate
        val_war, val_uar, _, _ = trainer_ft.validate(val_loader, f"FT-{epoch}")
        
        # Save per epoch
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': val_war,
        }, os.path.join(args.output_path, f'model_ft_epoch{epoch}.pth'))

    print("=> Fine-tuning Finished.")

if __name__ == '__main__':
    training_stage2()