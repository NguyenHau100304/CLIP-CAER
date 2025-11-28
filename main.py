# ==================== Imports ====================
import argparse
import datetime
import os
import random
import shutil
import time

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import warnings
from models.clip import clip
from dataloader.video_dataloader import train_data_loader, test_data_loader
from models.Generate_Model import GenerateModel
from models.Text import *
from trainer import Trainer
from utils.loss import *
from utils.utils import *
from utils.builders import *
from torch.utils.data import WeightedRandomSampler, DataLoader


# Ignore specific warnings (for cleaner output)
warnings.filterwarnings("ignore", category=UserWarning)
# Use 'Agg' backend for matplotlib (no GUI required)
matplotlib.use('Agg')

# ==================== Argument Parser ====================
parser = argparse.ArgumentParser(
    description='A highly configurable training script for RAER Dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# --- Experiment and Environment ---
exp_group = parser.add_argument_group('Experiment & Environment', 'Basic settings for the experiment')
exp_group.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help="Execution mode: 'train' for a full training run, 'eval' for evaluation only.")
exp_group.add_argument('--eval-checkpoint', type=str, default='/media/D/zlm/code/CLIP_CAER/outputs_1/test-[07-09]-[22:24]/model_best.pth',
                       help="Path to the model checkpoint for evaluation mode (e.g., outputs/exp_name/model_best.pth).")
exp_group.add_argument('--exper-name', type=str, default='test', help='A name for the experiment to create a unique output folder.')
exp_group.add_argument('--dataset', type=str, default='RAER', help='Name of the dataset to use.')
exp_group.add_argument('--gpu', type=int, default=2, help='ID of the GPU to use.')
exp_group.add_argument('--workers', type=int, default=4, help='Number of data loading workers.')
exp_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

# --- Data & Path ---
path_group = parser.add_argument_group('Data & Path', 'Paths to datasets and pretrained models')
path_group.add_argument('--root-dir', type=str, default='/media/F/FERDataset/AER-DB', help='Root directory of the dataset.')
path_group.add_argument('--train-annotation', type=str, default='RAER/train_abs.txt', help='Path to training annotation file, relative to root-dir.')
path_group.add_argument('--test-annotation', type=str, default='RAER/test_abs.txt', help='Path to testing annotation file, relative to root-dir.')
path_group.add_argument('--clip-path', type=str, default='/media/D/zlm/code/single_four/models/ViT-B-32.pt', help='Path to the pretrained CLIP model.')
path_group.add_argument('--bounding-box-face', type=str, default='/media/F/FERDataset/AER-DB/RAER/bounding_box/face_abs.json')
path_group.add_argument('--bounding-box-body', type=str, default="/media/F/FERDataset/AER-DB/RAER/bounding_box/body_abs.json")

# --- Training Control ---
train_group = parser.add_argument_group('Training Control', 'Parameters to control the training process')
train_group.add_argument('--epochs', type=int, default=20, help='Total number of training epochs.')
train_group.add_argument('--batch-size', type=int, default=8, help='Batch size for training and validation.')
train_group.add_argument('--print-freq', type=int, default=10, help='Frequency of printing training logs.')

# --- Optimizer & Learning Rate ---
optim_group = parser.add_argument_group('Optimizer & LR', 'Hyperparameters for the optimizer and scheduler')
optim_group.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate for main modules.')
optim_group.add_argument('--lr-image-encoder', type=float, default=1e-5, help='Learning rate for the image encoder part.')
optim_group.add_argument('--lr-prompt-learner', type=float, default=1e-3, help='Learning rate for the prompt learner.')
optim_group.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for the optimizer.')
optim_group.add_argument('--momentum', type=float, default=0.9, help='Momentum for the SGD optimizer.')
optim_group.add_argument('--milestones', nargs='+', type=int, default=[10, 15], help='Epochs at which to decay the learning rate.')
optim_group.add_argument('--gamma', type=float, default=0.1, help='Factor for learning rate decay.')

# --- Model & Input ---
model_group = parser.add_argument_group('Model & Input', 'Parameters for model architecture and data handling')
model_group.add_argument('--text-type', default='class_descriptor', choices=['class_names', 'class_names_with_context', 'class_descriptor'], help='Type of text prompts to use.')
model_group.add_argument('--temporal-layers', type=int, default=1, help='Number of layers in the temporal modeling part.')
model_group.add_argument('--contexts-number', type=int, default=8, help='Number of context vectors in the prompt learner.')
model_group.add_argument('--class-token-position', type=str, default="end", help='Position of the class token in the prompt.')
model_group.add_argument('--class-specific-contexts', type=str, default='True', choices=['True', 'False'], help='Whether to use class-specific context prompts.')
model_group.add_argument('--load_and_tune_prompt_learner', type=str, default='True', choices=['True', 'False'], help='Whether to load and fine-tune the prompt learner.')
model_group.add_argument('--num-segments', type=int, default=16, help='Number of segments to sample from each video.')
model_group.add_argument('--duration', type=int, default=1, help='Duration of each segment.')
model_group.add_argument('--image-size', type=int, default=224, help='Size to resize input images to.')


# ==================== Helper Functions ====================
def setup_environment(args: argparse.Namespace) -> argparse.Namespace:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    
    print("Environment and random seeds set successfully.")
    return args


def setup_paths_and_logging(args: argparse.Namespace) -> argparse.Namespace:
    now = datetime.datetime.now()
    time_str = now.strftime("-[%m-%d]-[%H:%M]")
    
    args.name = args.exper_name + time_str
        
    args.output_path = os.path.join("outputs", args.name)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    print('************************')
    print("Running with the following configuration:")
    for k, v in vars(args).items():
        print(f'{k} = {v}')
    print('************************')
    
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    with open(log_txt_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k} = {v}\n')
        f.write('*'*50 + '\n\n')
        
    return args

# ==================== Training Function ====================
def run_training(args: argparse.Namespace) -> None:
    # Paths for logging and saving
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    log_curve_path = os.path.join(args.output_path, 'log.png')
    log_confusion_matrix_path = os.path.join(args.output_path, 'confusion_matrix.png')
    checkpoint_path = os.path.join(args.output_path, 'model.pth')
    best_checkpoint_path = os.path.join(args.output_path, 'model_best.pth')        
    best_uar = 0.0
    start_epoch = 0
    recorder = RecorderMeter(args.epochs)
    
    # Build model
    print("=> Building model...")
    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text)
    model = model.to(args.device)
    print("=> Model built and moved to device successfully.")

    # Load data
    print("=> Building dataloaders...")
    # 1. Load dataset (nhưng chưa tạo loader vội)
    # Chúng ta cần truy cập vào dataset để lấy danh sách label
    # Lưu ý: Bạn cần sửa lại hàm build_dataloaders để trả về dataset, 
    # HOẶC tự khởi tạo dataset ở đây giống như trong builders.py
    
    # Ví dụ cách làm thủ công để lấy được dataset:
    train_dataset = train_data_loader(
        list_file=args.train_annotation,
        num_segments=args.num_segments,
        duration=args.duration,
        image_size=args.image_size,
        dataset_name=args.dataset,
        bounding_box_face=args.bounding_box_face,
        bounding_box_body=args.bounding_box_body
    )

    # 2. Tính toán trọng số cho Sampler
    print("Đang tính toán trọng số lấy mẫu (Weighted Sampler)...")
    
    # Lấy danh sách tất cả label trong tập train
    # train_dataset.video_list là list các object VideoRecord, mỗi cái có thuộc tính .label (int)
    # Lưu ý: record.label trong code của bạn có thể là 1-5 hoặc 0-4. 
    # VideoRecord trả về int(row[2]), thường dataset txt ghi 1-5.
    all_train_labels = [x.label for x in train_dataset.video_list]
    
    # Đếm số lượng: Class 1 (1383), 2 (129), 3 (30), 4 (157), 5 (422)
    # Giả sử label trong file txt là 0,1,2,3,4. Nếu là 1,2,3,4,5 thì cần trừ 1 khi index
    
    class_counts = [1383, 129, 30, 157, 422] 
    num_samples = sum(class_counts)
    
    # Trọng số cho mỗi CLASS (nghịch đảo tần suất)
    # Class ít xuất hiện -> trọng số cao
    class_weights = [num_samples / x for x in class_counts]
    
    # Trọng số cho từng MẪU (Sample)
    # Giả sử label dataset của bạn chạy từ 1 đến 5 -> trừ 1 để lấy index 0-4
    weights_per_sample = [class_weights[label - 1] for label in all_train_labels]
    
    # Chuyển sang tensor
    weights_per_sample = torch.DoubleTensor(weights_per_sample)

    # 3. Tạo Sampler
    # num_samples: Tổng số mẫu muốn lấy trong 1 epoch (thường bằng len dataset)
    # replacement=True: Cho phép lấy lặp lại (quan trọng với class ít mẫu)
    sampler = WeightedRandomSampler(weights_per_sample, len(weights_per_sample), replacement=True)

    # 4. Tạo DataLoader với Sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False, # QUAN TRỌNG: Khi dùng sampler, shuffle PHẢI là False
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler, # Đưa sampler vào đây
        drop_last=True
    )
    
    # Val loader giữ nguyên
    _, val_loader = build_dataloaders(args) # Chỉ lấy val_loader
    print("=> Dataloaders built successfully.")

    # Loss and optimizer
    print("DEBUG: Using Weighted Sampler -> Disable Class Weights in Loss")
    criterion = nn.CrossEntropyLoss().to(args.device) # <--- Trở về mặc định
    optimizer = torch.optim.SGD([
        {"params": model.prompt_learner.parameters(), "lr": args.lr_prompt_learner},
        {"params": model.visual_transformer_stage1.parameters(), "lr": args.lr},
        {"params": model.type_embedding.parameters(), "lr": args.lr},
        {"params": model.fusion_net.parameters(), "lr": args.lr},
        {"params": model.ln_post.parameters(), "lr": args.lr_image_encoder},
    ], momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    
    # Trainer
    trainer = Trainer(model, criterion, optimizer, scheduler, args.device, log_txt_path)
    
    for epoch in range(start_epoch, args.epochs):
        inf = f'******************** Epoch: {epoch} ********************'
        start_time = time.time()
        print(inf)
        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')

        # Log current learning rates
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        lr_str = ' '.join([f'{lr:.1e}' for lr in current_lrs])
        log_msg = f'Current learning rates: {lr_str}'
        with open(log_txt_path, 'a') as f:
            f.write(log_msg + '\n')
        print(log_msg)

        # Train & Validate
        train_war, train_uar, train_los, _ = trainer.train_epoch(train_loader, epoch)
        val_war, val_uar, val_los, _ = trainer.validate(val_loader, str(epoch))
        scheduler.step()

        # Save checkpoint
        is_best = val_uar > best_uar
        best_uar = max(val_uar, best_uar)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_uar, 
            'optimizer': optimizer.state_dict(),
            'recorder': recorder
        }, is_best, checkpoint_path, best_checkpoint_path)

        # Record metrics
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_war, val_los, val_war)
        recorder.plot_curve(log_curve_path)
        print(f'The best UAR: {best_uar:.4f}')
        print(f'An epoch time: {epoch_time:.2f}s\n')
        with open(log_txt_path, 'a') as f:
            f.write(f'The best UAR: {best_uar:.4f}\n')
            f.write(f'An epoch time: {epoch_time:.2f}s\n\n')

    # Final evaluation with best model
    pre_trained_dict = torch.load(best_checkpoint_path,map_location=f"cuda:{args.gpu}")['state_dict']
    model.load_state_dict(pre_trained_dict)
    computer_uar_war(
        val_loader=val_loader,
        model=model,
        device=args.device,
        class_names=class_names,
        log_confusion_matrix_path=log_confusion_matrix_path,
        log_txt_path=log_txt_path,
        title=f"Confusion Matrix on {args.dataset}"
    )

def run_eval(args: argparse.Namespace) -> None:
    print("=> Starting evaluation mode...")
    log_txt_path = os.path.join(args.output_path, 'log.txt')
    log_confusion_matrix_path = os.path.join(args.output_path, 'confusion_matrix.png')

    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text)
    model = model.to(args.device)

    # Load pretrained weights
    model.load_state_dict(torch.load(args.eval_checkpoint,map_location=args.device)['state_dict'])

    # Load data
    _, val_loader = build_dataloaders(args)

    # Run evaluation
    computer_uar_war(
        val_loader=val_loader,
        model=model,
        device=args.device,
        class_names=class_names,
        log_confusion_matrix_path=log_confusion_matrix_path,
        log_txt_path=log_txt_path,
        title=f"Confusion Matrix on {args.dataset}"
    )
    print("=> Evaluation complete.")


# ==================== Entry Point ====================
if __name__ == '__main__':
    args = parser.parse_args()
    args = setup_environment(args)
    args = setup_paths_and_logging(args)
    
    if args.mode == 'eval':
        run_eval(args)
    else:
        run_training(args)