# builders.py

import argparse
from typing import Tuple
import os
import torch
import torch.utils.data
from models.clip import clip

from dataloader.video_dataloader import train_data_loader, test_data_loader
from models.Generate_Model import GenerateModel
from models.Text import *
from utils.utils import *


def build_model(args: argparse.Namespace, input_text: list) -> torch.nn.Module:
    print("Loading pretrained CLIP model...")
    CLIP_model, _ = clip.load(args.clip_path, device='cpu')

    print("\nInput Text Prompts:")
    for text in input_text:
        print(text)

    print("\nInstantiating GenerateModel...")
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)

    for name, param in model.named_parameters():
        param.requires_grad = False

    trainable_params_keywords = ["image_encoder", "temporal_net", "prompt_learner", "temporal_net_body", "project_fc", "temporal_net_skeleton"] # THÊM MỚI
    print('\nTrainable parameters:')
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_params_keywords):
            param.requires_grad = True
            print(f"- {name}")
    print('************************\n')

    return model


def get_class_info(args: argparse.Namespace) -> Tuple[list, list]:
    """
    (Giữ nguyên như file gốc)
    """
    if args.dataset == "RAER":
        class_names = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction.']
        class_names_with_context = class_names_with_context_5
        class_descriptor = class_descriptor_5
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented yet.")

    if args.text_type == "class_names":
        input_text = class_names
    elif args.text_type == "class_names_with_context":
        input_text = class_names_with_context
    elif args.text_type == "class_descriptor":
        input_text = class_descriptor
    else:
        raise ValueError(f"Unknown text_type: {args.text_type}")

    return class_names, input_text



def build_dataloaders(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
    train_annotation_file_path = args.train_annotation
    test_annotation_file_path = args.test_annotation

    print("Loading train data...")
    # === CẬP NHẬT: Thêm bounding_box_skeleton ===
    train_data = train_data_loader(
        list_file=train_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,dataset_name=args.dataset,
        bounding_box_face=args.bounding_box_face, bounding_box_body=args.bounding_box_body,
        bounding_box_skeleton=args.bounding_box_skeleton # MỚI
    )
    
    print("Loading test data...")
    # === CẬP NHẬT: Thêm bounding_box_skeleton ===
    test_data = test_data_loader(
        list_file=test_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,
        bounding_box_face=args.bounding_box_face, bounding_box_body=args.bounding_box_body,
        bounding_box_skeleton=args.bounding_box_skeleton # MỚI
    )

    print("Creating DataLoader instances...")
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    return train_loader, val_loader