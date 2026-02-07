from torch import nn
import torch
from models.Temporal_Model import *
from models.Prompt_Learner import *
import copy

class GenerateModel(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        self.input_text = input_text
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        
        # Chỉ giữ lại Temporal Net cho nhánh Face
        self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        
        self.clip_model_ = clip_model
        # Đã loại bỏ self.temporal_net_body và self.project_fc
        
    def forward(self, image_face):
        ################# Visual Part #################
        # Face Part Processing
        n, t, c, h, w = image_face.shape
        image_face = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder(image_face.type(self.dtype))
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        video_face_features = self.temporal_net(image_face_features)  # Output shape: (Batch, 512)
        
        # Sử dụng trực tiếp đặc trưng khuôn mặt làm video_features
        video_features = video_face_features
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        ################# Text Part ###################
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        ###############################################

        # Tính toán Logits
        output = video_features @ text_features.t() / 0.01
        return output