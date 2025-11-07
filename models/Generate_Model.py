from torch import nn
from models.Temporal_Model import *
from models.Prompt_Learner import *
import copy
import torch # Thêm import torch

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
        
        # === CẬP NHẬT: Giữ 2 temporal net cũ và thêm 1 net cho skeleton ===
        self.temporal_net = Temporal_Transformer_Cls(num_patches=args.num_segments,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        
        self.temporal_net_body = Temporal_Transformer_Cls(num_patches=args.num_segments,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        
        # MỚI: Temporal net cho skeleton
        self.temporal_net_skeleton = Temporal_Transformer_Cls(num_patches=args.num_segments,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)
        
        self.clip_model_ = clip_model
        
        # CẬP NHẬT: Fusion layer nhận 3*512 = 1536-dim input
        self.project_fc = nn.Linear(1536, 512) 
        
    # === CẬP NHẬT: Thêm image_skeleton vào forward ===
    def forward(self, image_face, image_body, image_skeleton):
        ################# Visual Part #################
        # 1. Face Part
        n, t, c, h, w = image_face.shape
        image_face = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder(image_face.type(self.dtype))
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        video_face_features = self.temporal_net(image_face_features)  # (n, 512)
        
        # 2. Body Part
        n_b, t_b, c_b, h_b, w_b = image_body.shape
        image_body = image_body.contiguous().view(-1, c_b, h_b, w_b)
        image_body_features = self.image_encoder(image_body.type(self.dtype))
        image_body_features = image_body_features.contiguous().view(n, t, -1)
        video_body_features = self.temporal_net_body(image_body_features) # (n, 512)

        # 3. MỚI: Skeleton Part
        n_s, t_s, c_s, h_s, w_s = image_skeleton.shape
        image_skeleton = image_skeleton.contiguous().view(-1, c_s, h_s, w_s)
        image_skeleton_features = self.image_encoder(image_skeleton.type(self.dtype))
        image_skeleton_features = image_skeleton_features.contiguous().view(n, t, -1)
        video_skeleton_features = self.temporal_net_skeleton(image_skeleton_features) # (n, 512)

        # CẬP NHẬT: Nối (concatenate) cả 3 phần
        video_features = torch.cat((video_face_features, video_body_features, video_skeleton_features), dim=-1) # (n, 1536)
        video_features = self.project_fc(video_features) # (n, 512)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        ################# Text Part ###################
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        ###############################################

        output = video_features @ text_features.t() / 0.01
        return output