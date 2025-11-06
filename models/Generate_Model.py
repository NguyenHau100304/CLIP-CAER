from torch import nn
from models.Temporal_Model import * # Import class JointTemporalEncoder mới
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
        
        # --- NÂNG CẤP: SỬ DỤNG MỘT JOINT TEMPORAL ENCODER ---
        # Thay vì 2 encoder riêng biệt, chúng ta dùng 1 encoder
        # num_patches_per_stream = args.num_segments (ví dụ: 16)
        self.temporal_encoder = JointTemporalEncoder(
            num_patches_per_stream=args.num_segments,
            input_dim=512, # Output dim của CLIP-ViT-B/32 image encoder
            depth=args.temporal_layers,
            heads=8,
            mlp_dim=1024,
            dim_head=64
        )
        # --- KẾT THÚC NÂNG CẤP ---
        
        self.clip_model_ = clip_model
        # Không cần project_fc nữa vì đầu ra đã là 512-dim
        
    def forward(self, image_face, image_body):
        ################# Visual Part #################
        # Face Part
        n, t, c, h, w = image_face.shape
        image_face = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder(image_face.type(self.dtype))
        # Reshape về (batch, num_segments, dim) -> (n, 16, 512)
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        
        # Body Part
        n_body, t_body, c_body, h_body, w_body = image_body.shape
        # Đảm bảo batch size và số segment khớp nhau
        assert n == n_body and t == t_body, "Kích thước đầu vào của face và body không khớp"
        image_body = image_body.contiguous().view(-1, c_body, h_body, w_body)
        image_body_features = self.image_encoder(image_body.type(self.dtype))
        # Reshape về (batch, num_segments, dim) -> (n, 16, 512)
        image_body_features = image_body_features.contiguous().view(n, t, -1)

        # --- NÂNG CẤP: ĐƯA VÀO JOINT ENCODER ---
        # Đầu ra video_features sẽ có shape: (n, 512)
        video_features = self.temporal_encoder(image_face_features, image_body_features)
        
        # --- KẾT THÚC NÂNG CẤP ---
        
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        ################# Text Part ###################
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        ###############################################

        output = video_features @ text_features.t() / 0.01
        return output