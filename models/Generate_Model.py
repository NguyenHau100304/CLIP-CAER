import torch
import torch.nn as nn
from models.Temporal_Model import Temporal_Transformer_Cls
from models.Prompt_Learner import *

class FusionTransformer(nn.Module):
    """
    Thay thế Linear Fusion đơn giản bằng Transformer Fusion.
    Giúp Face và Body 'giao tiếp' với nhau để tìm ra đặc trưng chung.
    """
    def __init__(self, input_dim=512, hidden_dim=512, num_layers=1):
        super().__init__()
        # Encoder Layer của Transformer để trộn thông tin
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim*2, 
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_fusion = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Mạng chiếu cuối cùng
        self.project_fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim) # Thêm LayerNorm để ổn định feature cho CLIP
        )

    def forward(self, face_feat, body_feat):
        # face_feat: (B, 512), body_feat: (B, 512)
        
        # 1. Stack lại thành chuỗi (Sequence) độ dài 2
        # Shape: (B, 2, 512)
        combined = torch.stack([face_feat, body_feat], dim=1)
        
        # 2. Cho Face và Body 'nhìn' thấy nhau qua Self-Attention
        # Output: (B, 2, 512)
        fused = self.transformer_fusion(combined)
        
        # 3. Lấy trung bình (Mean Pooling) hoặc lấy CLS nếu có
        # Ở đây ta lấy trung bình của 2 vector đã được trộn thông tin
        fused_feat = fused.mean(dim=1) # (B, 512)
        
        # 4. Chiếu và chuẩn hóa
        out = self.project_fc(fused_feat)
        return out

class GenerateModel(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        self.input_text = input_text
        
        # --- 1. CLIP Backbone (Giữ nguyên) ---
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        
        # --- 2. Temporal Encoder (QUAY VỀ BẢN GỐC - CLS) ---
        # Vì bạn xác nhận bản này tốt nhất
        self.temporal_net = Temporal_Transformer_Cls(
            num_patches=16, input_dim=512, depth=args.temporal_layers,
            heads=8, mlp_dim=1024, dim_head=64
        )
        
        self.temporal_net_body = Temporal_Transformer_Cls(
            num_patches=16, input_dim=512, depth=args.temporal_layers,
            heads=8, mlp_dim=1024, dim_head=64
        )
        
        # --- 3. Fusion Upgrade: Dùng Transformer Fusion ---
        # Thay thế self.project_fc cũ
        self.fusion_module = FusionTransformer(input_dim=512, num_layers=2) # 2 lớp cho sâu hơn
        
    def forward(self, image_face, image_body):
        # --- Visual Part ---
        
        # 1. Face Encoding
        n, t, c, h, w = image_face.shape
        image_face = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder(image_face.type(self.dtype))
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        video_face_features = self.temporal_net(image_face_features) # (B, 512) - CLS Token
        
        # 2. Body Encoding
        n, t, c, h, w = image_body.shape
        image_body = image_body.contiguous().view(-1, c, h, w)
        image_body_features = self.image_encoder(image_body.type(self.dtype))
        image_body_features = image_body_features.contiguous().view(n, t, -1)
        video_body_features = self.temporal_net_body(image_body_features) # (B, 512) - CLS Token

        # 3. Smart Fusion (NÂNG CẤP)
        video_features = self.fusion_module(video_face_features, video_body_features)
        
        # Chuẩn hóa L2 (Bắt buộc cho CLIP)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        # --- Text Part (Prompt Learning) ---
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # --- Similarity ---
        output = video_features @ text_features.t() / 0.01
        
        return output