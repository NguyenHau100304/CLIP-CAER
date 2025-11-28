import torch
import torch.nn as nn
from torch.nn import functional as F

from models.clip import clip
from models.Temporal_Model import *
from models.Prompt_Learner import *
from models.Text import *

class GenerateModel(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super(GenerateModel, self).__init__()
        self.args = args
        self.input_text = input_text
        
        # 1. Load CLIP Backbone
        self.clip_model = clip_model
        
        # Freeze image encoder của CLIP
        for param in self.clip_model.visual.parameters():
            param.requires_grad = False
            
        # 2. Khởi tạo Prompt Learner và Text Encoder
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype

        # 3. Lớp S-ATT Số 1: Xử lý riêng lẻ Face và Body
        self.visual_transformer_stage1 = Temporal_Transformer_Cls(
            num_patches=16,
            input_dim=512,
            depth=args.temporal_layers,
            heads=8,
            mlp_dim=1024,
            dim_head=64
        )
        
        # 4. Type Embedding: Face (0), Body (1)
        self.type_embedding = nn.Embedding(2, 512)
        
        # 5. Lớp S-ATT Số 2: Fusion Transformer
        # --- QUAN TRỌNG: Thêm batch_first=True ---
        encoderlayer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            batch_first=True  # <--- BẮT BUỘC PHẢI CÓ
        )
        self.visual_transformer_stage2 = nn.TransformerEncoder(encoderlayer, num_layers=3)
        
        # CLS Token cho tầng Fusion
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        
        # Normalization layer cuối cùng
        self.ln_post = nn.LayerNorm(512)

        # --- KHỞI TẠO TRỌNG SỐ ---
        # Áp dụng khởi tạo chuẩn (Truncated Normal) cho các lớp mới thêm vào
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Hàm khởi tạo trọng số giống ViT/BERT để tránh Mode Collapse
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Parameter): # Cho cls_token
             nn.init.trunc_normal_(m, std=0.02)

    def forward(self, imgs, text=None):
        # --- A. Xử lý Text ---
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # --- B. Xử lý Image ---
        
        # 1. Trích xuất đặc trưng từ CLIP
        # -- Face --
        face_imgs = imgs['face']
        b, t, c, h, w = face_imgs.size()
        face_imgs = face_imgs.view(-1, c, h, w)
        with torch.no_grad():
            face_features_raw = self.clip_model.visual(face_imgs)
        face_features_raw = face_features_raw.view(b, t, -1).float()

        # -- Body --
        body_imgs = imgs['body']
        body_imgs = body_imgs.view(-1, c, h, w)
        with torch.no_grad():
            body_features_raw = self.clip_model.visual(body_imgs)
        body_features_raw = body_features_raw.view(b, t, -1).float()

        # 2. Qua lớp S-ATT Số 1 (Lấy Sequence)
        face_feat_seq = self.visual_transformer_stage1(face_features_raw, return_sequence=True)
        body_feat_seq = self.visual_transformer_stage1(body_features_raw, return_sequence=True)
        
        # 3. Cộng Type Embedding
        device = face_feat_seq.device
        type_face = torch.zeros((1, 1), dtype=torch.long, device=device)
        type_body = torch.ones((1, 1), dtype=torch.long, device=device)
        
        # Broadcasting tự động cộng (1,1,D) vào (B,T,D)
        face_feat_aug = face_feat_seq + self.type_embedding(type_face)
        body_feat_aug = body_feat_seq + self.type_embedding(type_body)
        
        # 4. Fusion (Nối chuỗi)
        # Input shape: (B, 2*T, D)
        fused_features = torch.cat([face_feat_aug, body_feat_aug], dim=1)
        
        # 5. Thêm CLS Token
        cls_tokens = self.cls_token.expand(b, -1, -1) # (B, 1, D)
        x = torch.cat((cls_tokens, fused_features), dim=1) # (B, 2*T+1, D)

        # 6. Qua Transformer Stage 2
        # Vì đã set batch_first=True, input x có dạng (B, Seq, D) là đúng
        out = self.visual_transformer_stage2(x)

        # 7. Lấy CLS Output (token đầu tiên)
        final_visual_features = out[:, 0, :]
        
        # 8. Normalize và Output
        final_visual_features = self.ln_post(final_visual_features)
        final_visual_features = final_visual_features / final_visual_features.norm(dim=-1, keepdim=True)
        
        logits = final_visual_features @ text_features.t() * self.clip_model.logit_scale.exp()
        return logits