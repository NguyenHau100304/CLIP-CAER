import torch
from torch import nn
from torch.nn import functional as F
from models.Temporal_Model import Temporal_Transformer_Cls
from models.Prompt_Learner import PromptLearner
from models.Prompt_Learner import TextEncoder

class GatedFusion(nn.Module):
    """
    Mô-đun nâng cấp: Tự động học trọng số để kết hợp Face và Body
    thay vì chỉ nối (concat) đơn thuần.
    """
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        # Mạng học trọng số cổng (Gate)
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2), # Output 2 giá trị: weight cho face và body
            nn.Softmax(dim=1)
        )
        # Mạng chiếu sau khi gộp
        self.project_fc = nn.Linear(input_dim, input_dim)

    def forward(self, face_feat, body_feat):
        # face_feat: (B, 512), body_feat: (B, 512)
        
        # 1. Tính toán trọng số dựa trên cả 2 đặc trưng
        concat = torch.cat([face_feat, body_feat], dim=1) # (B, 1024)
        weights = self.gate_net(concat) # (B, 2)
        
        w_face = weights[:, 0].unsqueeze(1) # (B, 1)
        w_body = weights[:, 1].unsqueeze(1) # (B, 1)
        
        # 2. Tổng có trọng số (Weighted Sum)
        # Cho phép mô hình "lắng nghe" bên nào quan trọng hơn
        fused_feat = (face_feat * w_face) + (body_feat * w_body)
        
        # 3. Chiếu và chuẩn hóa
        out = self.project_fc(fused_feat)
        return out

class GenerateModel(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        self.input_text = input_text
        
        # 1. CLIP Backbone & Text Encoder (GIỮ NGUYÊN TỪ MÔ HÌNH GỐC)
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        
        # 2. Temporal Models (GIỮ NGUYÊN)
        # Xử lý Face và Body độc lập
        self.temporal_net = Temporal_Transformer_Cls(
            num_patches=16, input_dim=512, depth=args.temporal_layers,
            heads=8, mlp_dim=1024, dim_head=64
        )
        
        self.temporal_net_body = Temporal_Transformer_Cls(
            num_patches=16, input_dim=512, depth=args.temporal_layers,
            heads=8, mlp_dim=1024, dim_head=64
        )
        
        # 3. Fusion Upgrade: Thay Linear đơn giản bằng GatedFusion
        # self.project_fc = nn.Linear(1024, 512) # <-- Cũ
        self.fusion_module = GatedFusion(input_dim=512) # <-- Mới
        
    def forward(self, image_face, image_body):
        # --- Visual Part ---
        
        # 1. Face Encoding
        n, t, c, h, w = image_face.shape
        image_face = image_face.contiguous().view(-1, c, h, w)
        image_face_features = self.image_encoder(image_face.type(self.dtype))
        image_face_features = image_face_features.contiguous().view(n, t, -1)
        video_face_features = self.temporal_net(image_face_features) # (B, 512)
        
        # 2. Body Encoding
        n, t, c, h, w = image_body.shape
        image_body = image_body.contiguous().view(-1, c, h, w)
        image_body_features = self.image_encoder(image_body.type(self.dtype))
        image_body_features = image_body_features.contiguous().view(n, t, -1)
        video_body_features = self.temporal_net_body(image_body_features) # (B, 512)

        # 3. Fusion (NÂNG CẤP)
        # Thay vì concat cứng, dùng Gated Fusion
        video_features = self.fusion_module(video_face_features, video_body_features)
        
        # Chuẩn hóa (Quan trọng cho CLIP)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        # --- Text Part (Prompt Learning) ---
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # --- Similarity ---
        # Tính Cosine Similarity (Temperature 0.01 như code gốc)
        output = video_features @ text_features.t() / 0.01
        
        return output