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
        #print(f"Đang khởi tạo mô hình với CLIP Backbone: {args.backbone}...")
        
        # 1. Load CLIP Backbone
        # (Giả sử clip.load trả về model và preprocess, ta chỉ cần model)
        self.clip_model = clip_model
        
        # Freeze image encoder của CLIP (thường làm vậy để tiết kiệm bộ nhớ và giữ feature tốt)
        for param in self.clip_model.visual.parameters():
            param.requires_grad = False
            
        # Lấy feature dimension (ví dụ ViT-B/32 là 512, ViT-L/14 là 768)
        
        # 2. Khởi tạo Prompt Learner và Text Encoder (Xử lý văn bản)
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        # 3. Lớp S-ATT Số 1: Xử lý riêng lẻ Face và Body
        # Lưu ý: Input vào đây là chuỗi features theo thời gian
        self.visual_transformer_stage1 = Temporal_Transformer_Cls(
            num_patches=16,
            input_dim=512,
            depth=args.temporal_layers,      # Số lớp encoder (ví dụ: 4)
            heads=8,      # Số head attention (ví dụ: 8)
            mlp_dim=1024,
            dim_head=64
        )
        
        # 4. Type Embedding: Để phân biệt đâu là Face (0), đâu là Body (1)
        # Kích thước (2, feature_dim)
        self.type_embedding = nn.Embedding(2, 512)
        
        # 5. Lớp S-ATT Số 2: Fusion Transformer (Cái mới thêm vào)
        # Lớp này sẽ học sự tương quan giữa Face và Body sau khi đã cộng gộp
        # self.visual_transformer_stage2 = Transformer(
        #     num_patches=32, 
        #     input_dim=512,
        #     depth=args.temporal_layers,  # Có thể dùng độ sâu giống hoặc khác stage 1
        #     heads=8,
        #     mlp_dim=1024,
        #     dim_head=64
        # )
        encoderlayer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu')
        self.visual_transformer_stage2 = nn.TransformerEncoder(encoderlayer, num_layers=3)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        # Normalization layer cuối cùng trước khi tính similarity
        self.ln_post = nn.LayerNorm(512)

    def forward(self, imgs, text=None):
        """
        image: dict chứa 'face' và 'body'. Shape mỗi cái: (Batch, Time, C, H, W)
               hoặc nếu đã qua dataloader xử lý thì có thể là (Batch, Time, 3, 224, 224)
        """
        
        # --- A. Xử lý Text (Prompt Learning) ---
        prompts = self.prompt_learner()                # Tạo learnable prompts
        tokenized_prompts = self.prompt_learner.tokenized_prompts # Lấy token cứng
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        # Normalize Text Features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # --- B. Xử lý Image (Visual Features) ---
        
        # 1. Trích xuất đặc trưng từ CLIP Image Encoder
        # Input shape: (B, T, C, H, W) -> Cần gộp B và T để đưa qua CLIP
        
        # -- Xử lý FACE --
        face_imgs = imgs['face'] # (B, T, C, H, W)
        b, t, c, h, w = face_imgs.size()
        face_imgs = face_imgs.view(-1, c, h, w) # (B*T, C, H, W)
        
        with torch.no_grad():
            face_features_raw = self.clip_model.visual(face_imgs) # (B*T, feature_dim)
        face_features_raw = face_features_raw.view(b, t, -1).float() # (B, T, feature_dim)

        # -- Xử lý BODY --
        body_imgs = imgs['body'] # (B, T, C, H, W)
        body_imgs = body_imgs.view(-1, c, h, w)
        
        with torch.no_grad():
            body_features_raw = self.clip_model.visual(body_imgs)
        body_features_raw = body_features_raw.view(b, t, -1).float()

        # 2. Qua lớp S-ATT Số 1 (Giữ lại Sequence)
        # YÊU CẦU: model Temporal_Transformer_Cls phải có tham số return_sequence
        face_feat_seq = self.visual_transformer_stage1(face_features_raw, return_sequence=True)
        body_feat_seq = self.visual_transformer_stage1(body_features_raw, return_sequence=True)
        
        # 3. Cộng Type Embedding
        device = face_feat_seq.device
        
        # Tạo vector id cho Face (toàn số 0) và Body (toàn số 1)
        # Shape của embedding sẽ broadcast tự động vào (B, T, D)
        type_face = torch.zeros((1, 1), dtype=torch.long, device=device) # Index 0
        type_body = torch.ones((1, 1), dtype=torch.long, device=device)  # Index 1
        
        embed_face = self.type_embedding(type_face) # (1, 1, D)
        embed_body = self.type_embedding(type_body) # (1, 1, D)
        
        # Cộng embedding vào feature
        face_feat_aug = face_feat_seq + embed_face
        body_feat_aug = body_feat_seq + embed_body
        
        # 4. Fusion (Cộng gộp đặc trưng)
        # Tại bước này, đặc trưng chứa cả thông tin hình ảnh gốc lẫn loại (face/body)
        
        fused_features = torch.cat([face_feat_aug, body_feat_aug], dim=1)  # (B, 2*T, D)
        # Trong forward
        # Giả sử fused_features có shape (Batch, Sequence_Length, Dim) = (B, T, 512)
        b, t, _ = fused_features.shape

        # 1. Nhân bản CLS token cho bằng số Batch
        cls_tokens = self.cls_token.expand(b, -1, -1)  # Shape: (B, 1, 512)

        # 2. Nối CLS vào trước chuỗi feature (Concatenate)
        # Input lúc này sẽ là: [CLS, Token1, Token2, ...]
        x = torch.cat((cls_tokens, fused_features), dim=1)  # Shape: (B, T+1, 512)

        # 3. Cho qua Transformer
        # Output trả về nguyên chuỗi đã encode
        out = self.visual_transformer_stage2(x)  # Shape: (B, T+1, 512)

        # 4. Chỉ lấy phần tử đầu tiên (vị trí 0) -> Đây chính là CLS output
        final_visual_features = out[:, 0, :]  # Shape: (B, 512)
        # 6. Normalize và Tính toán Output
        final_visual_features = self.ln_post(final_visual_features)
        final_visual_features = final_visual_features / final_visual_features.norm(dim=-1, keepdim=True)
        
        # Tính Cosine Similarity (Logits)
        # Scale logit bằng logit_scale_exp của CLIP
        logits = final_visual_features @ text_features.t() * self.clip_model.logit_scale.exp()
        return logits