import torch
from einops import rearrange, repeat
from torch import nn, einsum
import math

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # Đảm bảo dim là int
        self.norm = nn.LayerNorm(int(dim))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # Đảm bảo các dimension là int
        dim = int(dim)
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim),
                                 GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        # Đảm bảo dimension là int
        dim = int(dim)
        heads = int(heads)
        dim_head = int(dim_head)
        
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)               
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                                              Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

###########################################################
#############      output = class tokens      #############
###########################################################
class Temporal_Transformer_Cls(nn.Module):
    def __init__(self, num_patches, input_dim, depth, heads, mlp_dim, dim_head):
        super().__init__()
        dropout = 0.1
        self.num_patches = int(num_patches) # Ép kiểu int
        self.input_dim = int(input_dim)     # Ép kiểu int
        
        # Token đại diện cho cả chuỗi (Class Token)
        # Sửa lỗi ở đây: đảm bảo size là tuple of ints
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.input_dim))
        
        # Positional Embedding (cộng thêm 1 cho CLS token)
        # Sửa lỗi ở đây: num_patches + 1 cũng phải là int
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.input_dim))
        
        self.temporal_transformer = Transformer(self.input_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x, return_sequence=False):
        """
        Args:
            x: Input tensor (Batch, Tokens, Dim)
            return_sequence: 
                - True: Trả về chuỗi features gốc (bỏ CLS token) -> Dùng cho giai đoạn fusion
                - False: Trả về CLS token -> Dùng cho output cuối cùng
        """
        b, n, _ = x.shape
        
        # 1. Gắn thêm CLS token vào đầu chuỗi
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 2. Cộng Positional Embedding
        # Cắt pos_embedding theo chiều dài thực tế n+1 (phòng trường hợp n thay đổi)
        x = x + self.pos_embedding[:, :(n + 1)]
        
        # 3. Qua Transformer
        x = self.temporal_transformer(x)
        
        # 4. Xử lý đầu ra tùy theo mục đích
        if return_sequence:
            # Trả về chuỗi features (B, N, D), bỏ qua CLS token ở vị trí 0
            return x[:, 1:]
        else:
            # Trả về CLS token (B, D) đại diện cho toàn bộ chuỗi
            return x[:, 0]