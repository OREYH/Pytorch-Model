""" Vision Transformer (ViT) """

#### 라이브러리 호출 ####
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """ Multi-Head Self-Attention """
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim은 num_heads로 나누어 떨어져야 해요!")
        if batch_first is False:
            raise NotImplementedError('batch_first=False를 할 생각을 하다니. 제정신이 아니구만.')
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scaling   = self.head_dim ** -0.5
        
        ## Q / K / V Projection
        self.q_proj   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value):
        """ query/key/value: [B, seq_len, embed_dim] """
        
        ## 1. Linear Projection
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        ## 2. Split
        bsz, q_len, _ = q.size()
        _, k_len, _ = k.size()
        
        # [B, num_heads, seq_len, head_dim]
        q = self._reshape_to_heads(q, bsz, q_len)
        k = self._reshape_to_heads(k, bsz, k_len)
        v = self._reshape_to_heads(v, bsz, k_len)
        
        ## 3. Attention Score
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling # [B, num_heads, seq_len, seq_len]
        
        ## 4. Attention Weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        ## 5. Attention Outputs
        attn_output = torch.matmul(attn_weights, v)            # [B, num_heads, seq_len, head_dim]
        attn_output = self._merge_attention_heads(attn_output) # [B, seq_len, embed_dim]
        attn_output = self.out_proj(attn_output)               # [B, seq_len, embed_dim]
        attn_output = self.proj_dropout(attn_output)
        
        return attn_output, attn_weights
    
    def _reshape_to_heads(self, x:torch.Tensor, bsz, seq_len):
        ## x: [bsz, seq_len, embed_dim] -> [bsz, num_heads, seq_len, head_dim]
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous() # [bsz, num_heads, seq_len, embed_dim]
        
        return x
    
    def _merge_attention_heads(self, x:torch.Tensor):
        ## x: [bsz, num_heads, seq_len, head_dim] -> [bsz, seq_len, embed_dim]
        bsz, num_heads, seq_len, head_dim = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()         # [bsz, seq_len, num_heads, head_dim]
        x = x.view(bsz, seq_len, num_heads * head_dim) # [bsz, seq_len, embed_dim]
        
        return x

class PatchEmbedConv(nn.Module):
    """ Conv를 이용한 Patch Embedding """
    def __init__(self, img_size=32, patch_size=4, embed_dim=192):
        super().__init__()
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        ## Conv2d를 이용해 Patch Embedding
        self.proj = nn.Conv2d(in_channels=3, out_channels=embed_dim, 
                              kernel_size=patch_size, stride=patch_size,
                              padding=0)
    
    def forward(self, x:torch.Tensor):
        ## x: [B, 3, H, W] -> [B, num_patches, embed_dim]
        x = self.proj(x)                  # [B, embed_dim, H', W']
        x = torch.flatten(x, start_dim=2) # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)
        
        return x

class AddClsToken(nn.Module):
    """ CLS Token 추가 """
    def __init__(self, embed_dim, num_patches):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_patches = num_patches
        
        ## CLS Token(학습 파라미터)
        # [1, 1, embed_dim]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.002) # 정규분포를 따르나 [a, b] 사이 값으로 나타냄
    
    def forward(self, x):
        ## x: [B, num_patches, embed_dim]
        bsz = x.size(0)
        cls = self.cls_token.expand(bsz, 1, self.embed_dim) # [B, 1, embed_dim]
        x = torch.cat([cls, x], dim=1) # [B, 1+num_patches, embed_dim]
        
        return x
    
class PositionalEmbedding(nn.Module):
    """ Trainable Position Embedding 추가 """
    def __init__(self, embed_dim, num_tokens, dropout=0.1):
        super().__init__()
        # [1, num_tokens, embed_dim]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x:torch.Tensor):
        ## x: [B, num_patches+1, embed_dim]
        x = x + self.pos_embed
        x = self.dropout(x)
        
        return x

class EncoderBlock(nn.Module):
    """ ViT Encoder Block """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        # Transformer의 dim_feedforward에 해당함
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout))
    
    def forward(self, x):
        ## Self Attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop1(attn_out)
        ## MLP
        x = x + self.mlp(self.norm2(x))
        
        return x

class Encoder(nn.Module):
    """ ViT Encoder """
    def __init__(self, depth, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x:torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        
        return x

class VisionTransformer(nn.Module):
    """ ViT Classification Model """
    def __init__(self, img_size=32, patch_size=4, num_classes=10, embed_dim=192, depth=6, 
                    num_heads=3, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        ## Patch Embedding
        self.patch_embed = PatchEmbedConv(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        ## CLS + Position Embedding
        self.add_cls   = AddClsToken(embed_dim, self.num_patches)
        self.pos_embed = PositionalEmbedding(embed_dim, num_tokens=self.num_patches+1, dropout=dropout)
        
        ## Encoder
        self.encoder = Encoder(depth=depth, embed_dim=embed_dim, num_heads=num_heads,
                               mlp_ratio=mlp_ratio, dropout=dropout, attn_dropout=attn_dropout)
        
        ## Head
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x:torch.Tensor):
        ## Patch Embedding
        x = self.patch_embed(x) # [B, num_patches, embed_dim]
        ## CLS 추가 + Position Embedding
        x = self.add_cls(x)     # [B, num_patches+1, embed_dim]
        x = self.pos_embed(x)   # [B, num_patches+1, embed_dim]
        ## Encoder
        x = self.encoder(x)     # [B, num_patches+1, embed_dim]
        ## CLS Token으로 Classification
        cls_token = x[:, 0]     # [B, embed_dim]
        logits = self.head(cls_token)
        
        return logits