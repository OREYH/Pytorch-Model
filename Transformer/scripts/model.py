""" Transformer Model """

#### 라이브러리 호출 ####
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=True):
        """
        - embed_dim: d_model
        - num_heads: h
        - head_dim : d_k
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim은 num_heads로 나누어 떨어져야 합니다.")
        if batch_first == False:
            raise NotImplementedError("batch_first=False와 관련된 코드는 아직 구현되지 않았습니다.")
        
        self.embed_dim   = embed_dim
        self.num_heads   = num_heads
        self.head_dim    = embed_dim // num_heads
        self.batch_first = batch_first
        self.scaling     = self.head_dim ** -0.5
        
        ## Query / Key / Value projection (Linear Layer)
        self.q_proj   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        query    : [B, L, E]
        key/value: [B, S, E]
        attn_mask: [L, S] / key_padding_mask: [B, S]
        """
        ## Linear Layer 계산
        q, k, v = self._project_qkv(query, key, value)

        ## Attention Score 계산
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling # [B, H, L, S]
        
        ## Look-Ahead Mask 계산
        if attn_mask is not None:
            attn_mask = self._prepare_attn_mask(attn_mask, scores) # [B, 1, L, S]
            scores = scores + attn_mask # [B, H, L, S]
        
        ## Key Padding Mask 계산
        if key_padding_mask is not None:
            padding_mask = self._prepare_key_padding_mask(key_padding_mask, scores)
            scores = scores.masked_fill(padding_mask, float('-inf'))
        
        ## Attention Weights(=Value) 계산
        attn_weights = torch.softmax(scores, dim=-1) # [B, H, L, S]
        attn_weights = self.attn_dropout(attn_weights)
        
        ## 최종 출력 계산
        attn_output = torch.matmul(attn_weights, v) # [B, H, L, head_dim]
        attn_output = self._merge_attention_heads(attn_output) # [B, L, E]
        attn_output = self.out_proj(attn_output) # [B, L, E]
        
        return attn_output, attn_weights
        
    def _project_qkv(self, query, key, value):
        """ Linear Projection """
        q = self.q_proj(query) # [B, L, E]
        k = self.k_proj(key)   # [B, S, E]
        v = self.v_proj(value) # [B, S, E]
        
        bsz, q_len, _ = q.size()
        _  , k_len, _ = k.size()
        
        q = self._reshape_to_heads(q, bsz, q_len) # [B, H, L, d_k]
        k = self._reshape_to_heads(k, bsz, k_len) # [B, H, S, d_k]
        v = self._reshape_to_heads(v, bsz, k_len) # [B, H, S, d_k]
        
        return q, k, v
    
    def _reshape_to_heads(self, x:torch.Tensor, bsz, seq_len):
        """ [B, L or S, E] 형태의 벡터를 [B, H, L or S, d_k] 형태로 변환 """
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous() # [B, H, L or S, d_k]
        
        return x
    
    def _merge_attention_heads(self, x:torch.Tensor):
        """ [B, H, L or S, d_k] 형태의 벡터를 [B, L or S, E] 형태로 변환 """
        bsz, num_heads, seq_len, head_dim = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, seq_len, num_heads * head_dim) # [B, L or S, E]
        
        return x
    
    def _prepare_attn_mask(self, mask:torch.Tensor, scores:torch.Tensor):
        """ Look-Ahead Mask를 [B, 1, L, S] 형태로 변환 """
        ## mask: [L, S], [B, L, S]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(scores.dtype) # [B, 1, L, S]
        
        return mask 
    
    def _prepare_key_padding_mask(self, key_padding_mask:torch.Tensor, scores:torch.Tensor):
        """ 패딩 토큰 위치를 무시하도록 하는 Key Padding Mask를 [B, H, L, S] 형태로 변환 """
        ## key_padding_mask: [B, S]
        padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, S]
        padding_mask = padding_mask.expand(-1, self.num_heads, scores.size(-2), -1) # [B, H, L, S]

        return padding_mask

class PositionalEncoding(nn.Module):
    """ Sinusoidal Positional Encoding """
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        
        return self.dropout(x)

class EncoderLayer(nn.Module):
    """ Transformer Encoder Layer """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, norm_first=False):
        super().__init__()
        
        ## Attention Module
        self.self_attn = MultiHeadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.dropout   = nn.Dropout(dropout)
        
        ## FFL Module
        self.linear1    = nn.Linear(d_model, dim_feedforward)
        self.linear2    = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)
        
        ## LayerNorm Module
        self.norm1      = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2      = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if self.norm_first:
            ## Multi-Head Attention 계산
            src = src + self._self_attention(self.norm1(src), src_mask, src_key_padding_mask)
            ## FFL & LayerNorm 계산
            src = src + self._feed_forward(self.norm2(src))
        else:
            ## Multi-Head Attention 계산
            src = self.norm1(src + self._self_attention(src, src_mask, src_key_padding_mask))
            ## FFL 계산
            src = self.norm2(src + self._feed_forward(src))
        
        return src
    
    def _self_attention(self, x, src_mask, src_key_padding_mask):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        return self.dropout(attn_output)
    
    def _feed_forward(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)

class DecoderLayer(nn.Module):
    """ Transformer Decoder Layer """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, norm_first=False):
        super().__init__()
        
        ## Attention Module
        self.self_attn     = MultiHeadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.dropout_attn1 = nn.Dropout(dropout)
        self.cross_attn    = MultiHeadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.dropout_attn2 = nn.Dropout(dropout)
        
        ## FFL Module
        self.linear1    = nn.Linear(d_model, dim_feedforward)
        self.linear2    = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)
        
        ## LayerNorm Module
        self.norm1      = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2      = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3      = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        ## memory: Encoder의 Context
        if self.norm_first:
            ## Masked Multi-Head Attention 계산
            tgt = tgt + self._self_attention(self.norm1(tgt), tgt_mask, tgt_key_padding_mask)
            ## Cross Multi-Head Attention 계산
            tgt = tgt + self._cross_attention(self.norm2(tgt), memory, memory_mask, memory_key_padding_mask)
            ## FFL 계산
            tgt = tgt + self._feed_forward(self.norm3(tgt))
        else:
            ## Masked Multi-Head Attention 계산
            tgt = self.norm1(tgt + self._self_attention(tgt, tgt_mask, tgt_key_padding_mask))
            ## Cross Multi-Head Attention 계산
            tgt = self.norm2(tgt + self._cross_attention(tgt, memory, memory_mask, memory_key_padding_mask))
            ## FFL 계산
            tgt = self.norm3(tgt + self._feed_forward(tgt))
        
        return tgt
    
    def _self_attention(self, x, tgt_mask, tgt_key_padding_mask):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        return self.dropout_attn1(attn_output)
    
    def _cross_attention(self, x, memory, memory_mask, memory_key_padding_mask):
        attn_output, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        return self.dropout_attn2(attn_output)
    
    def _feed_forward(self, x):
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)

class Encoder(nn.Module):
    """ Transformer Encoder """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, norm_first=False):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, 
                         dropout=dropout, layer_norm_eps=layer_norm_eps, 
                         norm_first=norm_first) 
                                    for _ in range(num_layers)])
        self.norm   = nn.LayerNorm(d_model, eps=layer_norm_eps)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        return self.norm(output)

class Decoder(nn.Module):
    """ Transformer Decoder """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, norm_first=False):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout,
                         layer_norm_eps, norm_first) for _ in range(num_layers)])
        self.norm   = nn.LayerNorm(d_model, eps=layer_norm_eps)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask,
                           tgt_key_padding_mask, memory_key_padding_mask)
        
        return self.norm(output)

class Transformer(nn.Module):
    """ 한국어-일본어 기계번역용 Transformer """
    def __init__(self, vocab_size, pad_id, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_seq_len=256,
                 layer_norm_eps=1e-5, norm_first=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id     = pad_id
        self.d_model    = d_model
        
        ## Token Embedding + Positional Encoding
        self.embedding   = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout,
                                              max_len=max_seq_len)
        self.pos_decoder = PositionalEncoding(d_model, dropout=dropout,
                                              max_len=max_seq_len)
        
        ## Encoder / Decoder
        self.encoder = Encoder(d_model, nhead, num_encoder_layers,
                               dim_feedforward=dim_feedforward, dropout=dropout,
                               layer_norm_eps=layer_norm_eps, norm_first=norm_first)
        self.decoder = Decoder(d_model, nhead, num_decoder_layers,
                               dim_feedforward=dim_feedforward, dropout=dropout,
                               layer_norm_eps=layer_norm_eps, norm_first=norm_first)
        
        ## Generator
        self.generator = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, src_ids:torch.Tensor, tgt_ids:torch.Tensor, 
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        src_ids: [B, src_len]
        tgt_ids: [B, tgt_len]
        """
        if src_key_padding_mask is None:
            src_key_padding_mask = src_ids.eq(self.pad_id)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = tgt_ids.eq(self.pad_id)
        
        ## Token Embedding + Positional Encoding
        src_emb = self.embedding(src_ids) * math.sqrt(self.d_model) # [B, src_len, d_model]
        src_emb = self.pos_encoder(src_emb)
        ## Encoder
        memory  = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        ## Token Embedding + Positional Encoding
        tgt_emb = self.embedding(tgt_ids) * math.sqrt(self.d_model) # [B, tgt_len, d_model]
        tgt_emb = self.pos_decoder(tgt_emb)
        
        ## Look-Ahead Mask (Casual Mask)
        tgt_len  = tgt_ids.size(1)
        tgt_mask = self.generate_sqaure_casual_mask(tgt_len, tgt_ids.device)
        
        ## Decoder
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)
        
        ## Generator
        logits = self.generator(output) # [B, tgt_len, vocab_size]
        return logits
        
    def generate_sqaure_casual_mask(self, size, device):
        """ Decoder용 casual mask 생성 """
        mask = torch.triu(torch.ones((size, size), device=device) * float('-inf'), diagonal=1)
        return mask