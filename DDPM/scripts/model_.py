""" DDPM U-Net Model """

#### 라이브러리 호출 ####
import math
import torch
import torch.nn as nn

def make_norm(channels, num_groups=32):
    """ GroupNorm 채널 수 자동 조정 """
    groups = min(num_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)

def sinusoidal_embedding(timesteps, dim, max_period=10000):
    """ Sinusoidal Time Embedding """
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(timesteps)
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    device = timesteps.device
    half_dim = dim // 2
    freq = torch.exp(torch.arange(half_dim, device=device) * -(math.log(max_period) / max(1, half_dim - 1)))
    args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class TimeEmbedding(nn.Module):
    """ Time Step을 위한 Embedding """
    def __init__(self, model_channels, time_embed_dim=None, max_period=10000):
        super().__init__()
        if time_embed_dim is None:
            time_embed_dim = model_channels * 4
        self.model_channels = model_channels
        self.max_period = max_period
        self.linear1 = nn.Linear(model_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(time_embed_dim, time_embed_dim)
    
    def forward(self, timesteps):
        ## input: timesteps [B] / output: [B, time_embed_dim]
        emb = sinusoidal_embedding(timesteps, self.model_channels, self.max_period)  # [B, model_channels]
        emb = self.linear1(emb)                                                      # [B, time_embed_dim]
        emb = self.act(emb)                                                          # [B, time_embed_dim]
        emb = self.linear2(emb)                                                      # [B, time_embed_dim]
        return emb  # [B, time_embed_dim]

class ResBlock(nn.Module):
    """ ResNet Block + Time Embedding """
    def __init__(self, in_channels, out_channels, time_embed_dim, dropout=0.0, num_groups=32):
        super().__init__()
        self.norm1 = make_norm(in_channels, num_groups)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
        self.norm2 = make_norm(out_channels, num_groups)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        ## input: x [B, in_channels, H, W], time_emb [B, time_embed_dim]
        ## output: [B, out_channels, H, W]
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        temb = self.time_proj(time_emb)
        temb = temb.view(temb.size(0), temb.size(1), 1, 1)
        h = h + temb
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """ Self-Attention Block """
    def __init__(self, channels, num_heads=4, num_groups=32):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels는 num_heads로 나누어 떨어져야 합니다.")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = make_norm(channels, num_groups)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        ## input/output: x [B, C, H, W]
        b, c, h, w = x.size()
        h_in = self.norm(x)
        
        q = self.q(h_in).view(b, self.num_heads, self.head_dim, h * w)
        k = self.k(h_in).view(b, self.num_heads, self.head_dim, h * w)
        v = self.v(h_in).view(b, self.num_heads, self.head_dim, h * w)
        
        q = q.permute(0, 1, 3, 2) # [B, heads, HW, head_dim]
        attn = torch.matmul(q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        
        v = v.permute(0, 1, 3, 2)   # [B, heads, HW, head_dim]
        out = torch.matmul(attn, v) # [B, heads, HW, head_dim]
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        out = self.proj(out)
        
        return x + out

class Downsample(nn.Module):
    """ Strided Convolution Downsample """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        ## input: [B, C, H, W] / output: [B, C, H//2, W//2]
        return self.conv(x)

class Upsample(nn.Module):
    """ Nearest Neighbor Upsample + Conv """
    def __init__(self, channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        ## input: [B, C, H, W] / output: [B, C, H*2, W*2]
        x = self.upsample(x)
        x = self.conv(x)
        return x

class DDPMUNet(nn.Module):
    """ DDPM 용 U-Net (128x128) """
    def __init__(self, sample_size=128, num_in_channels=3, num_out_channels=3, 
                 model_channels=128, channel_mult=(1, 2, 2, 4), num_res_blocks=2, 
                 attention_resolutions=(16,), num_heads=4, dropout=0.0, 
                 num_train_timesteps=1000, beta_schedule='linear'):
        super().__init__()
        self.sample_size           = sample_size                # 입력 이미지 크기 (예: 128)
        self.num_in_channels       = num_in_channels            # 입력 채널 수(예: RGB=3)
        self.num_out_channels      = num_out_channels           # 출력 채널 수(일반적으로 입력 채널과 동일)
        self.model_channels        = model_channels             # U-Net 기본 채널 폭
        self.channel_mult          = channel_mult               # 해상도 단계별 채널 multiplier
        self.num_res_blocks        = num_res_blocks             # 각 해상도 스테이지의 ResBlock 개수
        self.attention_resolutions = set(attention_resolutions) # self-attention을 넣을 해상도
        self.num_heads             = num_heads                  # attention head 개수
        self.dropout               = dropout                    # ResBlock 내부 dropout 비율
        self.num_train_timesteps   = num_train_timesteps        # 학습 시 사용한 총 타임스텝 수
        self.beta_schedule         = beta_schedule              # 베타 스케줄 형태(linear 등)
        
        self.time_embed_dim = model_channels * 4                # time embed dim
        # time embedding 모듈
        self.time_embed = TimeEmbedding(model_channels, time_embed_dim=self.time_embed_dim) 
        # 첫 입력 conv
        self.input_conv = nn.Conv2d(num_in_channels, model_channels, kernel_size=3, padding=1) 
        
        ## Down Path
        self.down_blocks = nn.ModuleList()
        self.skip_channels = [model_channels] # input conv output
        curr_res = sample_size
        in_channels = model_channels
        
        for level, mult in enumerate(channel_mult):
            out_channels = model_channels * mult
            res_layers = nn.ModuleList()
            attn_layers = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                res_layers.append(ResBlock(in_channels, out_channels, self.time_embed_dim, dropout=dropout))
                if curr_res in self.attention_resolutions:
                    attn_layers.append(AttentionBlock(out_channels, num_heads=num_heads))
                else:
                    attn_layers.append(nn.Identity())
                in_channels = out_channels
                self.skip_channels.append(out_channels)
            
            if level != len(channel_mult) - 1:
                downsample = Downsample(in_channels)
                self.skip_channels.append(in_channels)
                curr_res = curr_res // 2
            else:
                downsample = nn.Identity()
            
            self.down_blocks.append(nn.ModuleDict({
                'resblocks': res_layers,
                'attentions': attn_layers,
                'downsample': downsample
            }))
        
        ## Middle Block
        self.mid_block1 = ResBlock(in_channels, in_channels, self.time_embed_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(in_channels, num_heads=num_heads)
        self.mid_block2 = ResBlock(in_channels, in_channels, self.time_embed_dim, dropout=dropout)
        
        ## Up Path
        self.up_blocks = nn.ModuleList()
        skip_channels = list(self.skip_channels)
        for level in reversed(range(len(channel_mult))):
            out_channels = model_channels * channel_mult[level]
            res_layers = nn.ModuleList()
            attn_layers = nn.ModuleList()
            
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                res_layers.append(ResBlock(in_channels + skip_ch, out_channels, self.time_embed_dim, dropout=dropout))
                if (sample_size // (2 ** level)) in self.attention_resolutions:
                    attn_layers.append(AttentionBlock(out_channels, num_heads=num_heads))
                else:
                    attn_layers.append(nn.Identity())
                in_channels = out_channels
            
            if level != 0:
                upsample = Upsample(in_channels)
            else:
                upsample = nn.Identity()
            
            self.up_blocks.append(nn.ModuleDict({
                'resblocks': res_layers,
                'attentions': attn_layers,
                'upsample': upsample
            }))
        
        self.out_norm = make_norm(in_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_channels, num_out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, timesteps, return_dict=False):
        """
        input: x [B, num_in_channels, H, W], timesteps [B]
        output: [B, num_out_channels, H, W]
        """
        hs = []
        h = self.input_conv(x)
        hs.append(h)
        
        t_emb = self.time_embed(timesteps)
        
        ## Down Path
        for level, block in enumerate(self.down_blocks):
            for res_layer, attn_layer in zip(block['resblocks'], block['attentions']):
                h = res_layer(h, t_emb)
                h = attn_layer(h)
                hs.append(h)
            if level != len(self.down_blocks) - 1:
                h = block['downsample'](h)
                hs.append(h)
        
        ## Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        ## Up Path
        for level, block in enumerate(self.up_blocks):
            for res_layer, attn_layer in zip(block['resblocks'], block['attentions']):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = res_layer(h, t_emb)
                h = attn_layer(h)
            if level != len(self.up_blocks) - 1:
                h = block['upsample'](h)
        
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        
        if return_dict:
            return {'sample': h}
        return h
    
    @classmethod
    def from_config(cls, config):
        return build_model(config)

def build_model(config=None):
    """ HuggingFace diffusers DDPMPipeline와 호환되는 설정으로 모델 생성 """
    if config is None:
        config = {}
    
    sample_size = config.get('sample_size', config.get('image_size', 128))
    num_in_channels = config.get('num_in_channels', config.get('in_channels', 3))
    num_out_channels = config.get('num_out_channels', config.get('out_channels', num_in_channels))
    model_channels = config.get('model_channels', config.get('base_channels', config.get('num_channels', 128)))
    channel_mult = tuple(config.get('channel_mult', config.get('channel_multiplier', (1, 2, 2, 4))))
    num_res_blocks = config.get('num_res_blocks', config.get('layers_per_block', 2))
    attention_resolutions = tuple(config.get('attention_resolutions', config.get('attn_resolutions', (16,))))
    num_heads = config.get('num_heads', config.get('num_head_channels', 4))
    dropout = config.get('dropout', 0.0)
    num_train_timesteps = config.get('num_train_timesteps', 1000)
    beta_schedule = config.get('beta_schedule', 'linear')
    
    return DDPMUNet(
        sample_size=sample_size,
        num_in_channels=num_in_channels,
        num_out_channels=num_out_channels,
        model_channels=model_channels,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        dropout=dropout,
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule
    )

def get_model(config=None):
    """ build_model 별칭 """
    return build_model(config)
