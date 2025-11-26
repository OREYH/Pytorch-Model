""" DDPM U-Net Model """

#### 라이브러리 호출 ####
import math
import torch
import torch.nn as nn

def make_group_norm(channels, num_groups=32):
    """ GroupNorm 채널 수 자동 조정 
    만약 C: 48, G: 32
    groups = 48
    48 % 32 = 16 -> 48 % 24 = 0 될 때까지 groups 감소
    """
    groups = min(num_groups, channels)
    while channels % groups != 0:
        groups -=1 
    
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
    def __init__(self, in_channels, out_channels, )