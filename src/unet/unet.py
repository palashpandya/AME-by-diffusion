import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        # Using a flexible number of groups to avoid errors if channels are low
        num_groups = min(32, out_ch)
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        
        self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):    
        h = self.conv1(F.silu(self.norm1(x)))
        # Inject time
        h = h + self.time_mlp(F.silu(t))[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        num_groups = min(32, channels)
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h = x.shape
        qkv = self.qkv(self.norm(x)).view(b, 3, c, h)  # (b, 3, c, h)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Scaling dot product for stability
        attn = torch.einsum('bcn,bcm->bnm', q, k) * (c ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.einsum('bnm,bcm->bcn', attn, v).view(b, c, h)
        return x + self.proj(out)

class GeneralDynamicUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = config.D
        self.n = config.N
        time_emb_dim = 128
        
        # Define channel progression: e.g., [64, 128, 256...]
        base_ch = 16
        # chs = [min(1024, base_ch * (2**i)) for i in range(self.n + 1)]
        chs = [ base_ch * (2**i) for i in range(self.n + 1)]
        # 1. Initial Projection (Fixes the "1-channel GroupNorm" error)
        self.init_conv = nn.Conv1d(2, chs[0], kernel_size=3, padding=1)
        
        # 2. Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 3. Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        for i in range(self.n):
            # In each level, process features then downsample
            self.encoders.append(ResBlock(chs[i], chs[i], time_emb_dim))
            self.downs.append(nn.Conv1d(chs[i], chs[i+1], kernel_size=self.d, stride=self.d))

        # 4. Bottleneck
        mid_ch = chs[self.n]
        self.mid_res1 = ResBlock(mid_ch, mid_ch, time_emb_dim)
        self.mid_attn = SelfAttention(mid_ch)
        self.mid_res2 = ResBlock(mid_ch, mid_ch, time_emb_dim)

        # 5. Decoder
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        for i in reversed(range(self.n)):
            # Upsample back to the size of the skip connection
            self.ups.append(nn.ConvTranspose1d(chs[i+1], chs[i+1], kernel_size=self.d, stride=self.d))
            # Input to ResBlock = Upsampled Channels + Skip Connection Channels
            self.decoders.append(ResBlock(chs[i+1] + chs[i], chs[i], time_emb_dim))

        # Output 2 channels to match complex real/imag representation
        self.final_conv = nn.Conv1d(chs[0], 2, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        # Initial project
        x = self.init_conv(x)
        
        # Encoder
        skips = []
        for i in range(self.n):
            x = self.encoders[i](x, t_emb)
            skips.append(x) # Store for the "U" bridge
            x = self.downs[i](x)
            
        # Bottleneck
        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, t_emb)
        
        # Decoder
        for i in range(self.n):
            x = self.ups[i](x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1) # The skip connection happens here
            x = self.decoders[i](x, t_emb)
            
        return self.final_conv(x)