import copy
import torch.nn as nn
import torch
from ..builder import PROJECTIONS
from timm.models.layers import DropPath, trunc_normal_,to_2tuple
import torch.nn.functional as F


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input with shape: [B,D,C]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if mask is not None:
            # mask: B, N
            mask = mask.unsqueeze(1).unsqueeze(2)  # B,1,1,N
            attn = attn.masked_fill(~mask, -1e4)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Encoder(nn.Module):

    """Transformer_1D encoder"""

    def __init__(self, encoder_layer, num_layers):
        """

        Args:
            encoder_layer (TODO): TODO
            num_layers (TODO): TODO


        """
        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """forward function

        Args:
            x (torch.Tensor): input with shape [B,T,C]

        Returns: torch.Tensor. The same shape as input: [B,T,C]

        """
        for mod in self.layers:
            x = mod(x, mask)
        return x



class EncoderLayer1D(nn.Module):
    """swin transformer 1D"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len=-1,  
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        """TODO: to be defined."""
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)

        self.max_seq_len=max_seq_len  
        if self.max_seq_len == -1:
            self.attn = Attention(dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop,)
        else:
            self.attn = WindowAttention1D(  
                dim,
                max_seq_len=max_seq_len,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, mask=mask)  # ✅ 传 mask 给 Attention
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@PROJECTIONS.register_module()
class Transformer1DRelPos(nn.Module):

    """Docstring for Transformer1DRelPos."""

    def __init__(self,in_channels,out_channels, encoder_layer_cfg: dict, num_layers: int, **kwargs):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        encoder_layer = EncoderLayer1D(**encoder_layer_cfg)
        self.encoder = Encoder(encoder_layer, num_layers)

    def init_weights(self):
        pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): [B, C, D]
            mask (torch.Tensor, optional): [B, D] or [B, 1, D]
        Returns:
            torch.Tensor: [B, C, D]
        """
        # [B, C, D] -> [B, D, C]
        x = x.permute(0, 2, 1)
        # forward through encoder
        x = self.encoder(x, mask)
        # [B, D, C] -> [B, C, D]
        return x.permute(0, 2, 1)
        
    print(f"✅ Transformer1DRelPos registered from {__file__}")

