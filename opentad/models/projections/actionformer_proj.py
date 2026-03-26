import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_,to_2tuple

from ..bricks import ConvModule, TransformerBlock
from ..builder import PROJECTIONS



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
        print(f"After encoder x: {x.shape}, mask: {mask.shape}")
        # [B, D, C] -> [B, C, D]
        x = x.permute(0, 2, 1)
        print(f"After permute 2: {x.shape}, mask: {mask.shape}")
        return x, mask 
        
    print(f"✅ Transformer1DRelPos registered from {__file__}")


@PROJECTIONS.register_module()
class Conv1DTransformerProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
        conv_cfg=None,  # kernel_size proj_pdrop
        norm_cfg=None,
        attn_cfg=None,  # n_head n_mha_win_size attn_pdrop
        path_pdrop=0.0,  # dropout rate for drop path
        use_abs_pe=False,  # use absolute position embedding
        max_seq_len=2304,
        input_pdrop=0.0, # drop out the input feature
        transformer1d_rel_pos_cfg=None,  
    ):
        super().__init__()
        assert len(arch) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.proj_pdrop = conv_cfg["proj_pdrop"]
        self.scale_factor = 2  # as default
        self.n_mha_win_size = attn_cfg["n_mha_win_size"]
        self.n_head = attn_cfg["n_head"]
        self.attn_pdrop = 0.0  # as default
        self.path_pdrop = path_pdrop
        self.with_norm = norm_cfg is not None
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len

        self.transformer1d_rel_pos = None
        self.transformer1d_rel_pos = Transformer1DRelPos(
                in_channels=768,
                out_channels=768,
                encoder_layer_cfg=transformer1d_rel_pos_cfg["encoder_layer_cfg"],
                num_layers=transformer1d_rel_pos_cfg["num_layers"],
            )

        self.input_pdrop = nn.Dropout1d(p=input_pdrop) if input_pdrop > 0 else None

        if isinstance(self.n_mha_win_size, int):
            self.mha_win_size = [self.n_mha_win_size] * (1 + arch[-1])
        else:
            assert len(self.n_mha_win_size) == (1 + arch[-1])
            self.mha_win_size = self.n_mha_win_size

        if isinstance(self.in_channels, (list, tuple)):
            assert isinstance(self.out_channels, (list, tuple)) and len(self.in_channels) == len(self.out_channels)
            self.proj = nn.ModuleList([])
            for n_in, n_out in zip(self.in_channels, self.out_channels):
                self.proj.append(
                    ConvModule(
                        n_in,
                        n_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            in_channels = out_channels = sum(self.out_channels)
        else:
            self.proj = None

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embed)
        if self.use_abs_pe:
            pos_embed = get_sinusoid_encoding(self.max_seq_len, out_channels) / (out_channels**0.5)
            self.register_buffer("pos_embed", pos_embed, persistent=False)

        # embedding network using convs
        self.embed = nn.ModuleList()
        for i in range(arch[0]):
            self.embed.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="relu"),
                )
            )

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    out_channels,
                    self.n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=self.attn_pdrop,
                    proj_pdrop=self.proj_pdrop,
                    path_pdrop=self.path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    out_channels,
                    self.n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=self.attn_pdrop,
                    proj_pdrop=self.proj_pdrop,
                    path_pdrop=self.path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, sequence length (bool)
        # feature projection
        print(f"Before conv1dproj - x shape: {x.shape}, mask shape: {mask.shape}")
        if self.transformer1d_rel_pos is not None:
            x, mask = self.transformer1d_rel_pos(x, mask)  # 调用Transformer1DRelPos处理
        print(f"After transformer1drelpos - x shape: {x.shape}, mask shape: {mask.shape}")
        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        # drop out input if needed
        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

        # embedding network
        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max length."
            pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x,)
        out_masks = (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks


def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)
