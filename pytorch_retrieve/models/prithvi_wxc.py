"""
pytorch_retrieve.models.prithvi_wxc
===================================

Provides extensions of the PrithviWxC foundation model.

NOTE: Requires the PrithviWxC package to be installed.
"""
from importlib.metadata import version
from typing import Callable, Dict, Optional, Tuple, Union
from functools import cached_property

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
if version('torch') > '2.3.0':
    from torch.nn.attention import SDPBackend, sdpa_kernel

from pytorch_retrieve.modules.normalization import LayerNormFirst
from pytorch_retrieve.modules.conv.blocks import ResNeXtBlock
from pytorch_retrieve.modules.conv.padding import Reflect

try:
    import PrithviWxC
    import PrithviWxC.model as mdl
    from PrithviWxC.model import (
        PrithviWxC,
        PatchEmbed,
        LayerNormPassThrough,
        Transformer,
        _Shift,
        SWINShiftNoBuffer,
        SWINShift,
        version,
        Mlp

    )

except ImportError:
    raise ImportError(
        "Could not import the 'PrithviWxC' package. Please make sure it is installed."
    )

TORCH_VERSION = version('torch')


def drop_path(
    x: torch.Tensor,
    drop_prob: torch.Tensor,
    training: bool = False,
    scale_by_keep: bool = True
):
    """
    Drop path supporting variable dropping frequencies.
    """
    if (drop_prob == 0.0).all() or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )
    keep_mask = torch.bernoulli(keep_prob).to(dtype=x.dtype)
    if scale_by_keep and (keep_prob > 0.0).any():
        keep_mask.div_(torch.where(0.0 < keep_prob, keep_prob, torch.tensor(1.0, device=x.device, dtype=x.dtype)))
    return x * keep_mask.view(shape)


class LeadTimeDropPath(nn.Module):
    """
    Lead-time conditioned drop path.
    """
    def __init__(
            self,
            drop_probs: Union[float, Tuple[float, float]] = (0.0, 0.0),
            scale_by_keep=True
    ):
        super().__init__()
        if isinstance(drop_probs, float):
            drop_probs = (drop_probs, drop_probs)
        self.drop_probs = drop_probs
        self.scale_by_keep = scale_by_keep
        self.lead_time_min = 3
        self.lead_time_max = 10

    def forward(self, x: torch.Tensor, lead_time: torch.Tensor = None):
        """
        Apply dropout.
        """
        if not self.training:
            return x

        if lead_time is None:
            drop_prob = self.drop_probs[0] * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        else:
            rel_dist = (lead_time - self.lead_time_min) / (self.lead_time_max - self.lead_time_min)
            drop_prob = self.drop_probs[0] + (self.drop_probs[1] - self.drop_probs[0]) * rel_dist
            drop_prob = torch.clip(drop_prob, torch.zeros_like(drop_prob), torch.ones_like(drop_prob))

        return drop_path(x, drop_prob, self.training, self.scale_by_keep)


class Transformer(nn.Module):
    """
    Transformer for inputs of shape [..., S, features].
    """

    def __init__(
        self,
        features: int,
        mlp_multiplier: int,
        n_heads: int,
        dropout: float,
        drop_path: Union[float, Tuple[float, float]],
    ) -> None:
        """
        Args:
            features: Number of features for inputs to the layer.
            mlp_multiplier: Model will use features*mlp_multiplier hidden units.
            n_heads: Number of attention heads. Should be a factor of features.
                (I.e. the layer uses features // n_heads.)
            dropout: Dropout.
            drop_path: DropPath.
        """
        super().__init__()

        self.features = features
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        if isinstance(drop_path, (list, tuple)):
            self.drop_path = LeadTimeDropPath(drop_path)
        elif drop_path > 0.0:
            self.drop_path = LeadTimeDropPath((drop_path, drop_path))
        else:
            self.drop_path = None

        self.attention = nn.Sequential(
            LayerNormPassThrough(features),
            mdl.MultiheadAttention(features, n_heads, dropout),
        )

        self.ff = nn.Sequential(
            nn.LayerNorm(features),
            Mlp(
                features=features,
                hidden_features=features * mlp_multiplier,
                dropout=dropout,
            ),
        )

    def forward(
            self,
            d: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [..., sequence, features]
            lead_time: A tensor containing the lead time for each sample.
        Returns:
            Tensor of shape [..., sequence, features]
        """
        x, attn_mask, lead_time = d
        if not x.shape[-1] == self.features:
            raise ValueError(
                f"Expecting tensor with last dimension of size {self.features}."
            )

        attention_x = self.attention(d[:2])

        if self.drop_path is None:
            x = x + attention_x
            x = x + self.ff(x)
        else:
            x = x + self.drop_path(attention_x, lead_time=lead_time)
            x = x + self.drop_path(self.ff(x), lead_time=lead_time)

        return x



class InvertedBottleneckBlock(nn.Module):
    """
    Inverted-bottleneck block is used in MobileNet and Efficient net where it is referred
    to as MBConv
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion_factor: int = 4,
            kernel_size: int = 3,
            activation_factory: Callable[[], nn.Module] = nn.GELU,
            normalization_factory: Callable[[int], nn.Module] = LayerNormFirst,
            padding: Optional[Tuple[int]] = None,
            padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
            downsample: Optional[int] = None,
            fused: bool = False,
            stochastic_depth: Optional[float] = None,
    ):
        """
        Args:
            in_channels: The number of channels in the input tensor.
            out_channels: The number of channels in the output.
            expansion_factor: The number of channels in the inverted bottleneck is calculated by
                multiplying 'out_channels' with this expansion factor.
            kernel_size: The kernel size to use for spatial mixing.
            activation_factory: A factory functional to create the activation layers.
            normalization_factory: A factory functional to create the normalization layers.
            padding: The padding to apply before the spatial convolutions.
            padding_factory: A factory functional to create the padding layer.
            downsample: The downsampling to apply in the layer.
            fused: Whether or not to fuse the first two convolution layers.
            stochastic_depth: The probabilistic depth of the layer, i.e., the probability that the
                layer is applied to the input.
        """
        super().__init__()
        self.act = activation_factory()
        act = activation_factory()

        hidden_channels = out_channels * expansion_factor
        self.stochastic_depth = stochastic_depth

        stride = (1, 1)
        if downsample is not None:
            if isinstance(downsample, int):
                downsample = (downsample,) * 2
            if max(downsample) > 1:
                stride = downsample


        if in_channels == out_channels:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=stride,
                    stride=stride
                ),
                LayerNormFirst(out_channels)
            )

        padding = (kernel_size // 2,) * 2
        if 1 < max(stride):
            padding = 0

        blocks = []
        if not fused:
            blocks += [
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
                normalization_factory(hidden_channels),
                act
            ]


            blocks += [
                padding_factory(padding),
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size if max(stride) < 2 else stride,
                    stride=stride,
                    groups=hidden_channels,
                ),
                normalization_factory(hidden_channels),
                act
            ]
        else:

            blocks += [
                padding_factory(padding),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                normalization_factory(hidden_channels),
                act
            ]

        blocks += [
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            normalization_factory(out_channels),
            act
        ]
        self.body = nn.Sequential(*blocks)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate input through layer.
        """
        shortcut = self.projection(x)
        return shortcut + self.body(x)


class ObservationEncoder(nn.Module):
    """
    Convolutional encoder used to encode satellite observations and meta data.

    The observation encoder downsamples and encodes all satellite observation layers separately.
    """
    def __init__(
            self,
            n_meta_features: int,
            obs_patch_size: Tuple[int, int] = (4, 4),
            channels: Tuple[int, int] = (16, 32, 64)
    ):
        """
        Observation encoder for PrithviWxC model.
        """
        super().__init__()
        self.n_meta_features = n_meta_features

        patching = tuple([sze // 2 for sze in obs_patch_size])
        channels_s1, channels_s2, channels_s3 = channels
        self.channels = channels
        self.patch_height, self.patch_width = obs_patch_size

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(1, channels_s1, kernel_size=patching, stride=patching),
            LayerNormFirst(channels_s1),
            InvertedBottleneckBlock(channels_s1, channels_s2, expansion_factor=1, downsample=(2, 2)),
            InvertedBottleneckBlock(channels_s2, channels_s3, expansion_factor=2),
            InvertedBottleneckBlock(channels_s3, channels_s3),
        )
        self.meta_encoder = nn.Sequential(
            nn.Conv2d(n_meta_features, channels_s1, kernel_size=patching, stride=patching),
            LayerNormFirst(channels_s1),
            InvertedBottleneckBlock(channels_s1, channels_s2, expansion_factor=1, downsample=(2, 2)),
            InvertedBottleneckBlock(channels_s2, channels_s3, expansion_factor=2),
            InvertedBottleneckBlock(channels_s3, channels_s3),
        )
        self.mask_encoder = nn.MaxPool2d(kernel_size=obs_patch_size, stride=obs_patch_size)


    def forward(
            self,
            obs: torch.Tensor,
            obs_mask: torch.Tensor,
            meta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode observations, observation mask, and meta data.

        The observation tensor is expected to have shape [B x T x GY x GX x O x 1 x LY x LX], where:
            - B: is the batch dimension
            - T: is the time dimension
            - GY: Is the number of vertical (meridional) mask units.
            - GX: Is the number of horizontal (zonal) mask units.
            - O: Is the number of observation layers.
            - 1: Is just the channels dimension, which is 1 by definition for the observation layers.
            - LY: Is the vertical (meridional) dimension of the observation patches.
            - LX: Is the horizontal (zonal) dimension of the observation patches.

        Args:
            obs: Tensor holding all input observation layers.
            obs_mask: A binary mask identifying valid observation pixels.
            meta: The observation meta data.
            pos: The position encoding of the PrithviWxC model.
        """
        if obs.dim() < 8:
            obs = obs.unsqueeze(-3)
        B, T, GY, GX, O, _, H, W = obs.shape

        obs_enc = self.obs_encoder(obs.view(-1, 1, H, W))
        obs_enc = obs_enc.reshape((B, T, GY, GX, O, self.channels[-1], H // self.patch_height, W // self.patch_width))

        # Use MaxPooling with negative mask to perform MinPooling
        obs_mask_enc = -1.0 * self.mask_encoder(-1.0 * obs_mask.reshape(-1, 1, H, W))
        obs_mask_enc = obs_mask_enc.reshape((B, T, GY, GX, O, H // self.patch_height, W // self.patch_width))

        meta_enc = self.meta_encoder(meta.view(-1, self.n_meta_features, H, W))
        meta_enc = meta_enc.reshape((B, T, GY, GX, O, self.channels[-1], H // self.patch_height, W // self.patch_width))

        return obs_enc, obs_mask_enc, meta_enc


class MultiheadAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0

    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = num_heads
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.w = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        B, T, _ = query.shape
        B, S, _ = key.shape

        q = self.q(query).view(B, T, self.n_heads, self.embed_dim // self.n_heads)
        q = q.transpose(1, 2)
        k = self.k(key).view(B, S, self.n_heads, self.embed_dim // self.n_heads)
        k = k.transpose(1, 2)
        v = self.v(value).view(B, S, self.n_heads, self.embed_dim // self.n_heads)
        v = v.transpose(1, 2)

        if attn_mask is not None:
            if attn_mask.dim() < 4:
                attn_mask = attn_mask[:, None]

        # Let us enforce either flash (A100+) or memory efficient attention.
        if version("torch") > "2.3.0":
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=self.dropout
                )
        else:
            with torch.backends.cuda.sdp_kernel(
                    enable_flash=True, enable_math=False, enable_mem_efficient=True
            ):
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=self.dropout
                )

        x = x.transpose(1, 2).view(B, T, self.embed_dim)
        x = self.w(x)
        return x


class PerceiverBlock(nn.Module):
    """
    Perceiver block to project arbitrary length into a fixed-length feature vector.
    """
    def __init__(
            self,
            latent_dim: int,
            input_dim: int,
            num_heads: int = 4
    ):
        """
        Args:
            latent_dim: The dimensionality of the latent space onto which to project the input.
            input_dim: The dimensionality of the input sequence.
            num_heads: The number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.input_to_latent = nn.Linear(input_dim, latent_dim)
        self.cross_attn = MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads)
        self.cross_ff = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        self.self_attn = MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads)
        self.self_ff = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(
            self,
            latent: torch.Tensor,
            input_data: torch.Tensor,
            input_mask: torch.Tensor
    ):
        input_data = self.input_to_latent(input_data)

        if input_mask.dim() < 3:
            input_mask = input_mask[:, None]

        cross_attn_out = self.cross_attn(
            query=latent,
            key=input_data,
            value=input_data,
            attn_mask=input_mask
        )
        latent = latent + cross_attn_out
        latent = latent + self.cross_ff(self.norm1(latent))

        # Self-attention: within latent
        self_attn_out = self.self_attn(query=latent, key=latent, value=latent)
        latent = latent + self_attn_out
        latent = latent + self.self_ff(self.norm2(latent))

        return latent


class CondLayerNorm(nn.Module):
    """
    LayerNorm whose affine parameters are conditioned on auxiliary input.
    """
    def __init__(
        self,
        feature_dim: int,
        cond_dim: int,
        hidden_dim: int = 128,
        eps: float = 1e-5,
        activation: nn.Module = nn.GELU(),  # optional: keep if you want LN->act fused
        use_activation: bool = False
    ):
        super().__init__()
        self.ln = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=eps)
        self.film_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * feature_dim)  # gamma | beta
        )


    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Normalize x and apply scaling conditioned on cond.

        Args:
            x: The input tensor with shape [B, ..., F]
            cond: The auxiliary input used to condition the scaling [B, F_cond]

        Return:
            The normalized and scaled x.
        """
        B = x.shape[0]
        F = x.shape[-1]
        z = self.ln(x)

        gammabeta = self.film_mlp(cond)
        gamma, beta = gammabeta.chunk(2, dim=-1)
        expand_shape = [B] + [1] * (z.dim() - 2) + [F]
        gamma = gamma.view(*expand_shape)
        beta  = beta.view(*expand_shape)

        out = gamma * z + beta
        return out


class MergingModule(nn.Module):
    """
    Conditional merging module to merge latent observations with latent model state.
    """
    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: The dimenionality of the latent observation and model states.
        """
        super().__init__()
        self.encoding = nn.Linear(1, 32)
        self.linear_1 = nn.Linear(2 * embed_dim, embed_dim)
        self.cond_norm_1 = CondLayerNorm(embed_dim, 32)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(embed_dim, embed_dim)
        self.cond_norm_2 = CondLayerNorm(embed_dim, 32)

    def forward(
            self,
            x_latent: torch.Tensor,
            obs_latent: torch.Tensor,
            total_lead_time: torch.Tensor
    ):
        """
        Merge modell state 'x_latent' and latent observations 'obs_latent' conditioned on total_lead_time.
        """
        B, *_ = x_latent.shape
        total_lead_time = torch.tensor([[total_lead_time]]).to(device=x_latent.device, dtype=x_latent.dtype)
        total_lead_time = total_lead_time.repeat_interleave(B, 0)

        enc = self.encoding(total_lead_time)
        x = self.linear_1(torch.cat([x_latent, obs_latent], -1))
        x = self.act(self.cond_norm_1(x, enc))
        x = self.linear_2(x)
        x = self.act(self.cond_norm_2(x, enc))
        return self.act(x)


class PrithviWxC(nn.Module):
    """
    Encoder-decoder fusing Hiera with MaxViT. See
    - Ryali et al. "Hiera: A Hierarchical Vision Transformer without the
        Bells-and-Whistles" (https://arxiv.org/abs/2306.00989)
    - Tu et al. "MaxViT: Multi-Axis Vision Transformer"
        (https://arxiv.org/abs/2204.01697)
    """

    def __init__(
        self,
        in_channels: int,
        input_size_time: int,
        in_channels_static: int,
        input_scalers_mu: torch.Tensor,
        input_scalers_sigma: torch.Tensor,
        input_scalers_epsilon: float,
        static_input_scalers_mu: torch.Tensor,
        static_input_scalers_sigma: torch.Tensor,
        static_input_scalers_epsilon: float,
        output_scalers: torch.Tensor,
        n_lats_px: int,
        n_lons_px: int,
        patch_size_px: tuple[int],
        mask_unit_size_px: tuple[int],
        mask_ratio_inputs: float,
        mask_ratio_targets: float,
        embed_dim: int,
        n_blocks_encoder: int,
        n_blocks_decoder: int,
        mlp_multiplier: float,
        n_heads: int,
        dropout: float,
        drop_path: float,
        parameter_dropout: float,
        residual: str,
        masking_mode: str,
        positional_encoding: str,
        encoder_shifting: bool = False,
        decoder_shifting: bool = False,
        checkpoint_encoder: list[int]=[],
        checkpoint_decoder: list[int]=[],
    ) -> None:
        """
        Args:
            in_channels: Number of input channels.
            input_size_time: Number of timestamps in input.
            in_channels_static: Number of input channels for static data.
            input_scalers_mu: Tensor of size (in_channels,). Used to rescale
                input.
            input_scalers_sigma: Tensor of size (in_channels,). Used to rescale
                input.
            input_scalers_epsilon: Float. Used to rescale input.
            static_input_scalers_mu: Tensor of size (in_channels_static). Used
                to rescale static inputs.
            static_input_scalers_sigma: Tensor of size (in_channels_static).
                Used to rescale static inputs.
            static_input_scalers_epsilon: Float. Used to rescale static inputs.
            output_scalers: Tensor of shape (in_channels,). Used to rescale
                output.
            n_lats_px: Total latitudes in data. In pixels.
            n_lons_px: Total longitudes in data. In pixels.
            patch_size_px: Patch size for tokenization. In pixels lat/lon.
            mask_unit_size_px: Size of each mask unit. In pixels lat/lon.
            mask_ratio_inputs: Masking ratio for inputs. 0 to 1.
            mask_ratio_targets: Masking ratio for targets. 0 to 1.
            embed_dim: Embedding dimension
            n_blocks_encoder: Number of local-global transformer pairs in
                encoder.
            n_blocks_decoder: Number of local-global transformer pairs in
                decoder.
            mlp_multiplier: MLP multiplier for hidden features in feed forward
                networks.
            n_heads: Number of attention heads.
            dropout: Dropout.
            drop_path: DropPath.
            parameter_dropout: Dropout applied to parameters.
            residual: Indicates whether and how model should work as residual
                model. Accepted values are 'climate', 'temporal' and 'none'
            positional_encoding: possible values are ['absolute' (default), 'fourier'].
                'absolute'  lat lon encoded in 3 dimensions using sine and cosine
                'fourier' lat/lon to be encoded using various frequencies
            masking_mode: String ['local', 'global', 'both'] that controls the
                type of masking used.
            checkpoint_encoder: List of integers controlling if gradient checkpointing is used on encoder.
                Format: [] for no gradient checkpointing. [3, 7] for checkpointing after 4th and 8th layer etc.
            checkpoint_decoder: List of integers controlling if gradient checkpointing is used on decoder.
                Format: See `checkpoint_encoder`.
            masking_mode: The type of masking to use {'global', 'local', 'both'}
            encoder_shifting: Whether to use swin shifting in the encoder.
            decoder_shifting: Whether to use swin shifting in the decoder.
        """
        super().__init__()

        if mask_ratio_targets > 0.0:
            raise NotImplementedError("Target masking is not implemented.")

        self.in_channels = in_channels
        self.input_size_time = input_size_time
        self.in_channels_static = in_channels_static
        self.n_lats_px = n_lats_px
        self.n_lons_px = n_lons_px
        self.patch_size_px = patch_size_px
        self.mask_unit_size_px = mask_unit_size_px
        self.mask_ratio_inputs = mask_ratio_inputs
        self.mask_ratio_targets = mask_ratio_targets
        self.embed_dim = embed_dim
        self.n_blocks_encoder = n_blocks_encoder
        self.n_blocks_decoder = n_blocks_decoder
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self.drop_path = drop_path
        self.residual = residual
        self._encoder_shift = encoder_shifting
        self._decoder_shift = decoder_shifting
        self.positional_encoding = positional_encoding
        self._checkpoint_encoder = checkpoint_encoder
        self._checkpoint_decoder = checkpoint_decoder

        assert self.n_lats_px % self.mask_unit_size_px[0] == 0
        assert self.n_lons_px % self.mask_unit_size_px[1] == 0
        assert self.mask_unit_size_px[0] % self.patch_size_px[0] == 0
        assert self.mask_unit_size_px[1] % self.patch_size_px[1] == 0

        if self.patch_size_px[0] != self.patch_size_px[1]:
            raise NotImplementedError(
                "Current pixel shuffle implementation assumes same patch size along both dimensions."
            )

        self.local_shape_mu = (
            self.mask_unit_size_px[0] // self.patch_size_px[0],
            self.mask_unit_size_px[1] // self.patch_size_px[1],
        )
        self.global_shape_mu = (
            self.n_lats_px // self.mask_unit_size_px[0],
            self.n_lons_px // self.mask_unit_size_px[1],
        )

        assert input_scalers_mu.shape == (in_channels,)
        assert input_scalers_sigma.shape == (in_channels,)
        assert output_scalers.shape == (in_channels,)

        if self.positional_encoding != 'fourier':
            assert static_input_scalers_mu.shape == (in_channels_static,)
            assert static_input_scalers_sigma.shape == (in_channels_static,)

        # Input shape [batch, time, parameter, lat, lon]
        self.input_scalers_epsilon = input_scalers_epsilon
        self.register_buffer('input_scalers_mu', input_scalers_mu.reshape(1, 1, -1, 1, 1))
        self.register_buffer('input_scalers_sigma', input_scalers_sigma.reshape(1, 1, -1, 1, 1))

        # Static inputs shape [batch, parameter, lat, lon]
        self.static_input_scalers_epsilon = static_input_scalers_epsilon
        self.register_buffer('static_input_scalers_mu', static_input_scalers_mu.reshape(1, -1, 1, 1))
        self.register_buffer('static_input_scalers_sigma', static_input_scalers_sigma.reshape(1, -1, 1, 1))

        # Output shape [batch, parameter, lat, lon]
        self.register_buffer('output_scalers', output_scalers.reshape(1, -1, 1, 1))

        self.parameter_dropout = nn.Dropout2d(p=parameter_dropout)

        self.patch_embedding = PatchEmbed(
            patch_size=patch_size_px,
            channels=in_channels * input_size_time,
            embed_dim=embed_dim,
        )

        if self.residual == "climate":
            self.patch_embedding_static = PatchEmbed(
                patch_size=patch_size_px,
                channels=in_channels + in_channels_static,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embedding_static = PatchEmbed(
                patch_size=patch_size_px,
                channels=in_channels_static,
                embed_dim=embed_dim,
            )

        self.input_time_embedding = nn.Linear(1, embed_dim//4, bias=True)
        self.lead_time_embedding = nn.Linear(1, embed_dim//4, bias=True)

        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, self.embed_dim))
        self._nglobal_mu = np.prod(self.global_shape_mu)
        self._global_idx = torch.arange(self._nglobal_mu)

        self._nlocal_mu = np.prod(self.local_shape_mu)
        self._local_idx = torch.arange(self._nlocal_mu)

        if self._encoder_shift:
            self.encoder_shifter = e_shifter = SWINShiftNoBuffer(
                self.mask_unit_size_px,
                self.global_shape_mu,
                self.local_shape_mu,
                self.patch_size_px,
                n_context_tokens=0,
            )
        else:
            self.encoder_shifter = e_shifter = None
        self.encoder = PrithviWxCEncoderDecoder(
            embed_dim=embed_dim,
            n_blocks=n_blocks_encoder,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            dropout=dropout,
            drop_path=drop_path,
            shifter=e_shifter,
            transformer_cp=checkpoint_encoder,
        )

        if n_blocks_decoder != 0:
            if self._decoder_shift:
                self.decoder_shifter = d_shifter = SWINShift(
                    self.mask_unit_size_px,
                    self.global_shape_mu,
                    self.local_shape_mu,
                    self.patch_size_px,
                    n_context_tokens=0,
                )
            else:
                self.decoder_shifter = d_shifter = None

            self.decoder = PrithviWxCEncoderDecoder(
                embed_dim=embed_dim,
                n_blocks=n_blocks_decoder,
                mlp_multiplier=mlp_multiplier,
                n_heads=n_heads,
                dropout=dropout,
                drop_path=0.,
                shifter=d_shifter,
                transformer_cp=checkpoint_decoder,
            )

            self.unembed = nn.Linear(
                self.embed_dim,
                self.in_channels * self.patch_size_px[0] * self.patch_size_px[1],
                bias=True,
            )

        self.masking_mode = masking_mode.lower()
        match self.masking_mode:
            case "local":
                self.generate_mask = self._gen_mask_local
            case "global":
                self.generate_mask = self._gen_mask_global
            case "both":
                self._mask_both_local: bool = True
                self.generate_mask = self._gen_mask_both
            case _:
                raise ValueError(f"Masking mode '{masking_mode}' not supported")

    def swap_masking(self) -> None:
        if hasattr(self, '_mask_both_local'):
            self._mask_both_local = not self._mask_both_local

    @cached_property
    def n_masked_global(self):
        return int(self.mask_ratio_inputs * np.prod(self.global_shape_mu))

    @cached_property
    def n_masked_local(self):
        return int(self.mask_ratio_inputs * np.prod(self.local_shape_mu))

    @staticmethod
    def _shuffle_along_axis(a, axis):
        # https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
        idx = torch.argsort(input=torch.rand(*a.shape), dim=axis)
        return torch.gather(a, dim=axis, index=idx)

    def _gen_mask_local(self, sizes: tuple[int]) -> tuple[torch.Tensor]:
        """
        Args:
            batch_size: Number of elements in batch
        Returns:
            Tuple of torch tensors. [indices masked, indices unmasked].
            Each of these is a tensor of shape (batch, global sequene)
        """
        # We identifies which indices (values) should be masked

        maskable_indices = self._local_idx.view(1, -1).expand(*sizes[:2], -1)

        maskable_indices = self._shuffle_along_axis(maskable_indices, 2)

        # `...` cannot be jit'd :-(
        indices_masked = maskable_indices[:, :, : self.n_masked_local]
        indices_unmasked = maskable_indices[:, :, self.n_masked_local :]

        return indices_masked, indices_unmasked

    def _gen_mask_global(self, sizes: tuple[int]) -> tuple[torch.Tensor]:
        """
        Args:
            batch_size: Number of elements in batch
        Returns:
            Tuple of torch tensors. [indices masked, indices unmasked].
            Each of these is a tensor of shape (batch, global sequene)
        """
        # We identifies which indices (values) should be masked

        maskable_indices = self._global_idx.view(1, -1).expand(*sizes[:1], -1)

        maskable_indices = self._shuffle_along_axis(maskable_indices, 1)

        indices_masked = maskable_indices[:, : self.n_masked_global]
        indices_unmasked = maskable_indices[:, self.n_masked_global :]

        return indices_masked, indices_unmasked

    def _gen_mask_both(self, sizes: tuple[int]) -> tuple[torch.Tensor]:
        if self._mask_both_local:
            return self._gen_mask_local(sizes)
        else:
            return self._gen_mask_global(sizes)

    @staticmethod
    def reconstruct_batch(
        idx_masked: torch.Tensor,
        idx_unmasked: torch.Tensor,
        data_masked: torch.Tensor,
        data_unmasked: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstructs a tensor along the mask unit dimension. Batched version.

        Args:
            idx_masked: Tensor of shape `batch, mask unit sequence`.
            idx_unmasked: Tensor of shape `batch, mask unit sequence`.
            data_masked: Tensor of shape `batch, mask unit sequence, ...`.
                Should have same size along mask unit sequence dimension as
                idx_masked. Dimensions beyond the first two, marked here as ...
                will typically be `local_sequence, channel` or `channel, lat, lon`.
                  These dimensions should agree with data_unmasked.
            data_unmasked: Tensor of shape `batch, mask unit sequence, ...`.
                Should have same size along mask unit sequence dimension as
                idx_unmasked. Dimensions beyond the first two, marked here as
                ... will typically be `local_sequence, channel` or `channel,
                lat, lon`. These dimensions should agree with data_masked.
        Returns:
            Tensor of same shape as inputs data_masked and data_unmasked. I.e.
            `batch, mask unit sequence, ...`. Index for the total data composed
            of the masked and the unmasked part
        """
        dim: int = idx_masked.ndim

        idx_total = torch.argsort(torch.cat([idx_masked, idx_unmasked], dim=-1), dim=-1)
        idx_total = idx_total.view(*idx_total.shape, *[1] * (data_unmasked.ndim - dim))
        idx_total = idx_total.expand(*idx_total.shape[:dim], *data_unmasked.shape[dim:])

        data = torch.cat([data_masked, data_unmasked], dim=dim - 1)
        data = torch.gather(data, dim=dim - 1, index=idx_total)

        return data, idx_total

    def fourier_pos_encoding(self, x_static):
        """
        Args
            x_static: B x C x H x W. first two channels are lat, and lon respectively
        Returns
            Tensor of shape B x E x H x W where E is the embedding dimension.
        """

        # B x C x H x W -> B x 1 x H/P x W/P
        latitudes_patch = F.avg_pool2d(x_static[:, [0]], kernel_size=self.patch_size_px, stride=self.patch_size_px)
        longitudes_patch = F.avg_pool2d(x_static[:, [1]], kernel_size=self.patch_size_px, stride=self.patch_size_px)

        modes = torch.arange(self.embed_dim//4, device=x_static.device).view(1, -1, 1, 1) + 1.
        pos_encoding = torch.cat(
            (
                torch.sin(latitudes_patch*modes),
                torch.sin(longitudes_patch*modes),
                torch.cos(latitudes_patch*modes),
                torch.cos(longitudes_patch*modes),
            ),
            axis=1
        )

        return pos_encoding # B x E x H/P x W/P

    def time_encoding(self, input_time, lead_time):
        '''
        Args:
            input_time: Tensor of shape [batch].
            lead_time: Tensor of shape [batch].
        Returns:
            Tensor of shape [batch, embed_dim, 1, 1]
        '''
        input_time = self.input_time_embedding(input_time.view(-1, 1, 1, 1))
        lead_time = self.lead_time_embedding(lead_time.view(-1, 1, 1, 1))

        time_encoding = torch.cat(
            (
                torch.cos(input_time),
                torch.cos(lead_time),
                torch.sin(input_time),
                torch.sin(lead_time),
            ),
            axis=3
        )
        return time_encoding

    def to_patching(self, x: torch.Tensor) -> torch.Tensor:
        """Transform data from lat/lon space to two axis patching

        Args: ->
            x: Tesnor in lat/lon space (N, C, Nlat//P_0, Nlon//P_1)

        Returns:
            Tensor in patch space (N, G, L, C)
        """
        n_batch = x.shape[0]

        x = x.view(
            n_batch,
            self.embed_dim,
            self.global_shape_mu[0],
            self.local_shape_mu[0],
            self.global_shape_mu[1],
            self.local_shape_mu[1],
        )
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()

        s = x.shape
        return x.view(n_batch, s[1] * s[2], s[3] * s[4], -1)

    def from_patching(self, x: torch.Tensor) -> torch.Tensor:
        """Transform data from two axis patching to lat/lon space

        Args:
            x: Tensor in patch space with shape (N, G, L, C*P_0*P_1)

        Returns:
            Tensor in lat/lon space (N, C*P_0*P_1, Nlat//P_0, Nlon // P_1)
        """
        n_batch = x.shape[0]

        x = x.view(
            n_batch,
            self.global_shape_mu[0],
            self.global_shape_mu[1],
            self.local_shape_mu[0],
            self.local_shape_mu[1],
            -1,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()

        s = x.shape
        return x.view(n_batch, -1, s[2]*s[3], s[4]*s[5])

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: Dictionary containing the keys 'x', 'y', 'input_time',
                'lead_time' and 'static'. The associated torch tensors have the
                following shapes:
                x: Tensor of shape [batch, time, parameter, lat, lon]
                y: Tensor of shape [batch, parameter, lat, lon]
                static: Tensor of shape [batch, channel_static, lat, lon]
                climate: Optional tensor of shape [batch, parameter, lat, lon]
                input_time: Tensor of shape [batch]. Or none.
                lead_time: Tensor of shape [batch]. Or none.
        Returns:
            Tensor of shape [batch, parameter, lat, lon].
        """
        assert batch["x"].shape[2] == self.in_channels
        assert batch["x"].shape[3] == self.n_lats_px
        assert batch["x"].shape[4] == self.n_lons_px
        assert batch["y"].shape[1] == self.in_channels
        assert batch["y"].shape[2] == self.n_lats_px
        assert batch["y"].shape[3] == self.n_lons_px
        if self.positional_encoding == 'fourier':
            # the first two features (lat, lon) are encoded separately
            assert batch['static'].shape[1] - 2 == self.in_channels_static, "When setting self.positional_encoding to fourier, the number of static params change in the dataset. So, in the config, reduce num_static_channels (e.g., 4 instead of 7)."
        else:
            assert batch['static'].shape[1] == self.in_channels_static
        assert batch["static"].shape[2] == self.n_lats_px
        assert batch["static"].shape[3] == self.n_lons_px

        dtype = batch["x"].dtype
        x_rescaled = (batch["x"].to(dtype=torch.float32) - self.input_scalers_mu) / (
            self.input_scalers_sigma + self.input_scalers_epsilon
        ).to(dtype=dtype)
        #x_rescaled = torch.clip(x_rescaled, -20, 20).to(dtype=dtype)
        print("noclip")
        batch_size = x_rescaled.shape[0]

        if self.positional_encoding == 'fourier':
            x_static_pos = self.fourier_pos_encoding(batch['static']) # B, embed_dim, lat / patch_size, lon / patch_size
            x_static = (batch['static'][:, 2:].to(dtype=torch.float32) - self.static_input_scalers_mu[:, 3:]) / ( # The first two channels in batch['static'] are used in positional encoding
                self.static_input_scalers_sigma[:, 3:] + self.static_input_scalers_epsilon # This translates to the first three channels in 'static_input_scalers_mu'
            ).to(dtype=dtype)
        else:
            x_static = (batch["static"] - self.static_input_scalers_mu) / (
                self.static_input_scalers_sigma + self.static_input_scalers_epsilon
            ).to(dtype=dtype)

        if self.residual == "temporal":
            # We create a residual of same shape as y
            index = torch.where(batch["lead_time"] > 0, batch["x"].shape[1] - 1, 0)
            index = index.view(-1, 1, 1, 1, 1)
            index = index.expand(batch_size, 1, *batch["x"].shape[2:])
            x_hat = torch.gather(batch["x"], dim=1, index=index)
            x_hat = x_hat.squeeze(1)
            assert (
                batch["y"].shape == x_hat.shape
            ), f'Shapes {batch["y"].shape} and {x_hat.shape} do not agree.'
        elif self.residual == "climate":
            climate_scaled = (
                batch["climate"].to(dtype=torch.float32) - self.input_scalers_mu.view(1, -1, 1, 1)
            ) / (
                self.input_scalers_sigma.view(1, -1, 1, 1) + self.input_scalers_epsilon
            ).to(dtype=dtype)
            #climate_scaled = torch.clip(climate_scaled, -20, 20).to(dtype=dtype)

        # [batch, time, parameter, lat, lon] -> [batch, time x parameter, lat, lon]
        x_rescaled = x_rescaled.flatten(1, 2)
        # Parameter dropout
        x_rescaled = self.parameter_dropout(x_rescaled)

        x_embedded = self.patch_embedding(x_rescaled)
        assert x_embedded.shape[1] == self.embed_dim

        if self.residual == "climate":
            static_embedded = self.patch_embedding_static(
                torch.cat((x_static, climate_scaled), dim=1)
            )
        else:
            static_embedded = self.patch_embedding_static(x_static)
        assert static_embedded.shape[1] == self.embed_dim

        if self.positional_encoding == 'fourier':
            static_embedded += x_static_pos

        x_embedded = self.to_patching(x_embedded)
        static_embedded = self.to_patching(static_embedded)

        time_encoding = self.time_encoding(batch['input_time'], batch['lead_time'])

        tokens = x_embedded + static_embedded + time_encoding

        # Now we generate masks based on masking_mode
        indices_masked, indices_unmasked = self.generate_mask(
            (batch_size, self._nglobal_mu)
        )
        indices_masked = indices_masked.to(device=tokens.device)
        indices_unmasked = indices_unmasked.to(device=tokens.device)
        maskdim: int = indices_masked.ndim

        # Unmasking
        unmask_view = (*indices_unmasked.shape, *[1] * (tokens.ndim - maskdim))
        unmasked = torch.gather(
            tokens,
            dim=maskdim - 1,
            index=indices_unmasked.view(*unmask_view).expand(
                *indices_unmasked.shape, *tokens.shape[maskdim:]
            ),
        )

        # Encoder
        lead_time = x["lead_time"]
        x_encoded = self.encoder(unmasked, lead_time=lead_time)

        # Generate and position encode the mask tokens
        # (1, 1, 1, embed_dim) -> (batch, global_seq_masked, local seq, embed_dim)
        mask_view = (*indices_masked.shape, *[1] * (tokens.ndim - maskdim))
        masking = self.mask_token.repeat(*static_embedded.shape[:3], 1)
        masked = masking + static_embedded
        masked = torch.gather(
            masked,
            dim=maskdim - 1,
            index=indices_masked.view(*mask_view).expand(
                *indices_masked.shape, *tokens.shape[maskdim:]
            ),
        )

        recon, _ = self.reconstruct_batch(
            indices_masked, indices_unmasked, masked, x_encoded
        )

        x_decoded = self.decoder(recon, lead_time=lead_time)

        # Output: (batch, global sequence, local sequence, in_channels * patch_size[0] * patch_size[1])
        x_unembed = self.unembed(x_decoded)

        # Reshape to (batch, global_lat, global_lon, local_lat, local_lon, in_channels * patch_size[0] * patch_size[1])
        assert x_unembed.shape[0] == batch_size
        assert x_unembed.shape[1] == self.global_shape_mu[0] * self.global_shape_mu[1]
        assert x_unembed.shape[2] == self.local_shape_mu[0] * self.local_shape_mu[1]
        assert (
            x_unembed.shape[3]
            == self.in_channels * self.patch_size_px[0] * self.patch_size_px[1]
        )

        x_out = self.from_patching(x_unembed)

        # Pixel shuffle to (batch, in_channels, lat, lon)
        x_out = F.pixel_shuffle(x_out, self.patch_size_px[0])

        if self.residual == "temporal":
            x_out = self.output_scalers * x_out + x_hat
        elif self.residual == "climate":
            x_out = self.output_scalers * x_out + batch["climate"]
        elif self.residual == "none":
            x_out = self.output_scalers * x_out + self.input_scalers_mu.reshape(
                1, -1, 1, 1
            )

        return x_out

class PrithviWxCObs(PrithviWxC):
    """
    Extension of the PrithviWxC for integrating satellite observations into model predictions.
    """
    def __init__(
        self,
        in_channels: int,
        input_size_time: int,
        in_channels_static: int,
        input_scalers_mu: torch.Tensor,
        input_scalers_sigma: torch.Tensor,
        input_scalers_epsilon: float,
        static_input_scalers_mu: torch.Tensor,
        static_input_scalers_sigma: torch.Tensor,
        static_input_scalers_epsilon: float,
        output_scalers: torch.Tensor,
        n_lats_px: int,
        n_lons_px: int,
        patch_size_px: tuple[int],
        mask_unit_size_px: tuple[int],
        mask_ratio_inputs: float,
        mask_ratio_targets: float,
        embed_dim: int,
        n_blocks_encoder: int,
        n_blocks_decoder: int,
        mlp_multiplier: float,
        n_heads: int,
        dropout: float,
        drop_path: float,
        parameter_dropout: float,
        residual: str,
        masking_mode: str,
        positional_encoding: str,
        obs_patch_size: Tuple[int, int] = (3, 2),
        obs_features: int = 64,
        conditional_merging: bool = False,
        encoder_shifting: bool = False,
        decoder_shifting: bool = False,
        checkpoint_encoder: list[int] | None = (),
        checkpoint_decoder: list[int] | None = (),
        drop_dynamic: float = 0.0,
        drop_obs: float = 0.0
    ) -> None:
        """
        Args:
            in_channels: Number of input channels.
            input_size_time: Number of timestamps in input.
            in_channels_static: Number of input channels for static data.
            input_scalers_mu: Tensor of size (in_channels,). Used to rescale
                input.
            input_scalers_sigma: Tensor of size (in_channels,). Used to rescale
                input.
            input_scalers_epsilon: Float. Used to rescale input.
            static_input_scalers_mu: Tensor of size (in_channels_static). Used
                to rescale static inputs.
            static_input_scalers_sigma: Tensor of size (in_channels_static).
                Used to rescale static inputs.
            static_input_scalers_epsilon: Float. Used to rescale static inputs.
            output_scalers: Tensor of shape (in_channels,). Used to rescale
                output.
            n_lats_px: Total latitudes in data. In pixels.
            n_lons_px: Total longitudes in data. In pixels.
            patch_size_px: Patch size for tokenization. In pixels lat/lon.
            mask_unit_size_px: Size of each mask unit. In pixels lat/lon.
            mask_ratio_inputs: Masking ratio for inputs. 0 to 1.
            embed_dim: Embedding dimension
            n_blocks_encoder: Number of local-global transformer pairs in
                encoder.
            n_blocks_decoder: Number of local-global transformer pairs in
                decoder.
            mlp_multiplier: MLP multiplier for hidden features in feed forward
                networks.
            n_heads: Number of attention heads.
            dropout: Dropout.
            drop_path: DropPath.
            parameter_dropout: Dropout applied to parameters.
            residual: Indicates whether and how model should work as residual
                model. Accepted values are 'climate', 'temporal' and 'none'
            positional_encoding: possible values are
              ['absolute' (default), 'fourier'].
                'absolute'  lat lon encoded in 3 dimensions using sine and
                  cosine
                'fourier' lat/lon to be encoded using various frequencies
            masking_mode: String ['local', 'global', 'both'] that controls the
                type of masking used.
            checkpoint_encoder: List of integers controlling if gradient
              checkpointing is used on encoder.
                Format: [] for no gradient checkpointing. [3, 7] for
                  checkpointing after 4th and 8th layer etc.
            checkpoint_decoder: List of integers controlling if gradient
              checkpointing is used on decoder.
                Format: See `checkpoint_encoder`.
            masking_mode: The type of masking to use
              {'global', 'local', 'both'}
            decoder_shifting: Whether to use swin shifting in the decoder.
        """
        super().__init__(
            in_channels=in_channels,
            input_size_time=input_size_time,
            in_channels_static=in_channels_static,
            input_scalers_mu=input_scalers_mu,
            input_scalers_sigma=input_scalers_sigma,
            input_scalers_epsilon=input_scalers_epsilon,
            static_input_scalers_mu=static_input_scalers_mu,
            static_input_scalers_sigma=static_input_scalers_sigma,
            static_input_scalers_epsilon=static_input_scalers_epsilon,
            output_scalers=output_scalers,
            n_lats_px=n_lats_px,
            n_lons_px=n_lons_px,
            patch_size_px=patch_size_px,
            mask_unit_size_px=mask_unit_size_px,
            mask_ratio_inputs=mask_ratio_inputs,
            mask_ratio_targets=mask_ratio_targets,
            embed_dim=embed_dim,
            n_blocks_encoder=n_blocks_encoder,
            n_blocks_decoder=n_blocks_decoder,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            dropout=dropout,
            drop_path=drop_path,
            parameter_dropout=parameter_dropout,
            residual=residual,
            masking_mode=masking_mode,
            positional_encoding=positional_encoding,
            encoder_shifting=encoder_shifting,
            decoder_shifting=decoder_shifting,
            checkpoint_encoder=checkpoint_encoder,
            checkpoint_decoder=checkpoint_decoder,
        )

        channels = (16, 32, obs_features)
        self.obs_patch_size = obs_patch_size
        self.obs_features = obs_features
        self.obs_latent = obs_features

        self.obs_encoder = ObservationEncoder(
            n_meta_features=8,
            obs_patch_size=obs_patch_size,
            channels=channels
        )
        self.perceiver = PerceiverBlock(self.obs_latent, obs_features)
        self.obs_projection = nn.Parameter(torch.randn(1, self.obs_latent))
        upsmpl = tuple([sze // 2 for sze in self.obs_patch_size])
        self.temporal_encoder = nn.Sequential(
            ResNeXtBlock(
                input_size_time * self.obs_latent,
                self.embed_dim,
                activation_factory=nn.GELU,
                normalization_factory=LayerNormFirst
            ),
            nn.Upsample(scale_factor=upsmpl, mode="bilinear")
        )


        if conditional_merging:
            self.obs_merger = MergingModule(self.embed_dim)
        else:
            self.obs_merger = nn.Sequential(
                nn.Linear(2 * self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU()
            )
        self.obs_scale = nn.Parameter(torch.tensor(1e-2))

        self.drop_dynamic = drop_dynamic
        self.drop_obs = drop_obs
        self.obs_dropout = 0.2

    def _gen_mask_local(self, sizes: tuple[int]) -> tuple[torch.Tensor]:
        """
        Args:
            batch_size: Number of elements in batch
        Returns:
            Tuple of torch tensors. [indices masked, indices unmasked].
            Each of these is a tensor of shape (batch, global sequene)
        """
        # We identifies which indices (values) should be masked

        maskable_indices = self._local_idx.view(1, -1).expand(*sizes[:2], -1)
        if self.n_masked_local > 0:
            maskable_indices = self._shuffle_along_axis(maskable_indices, 2)

        # `...` cannot be jit'd :-(
        indices_masked = maskable_indices[:, :, : self.n_masked_local]
        indices_unmasked = maskable_indices[:, :, self.n_masked_local :]

        return indices_masked, indices_unmasked

    def _gen_mask_global(self, sizes: tuple[int]) -> tuple[torch.Tensor]:
        """
        Args:
            batch_size: Number of elements in batch
        Returns:
            Tuple of torch tensors. [indices masked, indices unmasked].
            Each of these is a tensor of shape (batch, global sequene)
        """
        # We identifies which indices (values) should be masked

        maskable_indices = self._global_idx.view(1, -1).expand(*sizes[:1], -1)
        if self.n_masked_local > 0:
            maskable_indices = self._shuffle_along_axis(maskable_indices, 1)

        indices_masked = maskable_indices[:, : self.n_masked_global]
        indices_unmasked = maskable_indices[:, self.n_masked_global :]

        return indices_masked, indices_unmasked

    def encode_observations(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode observations.

        This function is useful to encode observation prior to unrolling..

        Args:
            batch: A dictionary containing the model input.

        Return:
            The encoded observations.
        """
        obs = batch["obs"]
        mask = batch["obs_mask"]
        meta = batch["obs_meta"]

        if self.training:
            obs_dropout = (torch.rand(*obs.shape[:-2], device=obs.device, dtype=obs.dtype) < self.obs_dropout)
            obs_dropout = obs_dropout[..., None, None]
            obs = torch.where(obs_dropout, -1.5, obs)
            meta = torch.where(obs_dropout, -1.5, meta)
            mask[obs_dropout[..., 0, 0, 0]] = torch.tensor(True)

        obs_enc, obs_mask, meta_enc = checkpoint(self.obs_encoder, obs, mask, meta, use_reentrant=False)

        # obs_enc: [B x T x GY x GX x O x C x LY x LX]
        obs_enc = obs_enc + meta_enc
        B, T, GY, GX, O, C, LY, LX = obs_enc.shape

        # obs_enc: B x T x GY x GX x LY x LX x O x C
        obs_enc = torch.permute(obs_enc, (0, 1, 2, 3, 6, 7, 4, 5)).reshape(-1, O, C)
        obs_mask = torch.permute(obs_mask, (0, 1, 2, 3, 5, 6, 4)).reshape(-1, O)
        latent = self.obs_projection[None].repeat_interleave(obs_enc.shape[0], 0)
        obs_latent = checkpoint(self.perceiver, latent, obs_enc, obs_mask < 1.0, use_reentrant=False)

        obs_latent = obs_latent.reshape((B, T, GY, GX, LY, LX, self.obs_latent))
        obs_latent = torch.permute(obs_latent, (0, 1, 6, 2, 4, 3, 5)).reshape(B, C * T, GY * LY, GX * LX)
        obs_latent = checkpoint(self.temporal_encoder, obs_latent, use_reentrant=False).reshape(B, self.embed_dim, GY, 15, GX, 16)
        obs_latent = torch.permute(obs_latent, (0, 2, 4, 3, 5, 1)).reshape(B, GY *  GX, 15 * 16, -1)
        return obs_latent


    def forward(
            self,
            batch: dict[str, torch.Tensor],
            apply_residual: bool = True,
            obs_only: bool = False,
            model_only: bool = False,
            obs_latent: Optional[torch.Tensor] = None,
            total_lead_time: int = 0
    ) -> torch.Tensor:
        """
        Args:
            batch: Dictionary containing the keys 'x', 'y', 'input_time',
                'lead_time' and 'static'. The associated torch tensors have the
                following shapes:
                x: Tensor of shape [batch, time, parameter, lat, lon]
                y: Tensor of shape [batch, parameter, lat, lon]
                static: Tensor of shape [batch, channel_static, lat, lon]
                climate: Optional tensor of shape [batch, parameter, lat, lon]
                input_time: Tensor of shape [batch]. Or none.
                lead_time: Tensor of shape [batch]. Or none.
        Returns:
            Tensor of shape [batch, parameter, lat, lon].
        """
        assert batch["x"].shape[2] == self.in_channels
        assert batch["x"].shape[3] == self.n_lats_px
        assert batch["x"].shape[4] == self.n_lons_px

        if self.positional_encoding == 'fourier':
            # the first two features (lat, lon) are encoded separately
            assert batch['static'].shape[1] - 2 == self.in_channels_static, "When setting self.positional_encoding to fourier, the number of static params change in the dataset. So, in the config, reduce num_static_channels (e.g., 4 instead of 7)."
        else:
            assert batch['static'].shape[1] == self.in_channels_static
        assert batch["static"].shape[2] == self.n_lats_px
        assert batch["static"].shape[3] == self.n_lons_px

        dtype = batch["x"].dtype
        x_rescaled = ((batch["x"].to(dtype=torch.float32) - self.input_scalers_mu) / (
            self.input_scalers_sigma + self.input_scalers_epsilon
        )).to(dtype=dtype)
        x_rescaled = torch.clip(x_rescaled, -20, 20).to(dtype=dtype)
        batch_size = x_rescaled.shape[0]

        if self.positional_encoding == 'fourier':
            x_static_pos = self.fourier_pos_encoding(batch['static']) # B, embed_dim, lat / patch_size, lon / patch_size
            x_static = (batch['static'][:, 2:].to(dtype=torch.float32) - self.static_input_scalers_mu[:, 3:]) / ( # The first two channels in batch['static'] are used in positional encoding
                self.static_input_scalers_sigma[:, 3:] + self.static_input_scalers_epsilon # This translates to the first three channels in 'static_input_scalers_mu'
            ).to(dtype=dtype)
        else:
            x_static = (batch["static"] - self.static_input_scalers_mu) / (
                self.static_input_scalers_sigma + self.static_input_scalers_epsilon
            ).to(dtype=dtype)
        x_static = torch.clip(x_static, -20, 20).to(dtype=dtype)

        if self.residual == "temporal":
            # We create a residual of same shape as y
            index = torch.where(batch["lead_time"] > 0, batch["x"].shape[1] - 1, 0)
            index = index.view(-1, 1, 1, 1, 1)
            index = index.expand(batch_size, 1, *batch["x"].shape[2:])
            x_hat = torch.gather(batch["x"], dim=1, index=index)
            x_hat = x_hat.squeeze(1)
            assert (
                batch["y"].shape == x_hat.shape
            ), f'Shapes {batch["y"].shape} and {x_hat.shape} do not agree.'
        elif self.residual == "climate":
            climate_scaled = (
                batch["climate"].to(dtype=torch.float32) - self.input_scalers_mu.view(1, -1, 1, 1)
            ) / (
                self.input_scalers_sigma.view(1, -1, 1, 1) + self.input_scalers_epsilon
            ).to(dtype=dtype)
            climate_scaled = torch.clip(climate_scaled, -20, 20).to(dtype=dtype)

        # [batch, time, parameter, lat, lon] -> [batch, time x parameter, lat, lon]
        x_rescaled = x_rescaled.flatten(1, 2)
        # Parameter dropout
        x_rescaled = self.parameter_dropout(x_rescaled)

        x_embedded = self.patch_embedding(x_rescaled)
        assert x_embedded.shape[1] == self.embed_dim

        if self.residual == "climate":
            static_embedded = self.patch_embedding_static(
                torch.cat((x_static, climate_scaled), dim=1)
            )
        else:
            static_embedded = self.patch_embedding_static(x_static)
        assert static_embedded.shape[1] == self.embed_dim

        if self.positional_encoding == 'fourier':
            static_embedded += x_static_pos

        x_embedded = self.to_patching(x_embedded)
        static_embedded = self.to_patching(static_embedded)

        time_encoding = self.time_encoding(batch['input_time'], batch['lead_time'])

        n_batch = x_embedded.shape[0]
        if self.training:
            drop_dynamic = torch.rand(n_batch, 1, 1, 1, device=x_embedded.device, dtype=x_embedded.dtype) < self.drop_dynamic
            tokens = torch.where(drop_dynamic, 0.0, x_embedded) + static_embedded + time_encoding
        elif obs_only:
            tokens = 0.0 * x_embedded + static_embedded + time_encoding
        else:
            tokens = x_embedded + static_embedded + time_encoding

        # Now we generate masks based on masking_mode
        indices_masked, indices_unmasked = self.generate_mask(
            (batch_size, self._nglobal_mu)
        )
        indices_masked = indices_masked.to(device=tokens.device)
        indices_unmasked = indices_unmasked.to(device=tokens.device)
        maskdim: int = indices_masked.ndim


        # Unmasking
        unmask_view = (*indices_unmasked.shape, *[1] * (tokens.ndim - maskdim))
        unmasked = torch.gather(
            tokens,
            dim=maskdim - 1,
            index=indices_unmasked.view(*unmask_view).expand(
                *indices_unmasked.shape, *tokens.shape[maskdim:]
            ),
        )

        # Observations
        if self.obs_encoder is not None:

            if obs_latent is None:
                obs_latent = self.encode_observations(batch)

            if isinstance(self.obs_merger, MergingModule):
                obs_merged = self.obs_merger(unmasked, obs_latent, total_lead_time)
            else:
                obs_merged = self.obs_merger(torch.cat((obs_latent, unmasked), -1))

        # Encoder
        #return unmasked, obs_enc, obs_mask_enc
        n_batch = unmasked.shape[0]
        if model_only:
            x_encoded = self.encoder(unmasked + 0.0 * obs_merged)
        elif self.training:
            drop_obs = torch.rand(n_batch, 1, 1, 1, device=obs_merged.device, dtype=obs_merged.dtype) < self.drop_obs
            x_encoded = self.encoder(unmasked + torch.where(drop_obs, 0.0, obs_merged))
        else:
            x_encoded = self.encoder(unmasked + obs_merged)

        # Generate and position encode the mask tokens
        # (1, 1, 1, embed_dim) -> (batch, global_seq_masked, local seq, embed_dim)
        mask_view = (*indices_masked.shape, *[1] * (tokens.ndim - maskdim))
        masking = self.mask_token.repeat(*static_embedded.shape[:3], 1)
        masked = masking + static_embedded
        masked = torch.gather(
            masked,
            dim=maskdim - 1,
            index=indices_masked.view(*mask_view).expand(
                *indices_masked.shape, *tokens.shape[maskdim:]
            ),
        )


        recon, _ = self.reconstruct_batch(
            indices_masked, indices_unmasked, masked, x_encoded
        )
        x_decoded = self.decoder(recon)

        # Output: (batch, global sequence, local sequence, in_channels * patch_size[0] * patch_size[1])
        x_unembed = self.unembed(x_decoded)

        # Reshape to (batch, global_lat, global_lon, local_lat, local_lon, in_channels * patch_size[0] * patch_size[1])
        assert x_unembed.shape[0] == batch_size
        assert x_unembed.shape[1] == self.global_shape_mu[0] * self.global_shape_mu[1]
        assert x_unembed.shape[2] == self.local_shape_mu[0] * self.local_shape_mu[1]
        assert (
            x_unembed.shape[3]
            == self.in_channels * self.patch_size_px[0] * self.patch_size_px[1]
        )

        x_out = self.from_patching(x_unembed)

        # Pixel shuffle to (batch, in_channels, lat, lon)
        x_out = F.pixel_shuffle(x_out, self.patch_size_px[0])

        if not apply_residual:
            return x_out

        if self.residual == "temporal":
            x_out = self.output_scalers * x_out + x_hat
        elif self.residual == "climate":
            x_out = self.output_scalers * x_out + batch["climate"]
        elif self.residual == "none":
            x_out = self.output_scalers * x_out + self.input_scalers_mu.reshape(
                1, -1, 1, 1
            )

        return x_out


class ObsAttention(nn.Module):
    """
    Multi-head cross attention module for integrating observations into the PrithviWxC model.

    This layer allows a group of latent PrithviWxC pixels to attend to the spatially collocated observations.
    """

    def __init__(
            self,
            features_target: int,
            features_source: int,
            n_heads: int, dropout: float,
            obs_patch_size: Tuple[int, int] = (6, 4)
    ) -> None:
        """
        Args:
            features_target: The number of features of the PrithviWxC encoding.
            features_source: The number of features of the encoded observations.
            n_heads: Number of attention heads.
            obs_path_size: The size of the observation patches in pixels
        """
        super().__init__()

        self.features_target = features_target
        self.features_source = features_source
        self.n_heads = n_heads
        self.dropout = dropout

        self.q_layer = torch.nn.Linear(features_target, self.n_heads * features_source, bias=False)
        self.k_layer = torch.nn.Linear(features_source, self.n_heads * features_source, bias=False)
        self.v_layer = torch.nn.Linear(features_source, self.n_heads * features_source, bias=False)
        self.w_layer = torch.nn.Linear(self.n_heads * features_source, features_target, bias=False)
        self.patch_height = obs_patch_size[0] // 2
        self.patch_width = obs_patch_size[1] // 2

    def forward(
            self,
            x: torch.Tensor,
            obs: torch.Tensor,
            obs_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            args: A tuple ``(x, obs, obs_mask)`` containing the latent model state ``x``, the encoded observations
                ``obs``, and the ``obs_mask`` indicating which observation pixel contain valid observations.
        Returns:
            The result of the cross attention between the latent model state and the observations.
        """  # noqa: E501
        x = x.contiguous()
        obs_shape = obs.shape
        B, GL, O, CS = obs.shape

        x_shape = x.shape

        obs = obs.reshape((-1,) + obs_shape[-2:])

        # Target sequence: [B x G x L x C]
        B, G, L, C = x.shape
        # Target sequence: B x G x LY x PY x LX x PX x C]
        LY = 15 // self.patch_height
        LX = 16 // self.patch_width
        x_patched = x.view(B, G, LY, self.patch_height, LX, self.patch_width, C)
        # Target sequence: B x G x LY x LX x PY x PX x C]
        x_patched = torch.permute(x_patched, (0, 1, 2, 4, 3, 5, 6)).contiguous()
        _, _, LY, LX, PY, PX, _ = x_patched.shape

        x_patched = x_patched.reshape((-1, PY * PX, C))
        q = self.q_layer(x_patched)
        q = q.reshape(-1, PY * PX, self.n_heads, self.features_source).transpose(1, 2).contiguous()
        k = self.k_layer(obs)
        k = k.reshape(-1, O, self.n_heads, self.features_source).transpose(1, 2).contiguous()
        v = self.v_layer(obs)
        v = v.reshape(-1, O, self.n_heads, self.features_source).transpose(1, 2).contiguous()

        if obs_mask is not None:
            obs_mask = obs_mask.flatten(0, 1).to(dtype=torch.bool)
            obs_mask = obs_mask[:, None].repeat_interleave(q.shape[-2], 1)
            obs_mask = obs_mask[:, None].repeat_interleave(q.shape[1], 1)
        else:
            obs_mask = torch.zeros((q.shape[0], q.shape[2]), device=q.device, dtype=bool)

        # Let us enforce either flash (A100+) or memory efficient attention.
        if TORCH_VERSION > "2.3.0":
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                # x [B, H, S, C//H]
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=~obs_mask, dropout_p=self.dropout
                )
        else:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            ):
                # x [B, H, S, C//H]
                x = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=~obs_mask, dropout_p=self.dropout
                )

        # x [B, L, C]
        x = x.transpose(1, 2).contiguous().reshape(B * G * LY * LX, PY * PX, self.n_heads * self.features_source)

        # x [B, L, C]
        x = self.w_layer(x)

        x = x.view(B, G, LY, LX, PY, PX, self.features_target)
        x = torch.permute(x, (0, 1, 2, 4, 3, 5, 6)).contiguous()
        x = x.reshape(B, G, L, self.features_target)

        return x


class ObsTransformer(nn.Module):
    """
    Transformer for integrating observation layers.
    """

    def __init__(
        self,
        features: int,
        obs_features: int,
        obs_patch_size: Tuple[int, int],
        n_heads: int,
    ) -> None:
        """
        Args:
            features: Number of features for inputs to the layer.
            mlp_multiplier: Model uses features*mlp_multiplier hidden units.
            n_heads: Number of attention heads. Should be a factor of features.
            (I.e. the layer uses features // n_heads.) dropout: Dropout.
            drop_path: DropPath.
        """
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.attention = ObsAttention(
            features,
            obs_features,
            n_heads,
            obs_patch_size=obs_patch_size,
            dropout=0.0
        )
        self.att_scale = nn.Parameter(torch.tensor([1e-2]))

    def forward(
            self,
            x: torch.Tensor,
            obs: Optional[torch.Tensor],
            obs_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [..., sequence, features]
        Returns:
            Tensor: Tensor of shape [..., sequence, features]
        """
        attention_x = self.att_scale * self.attention(self.norm(x), obs, obs_mask)
        return x + attention_x



class LocalGlobalLocalBlock(nn.Module):
    """
    Applies alternating block and grid attention. Given a parameter n_blocks, the entire
    module contains 2*n_blocks+1 transformer blocks. The first, third, ..., last apply
    local (block) attention. The second, fourth, ... global (grid) attention.

    This is heavily inspired by Tu et al. "MaxViT: Multi-Axis Vision Transformer"
    (https://arxiv.org/abs/2204.01697).
    """

    def __init__(
        self,
        features: int,
        mlp_multiplier: int,
        n_heads: int,
        dropout: float,
        n_blocks: int,
        drop_path: Tuple[float, float],
        shifter: nn.Module | None = None,
        checkpoint: list[int]=[],
        obs_features: Optional[int] = None,
        obs_patch_size: Optional[Tuple[int, int]] = (6, 4)
    ) -> None:
        """
        Args:
            features: Number of features for inputs to the layer.
            mlp_multiplier: Model will use features*mlp_multiplier hidden units.
            n_heads: Number of attention heads. Should be a factor of features.
            (I.e. the layer uses features // n_heads.)
            dropout: Dropout.
            drop_path: DropPath.
            n_blocks: Number of local-global transformer pairs.
        """
        super().__init__()

        self.features = features
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self.drop_path = drop_path
        self.n_blocks = n_blocks
        self._checkpoint = checkpoint

        if len(checkpoint) > 0:
            if min(checkpoint) < 0 or max(checkpoint) >= 2 * n_blocks + 1:
                raise ValueError(f'Checkpoints should satisfy 0 <= i < 2*n_blocks+1. We have {checkpoint}.')

        self.transformers = nn.ModuleList(
            [
                Transformer(
                    features=features,
                    mlp_multiplier=mlp_multiplier,
                    n_heads=n_heads,
                    dropout=dropout,
                    drop_path=drop_path,
                )
                for _ in range(2 * n_blocks + 1)
            ]
        )

        if obs_features is not None:
            self.obs_transformers = nn.ModuleList(
                [
                    ObsTransformer(
                        features=features,
                        obs_features=obs_features,
                        obs_patch_size=obs_patch_size,
                        n_heads=4,
                    ) for _ in range(n_blocks + 1)
                ]
            )
        else:
            self.obs_transformers = None

        self.evaluator = [
            self._checkpoint_wrapper if i in checkpoint else lambda m, *args: m(*args)
            for i, _ in enumerate(self.transformers)
        ]

        self.shifter = shifter or _Shift()

    @staticmethod
    def _checkpoint_wrapper(model, *args):
        return checkpoint(model, *args, use_reentrant=False)

    def forward(
            self,
            x: torch.Tensor,
            obs: Optional[torch.Tensor],
            obs_mask: Optional[torch.Tensor],
            lead_time: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, global_sequence, local_sequence, features]
        Returns:
            Tensor of shape [batch, global_sequence, local_sequence, features]
        """
        if x.shape[-1] != self.features:
            raise ValueError(
                f"Expecting tensor with last dimension of size {self.features}."
            )
        if x.ndim != 4:
            raise ValueError(
                f"Expecting tensor with exactly four dimensions. Input has shape {x.shape}."
            )

        self.shifter.reset()
        local: bool = True
        attn_mask = {True: None, False: None}

        transformer_iter = zip(self.evaluator, self.transformers)

        # First local block

        evaluator, transformer = next(transformer_iter)
        if self.obs_transformers is not None:
            x = evaluator(self.obs_transformers[0], x, obs, obs_mask, lead_time)
        x = evaluator(transformer, (x, attn_mask[local], lead_time))

        cntr = 1
        for evaluator, transformer in transformer_iter:
            local = not local
            # We are making exactly 2*n_blocks transposes.
            # So the output has the same shape as input.
            x = x.transpose(1, 2)

            if self.obs_transformers is not None and local:
                x = evaluator(self.obs_transformers[cntr], x, obs, obs_mask, lead_time)
                cntr += 1

            x = evaluator(transformer, (x, attn_mask[local], lead_time))

            if not local:
                x, attn_mask = self.shifter(x)

        return x


class PrithviWxCEncoderDecoder(nn.Module):
    """
    Hiera-MaxViT encoder/decoder code.
    """

    def __init__(
        self,
        embed_dim: int,
        n_blocks: int,
        mlp_multiplier: float,
        n_heads: int,
        dropout: float,
        drop_path: float,
        obs_features: int = None,
        obs_patch_size: int = None,
        shifter: nn.Module | None = None,
        transformer_cp: list[int]=[],
    ) -> None:
        """
        Args:
            embed_dim: Embedding dimension
            n_blocks: Number of local-global transformer pairs.
            mlp_multiplier: MLP multiplier for hidden features in feed forward
                networks.
            n_heads: Number of attention heads.
            dropout: Dropout.
            drop_path: DropPath.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_blocks = n_blocks
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self._transformer_cp = transformer_cp

        self.lgl_block = LocalGlobalLocalBlock(
            features=embed_dim,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            dropout=dropout,
            drop_path=drop_path,
            n_blocks=n_blocks,
            obs_features=obs_features,
            obs_patch_size=obs_patch_size,
            shifter=shifter,
            checkpoint=transformer_cp,
        )

    def forward(
        self,
        x: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
        lead_time: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, global sequence, local sequence, embed_dim]
        Returns:
            Tensor of shape [batch, mask_unit_sequence, local_sequence, embed_dim].
            Identical in shape to the input x.
        """

        x = self.lgl_block(x, obs=obs, obs_mask=obs_mask, lead_time=lead_time)

        return x


class PrithviWxCXObs(nn.Module):
    """
    Modifiation of the P
    """

    def __init__(
        self,
        in_channels: int,
        input_size_time: int,
        in_channels_static: int,
        input_scalers_mu: torch.Tensor,
        input_scalers_sigma: torch.Tensor,
        input_scalers_epsilon: float,
        static_input_scalers_mu: torch.Tensor,
        static_input_scalers_sigma: torch.Tensor,
        static_input_scalers_epsilon: float,
        output_scalers: torch.Tensor,
        n_lats_px: int,
        n_lons_px: int,
        patch_size_px: tuple[int],
        mask_unit_size_px: tuple[int],
        mask_ratio_inputs: float,
        mask_ratio_targets: float,
        embed_dim: int,
        n_blocks_encoder: int,
        n_blocks_decoder: int,
        mlp_multiplier: float,
        n_heads: int,
        dropout: float,
        drop_path: float,
        parameter_dropout: float,
        residual: str,
        masking_mode: str,
        positional_encoding: str,
        obs_features: int,
        obs_patch_size: Tuple[int, int],
        encoder_shifting: bool = False,
        decoder_shifting: bool = False,
        checkpoint_encoder: list[int]=[],
        checkpoint_decoder: list[int]=[],
    ) -> None:
        """
        Args:
            in_channels: Number of input channels.
            input_size_time: Number of timestamps in input.
            in_channels_static: Number of input channels for static data.
            input_scalers_mu: Tensor of size (in_channels,). Used to rescale
                input.
            input_scalers_sigma: Tensor of size (in_channels,). Used to rescale
                input.
            input_scalers_epsilon: Float. Used to rescale input.
            static_input_scalers_mu: Tensor of size (in_channels_static). Used
                to rescale static inputs.
            static_input_scalers_sigma: Tensor of size (in_channels_static).
                Used to rescale static inputs.
            static_input_scalers_epsilon: Float. Used to rescale static inputs.
            output_scalers: Tensor of shape (in_channels,). Used to rescale
                output.
            n_lats_px: Total latitudes in data. In pixels.
            n_lons_px: Total longitudes in data. In pixels.
            patch_size_px: Patch size for tokenization. In pixels lat/lon.
            mask_unit_size_px: Size of each mask unit. In pixels lat/lon.
            mask_ratio_inputs: Masking ratio for inputs. 0 to 1.
            mask_ratio_targets: Masking ratio for targets. 0 to 1.
            embed_dim: Embedding dimension
            n_blocks_encoder: Number of local-global transformer pairs in
                encoder.
            n_blocks_decoder: Number of local-global transformer pairs in
                decoder.
            mlp_multiplier: MLP multiplier for hidden features in feed forward
                networks.
            n_heads: Number of attention heads.
            dropout: Dropout.
            drop_path: DropPath.
            parameter_dropout: Dropout applied to parameters.
            residual: Indicates whether and how model should work as residual
                model. Accepted values are 'climate', 'temporal' and 'none'
            positional_encoding: possible values are ['absolute' (default), 'fourier'].
                'absolute'  lat lon encoded in 3 dimensions using sine and cosine
                'fourier' lat/lon to be encoded using various frequencies
            masking_mode: String ['local', 'global', 'both'] that controls the
                type of masking used.
            checkpoint_encoder: List of integers controlling if gradient checkpointing is used on encoder.
                Format: [] for no gradient checkpointing. [3, 7] for checkpointing after 4th and 8th layer etc.
            checkpoint_decoder: List of integers controlling if gradient checkpointing is used on decoder.
                Format: See `checkpoint_encoder`.
            masking_mode: The type of masking to use {'global', 'local', 'both'}
            encoder_shifting: Whether to use swin shifting in the encoder.
            decoder_shifting: Whether to use swin shifting in the decoder.
        """
        super().__init__()

        if mask_ratio_targets > 0.0:
            raise NotImplementedError("Target masking is not implemented.")

        self.in_channels = in_channels
        self.input_size_time = input_size_time
        self.in_channels_static = in_channels_static
        self.n_lats_px = n_lats_px
        self.n_lons_px = n_lons_px
        self.patch_size_px = patch_size_px
        self.mask_unit_size_px = mask_unit_size_px
        self.mask_ratio_inputs = mask_ratio_inputs
        self.mask_ratio_targets = mask_ratio_targets
        self.embed_dim = embed_dim
        self.n_blocks_encoder = n_blocks_encoder
        self.n_blocks_decoder = n_blocks_decoder
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self.drop_path = drop_path
        self.residual = residual
        self._encoder_shift = encoder_shifting
        self._decoder_shift = decoder_shifting
        self.positional_encoding = positional_encoding
        self._checkpoint_encoder = checkpoint_encoder
        self._checkpoint_decoder = checkpoint_decoder

        assert self.n_lats_px % self.mask_unit_size_px[0] == 0
        assert self.n_lons_px % self.mask_unit_size_px[1] == 0
        assert self.mask_unit_size_px[0] % self.patch_size_px[0] == 0
        assert self.mask_unit_size_px[1] % self.patch_size_px[1] == 0

        if self.patch_size_px[0] != self.patch_size_px[1]:
            raise NotImplementedError(
                "Current pixel shuffle implementation assumes same patch size along both dimensions."
            )

        self.local_shape_mu = (
            self.mask_unit_size_px[0] // self.patch_size_px[0],
            self.mask_unit_size_px[1] // self.patch_size_px[1],
        )
        self.global_shape_mu = (
            self.n_lats_px // self.mask_unit_size_px[0],
            self.n_lons_px // self.mask_unit_size_px[1],
        )

        assert input_scalers_mu.shape == (in_channels,)
        assert input_scalers_sigma.shape == (in_channels,)
        assert output_scalers.shape == (in_channels,)

        if self.positional_encoding != 'fourier':
            assert static_input_scalers_mu.shape == (in_channels_static,)
            assert static_input_scalers_sigma.shape == (in_channels_static,)

        # Input shape [batch, time, parameter, lat, lon]
        self.input_scalers_epsilon = input_scalers_epsilon
        self.register_buffer('input_scalers_mu', input_scalers_mu.reshape(1, 1, -1, 1, 1))
        self.register_buffer('input_scalers_sigma', input_scalers_sigma.reshape(1, 1, -1, 1, 1))

        # Static inputs shape [batch, parameter, lat, lon]
        self.static_input_scalers_epsilon = static_input_scalers_epsilon
        self.register_buffer('static_input_scalers_mu', static_input_scalers_mu.reshape(1, -1, 1, 1))
        self.register_buffer('static_input_scalers_sigma', static_input_scalers_sigma.reshape(1, -1, 1, 1))

        # Output shape [batch, parameter, lat, lon]
        self.register_buffer('output_scalers', output_scalers.reshape(1, -1, 1, 1))

        self.parameter_dropout = nn.Dropout2d(p=parameter_dropout)

        self.patch_embedding = PatchEmbed(
            patch_size=patch_size_px,
            channels=in_channels * input_size_time,
            embed_dim=embed_dim,
        )

        if self.residual == "climate":
            self.patch_embedding_static = PatchEmbed(
                patch_size=patch_size_px,
                channels=in_channels + in_channels_static,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embedding_static = PatchEmbed(
                patch_size=patch_size_px,
                channels=in_channels_static,
                embed_dim=embed_dim,
            )

        self.input_time_embedding = nn.Linear(1, embed_dim//4, bias=True)
        self.lead_time_embedding = nn.Linear(1, embed_dim//4, bias=True)

        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, self.embed_dim))
        self._nglobal_mu = np.prod(self.global_shape_mu)
        self._global_idx = torch.arange(self._nglobal_mu)

        self._nlocal_mu = np.prod(self.local_shape_mu)
        self._local_idx = torch.arange(self._nlocal_mu)

        if self._encoder_shift:
            self.encoder_shifter = e_shifter = SWINShiftNoBuffer(
                self.mask_unit_size_px,
                self.global_shape_mu,
                self.local_shape_mu,
                self.patch_size_px,
                n_context_tokens=0,
            )
        else:
            self.encoder_shifter = e_shifter = None
        self.encoder = PrithviWxCEncoderDecoder(
            embed_dim=embed_dim,
            n_blocks=n_blocks_encoder,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            dropout=dropout,
            drop_path=drop_path,
            shifter=e_shifter,
            transformer_cp=checkpoint_encoder,
            obs_features=obs_features,
            obs_patch_size=obs_patch_size
        )

        channels = (16, 32, obs_features)
        self.obs_patch_size = obs_patch_size
        self.obs_features = obs_features
        self.obs_latent = obs_features

        self.obs_encoder = ObservationEncoder(
            n_meta_features=8,
            obs_patch_size=obs_patch_size,
            channels=channels
        )

        if n_blocks_decoder != 0:
            if self._decoder_shift:
                self.decoder_shifter = d_shifter = SWINShift(
                    self.mask_unit_size_px,
                    self.global_shape_mu,
                    self.local_shape_mu,
                    self.patch_size_px,
                    n_context_tokens=0,
                )
            else:
                self.decoder_shifter = d_shifter = None

            self.decoder = PrithviWxCEncoderDecoder(
                embed_dim=embed_dim,
                n_blocks=n_blocks_decoder,
                mlp_multiplier=mlp_multiplier,
                n_heads=n_heads,
                dropout=dropout,
                drop_path=0.,
                shifter=d_shifter,
            )

            self.unembed = nn.Linear(
                self.embed_dim,
                self.in_channels * self.patch_size_px[0] * self.patch_size_px[1],
                bias=True,
            )

        self.masking_mode = masking_mode.lower()
        match self.masking_mode:
            case "local":
                self.generate_mask = self._gen_mask_local
            case "global":
                self.generate_mask = self._gen_mask_global
            case "both":
                self._mask_both_local: bool = True
                self.generate_mask = self._gen_mask_both
            case _:
                raise ValueError(f"Masking mode '{masking_mode}' not supported")

    def swap_masking(self) -> None:
        if hasattr(self, '_mask_both_local'):
            self._mask_both_local = not self._mask_both_local

    @cached_property
    def n_masked_global(self):
        return int(self.mask_ratio_inputs * np.prod(self.global_shape_mu))

    @cached_property
    def n_masked_local(self):
        return int(self.mask_ratio_inputs * np.prod(self.local_shape_mu))

    @staticmethod
    def _shuffle_along_axis(a, axis):
        # https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
        idx = torch.argsort(input=torch.rand(*a.shape), dim=axis)
        return torch.gather(a, dim=axis, index=idx)

    def _gen_mask_local(self, sizes: tuple[int]) -> tuple[torch.Tensor]:
        """
        Args:
            batch_size: Number of elements in batch
        Returns:
            Tuple of torch tensors. [indices masked, indices unmasked].
            Each of these is a tensor of shape (batch, global sequene)
        """
        # We identifies which indices (values) should be masked

        maskable_indices = self._local_idx.view(1, -1).expand(*sizes[:2], -1)

        if self.n_masked_local > 0:
            maskable_indices = self._shuffle_along_axis(maskable_indices, 2)

        # `...` cannot be jit'd :-(
        indices_masked = maskable_indices[:, :, : self.n_masked_local]
        indices_unmasked = maskable_indices[:, :, self.n_masked_local :]

        return indices_masked, indices_unmasked

    def _gen_mask_global(self, sizes: tuple[int]) -> tuple[torch.Tensor]:
        """
        Args:
            batch_size: Number of elements in batch
        Returns:
            Tuple of torch tensors. [indices masked, indices unmasked].
            Each of these is a tensor of shape (batch, global sequene)
        """
        # We identifies which indices (values) should be masked

        maskable_indices = self._global_idx.view(1, -1).expand(*sizes[:1], -1)
        if self.n_masked_global > 0:
            maskable_indices = self._shuffle_along_axis(maskable_indices, 1)

        indices_masked = maskable_indices[:, : self.n_masked_global]
        indices_unmasked = maskable_indices[:, self.n_masked_global :]

        return indices_masked, indices_unmasked

    def _gen_mask_both(self, sizes: tuple[int]) -> tuple[torch.Tensor]:
        if self._mask_both_local:
            return self._gen_mask_local(sizes)
        else:
            return self._gen_mask_global(sizes)

    @staticmethod
    def reconstruct_batch(
        idx_masked: torch.Tensor,
        idx_unmasked: torch.Tensor,
        data_masked: torch.Tensor,
        data_unmasked: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstructs a tensor along the mask unit dimension. Batched version.

        Args:
            idx_masked: Tensor of shape `batch, mask unit sequence`.
            idx_unmasked: Tensor of shape `batch, mask unit sequence`.
            data_masked: Tensor of shape `batch, mask unit sequence, ...`.
                Should have same size along mask unit sequence dimension as
                idx_masked. Dimensions beyond the first two, marked here as ...
                will typically be `local_sequence, channel` or `channel, lat, lon`.
                  These dimensions should agree with data_unmasked.
            data_unmasked: Tensor of shape `batch, mask unit sequence, ...`.
                Should have same size along mask unit sequence dimension as
                idx_unmasked. Dimensions beyond the first two, marked here as
                ... will typically be `local_sequence, channel` or `channel,
                lat, lon`. These dimensions should agree with data_masked.
        Returns:
            Tensor of same shape as inputs data_masked and data_unmasked. I.e.
            `batch, mask unit sequence, ...`. Index for the total data composed
            of the masked and the unmasked part
        """
        dim: int = idx_masked.ndim

        idx_total = torch.argsort(torch.cat([idx_masked, idx_unmasked], dim=-1), dim=-1)
        idx_total = idx_total.view(*idx_total.shape, *[1] * (data_unmasked.ndim - dim))
        idx_total = idx_total.expand(*idx_total.shape[:dim], *data_unmasked.shape[dim:])

        data = torch.cat([data_masked, data_unmasked], dim=dim - 1)
        data = torch.gather(data, dim=dim - 1, index=idx_total)

        return data, idx_total

    def fourier_pos_encoding(self, x_static):
        """
        Args
            x_static: B x C x H x W. first two channels are lat, and lon respectively
        Returns
            Tensor of shape B x E x H x W where E is the embedding dimension.
        """

        # B x C x H x W -> B x 1 x H/P x W/P
        latitudes_patch = F.avg_pool2d(x_static[:, [0]], kernel_size=self.patch_size_px, stride=self.patch_size_px)
        longitudes_patch = F.avg_pool2d(x_static[:, [1]], kernel_size=self.patch_size_px, stride=self.patch_size_px)

        modes = torch.arange(self.embed_dim//4, device=x_static.device).view(1, -1, 1, 1) + 1.
        pos_encoding = torch.cat(
            (
                torch.sin(latitudes_patch*modes),
                torch.sin(longitudes_patch*modes),
                torch.cos(latitudes_patch*modes),
                torch.cos(longitudes_patch*modes),
            ),
            axis=1
        )

        return pos_encoding # B x E x H/P x W/P

    def time_encoding(self, input_time, lead_time):
        '''
        Args:
            input_time: Tensor of shape [batch].
            lead_time: Tensor of shape [batch].
        Returns:
            Tensor of shape [batch, embed_dim, 1, 1]
        '''
        input_time = self.input_time_embedding(input_time.view(-1, 1, 1, 1))
        lead_time = self.lead_time_embedding(lead_time.view(-1, 1, 1, 1))

        time_encoding = torch.cat(
            (
                torch.cos(input_time),
                torch.cos(lead_time),
                torch.sin(input_time),
                torch.sin(lead_time),
            ),
            axis=3
        )
        return time_encoding

    def to_patching(self, x: torch.Tensor) -> torch.Tensor:
        """Transform data from lat/lon space to two axis patching

        Args: ->
            x: Tesnor in lat/lon space (N, C, Nlat//P_0, Nlon//P_1)

        Returns:
            Tensor in patch space (N, G, L, C)
        """
        n_batch = x.shape[0]

        x = x.view(
            n_batch,
            self.embed_dim,
            self.global_shape_mu[0],
            self.local_shape_mu[0],
            self.global_shape_mu[1],
            self.local_shape_mu[1],
        )
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()

        s = x.shape
        return x.view(n_batch, s[1] * s[2], s[3] * s[4], -1)

    def from_patching(self, x: torch.Tensor) -> torch.Tensor:
        """Transform data from two axis patching to lat/lon space

        Args:
            x: Tensor in patch space with shape (N, G, L, C*P_0*P_1)

        Returns:
            Tensor in lat/lon space (N, C*P_0*P_1, Nlat//P_0, Nlon // P_1)
        """
        n_batch = x.shape[0]

        x = x.view(
            n_batch,
            self.global_shape_mu[0],
            self.global_shape_mu[1],
            self.local_shape_mu[0],
            self.local_shape_mu[1],
            -1,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()

        s = x.shape
        return x.view(n_batch, -1, s[2]*s[3], s[4]*s[5])

    def forward(self, batch: dict[str, torch.Tensor], apply_residual: bool = True) -> torch.Tensor:
        """
        Args:
            batch: Dictionary containing the keys 'x', 'y', 'input_time',
                'lead_time' and 'static'. The associated torch tensors have the
                following shapes:
                x: Tensor of shape [batch, time, parameter, lat, lon]
                y: Tensor of shape [batch, parameter, lat, lon]
                static: Tensor of shape [batch, channel_static, lat, lon]
                climate: Optional tensor of shape [batch, parameter, lat, lon]
                input_time: Tensor of shape [batch]. Or none.
                lead_time: Tensor of shape [batch]. Or none.
        Returns:
            Tensor of shape [batch, parameter, lat, lon].
        """
        assert batch["x"].shape[2] == self.in_channels
        assert batch["x"].shape[3] == self.n_lats_px
        assert batch["x"].shape[4] == self.n_lons_px

        if self.positional_encoding == 'fourier':
            # the first two features (lat, lon) are encoded separately
            assert batch['static'].shape[1] - 2 == self.in_channels_static, "When setting self.positional_encoding to fourier, the number of static params change in the dataset. So, in the config, reduce num_static_channels (e.g., 4 instead of 7)."
        else:
            assert batch['static'].shape[1] == self.in_channels_static
        assert batch["static"].shape[2] == self.n_lats_px
        assert batch["static"].shape[3] == self.n_lons_px

        x_rescaled = (batch["x"] - self.input_scalers_mu) / (
            self.input_scalers_sigma + self.input_scalers_epsilon
        )
        batch_size = x_rescaled.shape[0]

        if self.positional_encoding == 'fourier':
            x_static_pos = self.fourier_pos_encoding(batch['static']) # B, embed_dim, lat / patch_size, lon / patch_size
            x_static = (batch['static'][:, 2:] - self.static_input_scalers_mu[:, 3:]) / ( # The first two channels in batch['static'] are used in positional encoding
                self.static_input_scalers_sigma[:, 3:] + self.static_input_scalers_epsilon # This translates to the first three channels in 'static_input_scalers_mu'
            )
        else:
            x_static = (batch["static"] - self.static_input_scalers_mu) / (
                self.static_input_scalers_sigma + self.static_input_scalers_epsilon
            )

        if self.residual == "temporal":
            # We create a residual of same shape as y
            index = torch.where(batch["lead_time"] > 0, batch["x"].shape[1] - 1, 0)
            index = index.view(-1, 1, 1, 1, 1)
            index = index.expand(batch_size, 1, *batch["x"].shape[2:])
            x_hat = torch.gather(batch["x"], dim=1, index=index)
            x_hat = x_hat.squeeze(1)
            assert (
                batch["y"].shape == x_hat.shape
            ), f'Shapes {batch["y"].shape} and {x_hat.shape} do not agree.'
        elif self.residual == "climate":
            climate_scaled = (
                batch["climate"] - self.input_scalers_mu.view(1, -1, 1, 1)
            ) / (
                self.input_scalers_sigma.view(1, -1, 1, 1) + self.input_scalers_epsilon
            )

        # [batch, time, parameter, lat, lon] -> [batch, time x parameter, lat, lon]
        x_rescaled = x_rescaled.flatten(1, 2)
        # Parameter dropout
        x_rescaled = self.parameter_dropout(x_rescaled)

        x_embedded = self.patch_embedding(x_rescaled)
        assert x_embedded.shape[1] == self.embed_dim

        if self.residual == "climate":
            static_embedded = self.patch_embedding_static(
                torch.cat((x_static, climate_scaled), dim=1)
            )
        else:
            static_embedded = self.patch_embedding_static(x_static)
        assert static_embedded.shape[1] == self.embed_dim

        if self.positional_encoding == 'fourier':
            static_embedded += x_static_pos

        x_embedded = self.to_patching(x_embedded)
        static_embedded = self.to_patching(static_embedded)

        time_encoding = self.time_encoding(batch['input_time'], batch['lead_time'])

        #tokens = static_embedded + time_encoding #x_embedded + static_embedded + time_encoding
        tokens = x_embedded + static_embedded + time_encoding
        #tokens = static_embedded + time_encoding

        # Now we generate masks based on masking_mode
        indices_masked, indices_unmasked = self.generate_mask(
            (batch_size, self._nglobal_mu)
        )
        indices_masked = indices_masked.to(device=tokens.device)
        indices_unmasked = indices_unmasked.to(device=tokens.device)
        maskdim: int = indices_masked.ndim


        # Unmasking
        unmask_view = (*indices_unmasked.shape, *[1] * (tokens.ndim - maskdim))
        unmasked = torch.gather(
            tokens,
            dim=maskdim - 1,
            index=indices_unmasked.view(*unmask_view).expand(
                *indices_unmasked.shape, *tokens.shape[maskdim:]
            ),
        )

        # Observations
        if self.obs_encoder is not None:
            obs = batch["obs"]
            mask = batch["obs_mask"]
            meta = batch["obs_meta"]
            pos = x_static_pos
            obs_enc, obs_mask_enc, meta_enc = self.obs_encoder(obs, mask, meta)

            # obs_enc: [B x T x GY x GX x O x C x LY x LX]
            obs_enc = obs_enc# + meta_enc + pos_enc

            # Move spatial dims to front: [B x T x GY x GX x O x C x LY x LX] -> [B x GY x GX x LY x LX x T x O x C]
            obs_enc = torch.permute(obs_enc, (0, 2, 3, 6, 7, 1, 4, 5)).contiguous()
            # Fold time and obs layers into one dim -> [B x GY x GX x LY x LX x TO x C]
            obs_enc = torch.flatten(obs_enc, 5, 6)
            # Fold global and local dims;  [B, GL, TO, C]
            obs_enc = torch.flatten(obs_enc, 1, 4)

            #assert obs_enc.shape[-2] == 64
            assert obs_enc.shape[-1] == self.obs_encoder.channels[-1]
            assert obs_enc.shape[1] == 12 * 18 * 30 * 32 // (self.obs_encoder.patch_height * self.obs_encoder.patch_width)

            # Fold obs layers and time: [B x TO X GY x GX x LY x LX]
            obs_mask_enc = torch.permute(obs_mask_enc, (0, 2, 3, 5, 6, 1, 4)).contiguous()
            obs_mask_enc = obs_mask_enc.flatten(5, 6)
            # Final shape: [B x GL x TO]
            obs_mask_enc = obs_mask_enc.flatten(1, 4)
            assert obs_enc.shape[:-1] == obs_mask_enc.shape
        else:
            obs_enc = None
            obs_mask_enc = None

        # Encoder
        #return unmasked, obs_enc, obs_mask_enc

        x_encoded = self.encoder(unmasked, obs=obs_enc, obs_mask=obs_mask_enc)

        # Generate and position encode the mask tokens
        # (1, 1, 1, embed_dim) -> (batch, global_seq_masked, local seq, embed_dim)
        mask_view = (*indices_masked.shape, *[1] * (tokens.ndim - maskdim))
        masking = self.mask_token.repeat(*static_embedded.shape[:3], 1)
        masked = masking + static_embedded
        masked = torch.gather(
            masked,
            dim=maskdim - 1,
            index=indices_masked.view(*mask_view).expand(
                *indices_masked.shape, *tokens.shape[maskdim:]
            ),
        )


        recon, _ = self.reconstruct_batch(
            indices_masked, indices_unmasked, masked, x_encoded
        )
        x_decoded = self.decoder(recon)

        # Output: (batch, global sequence, local sequence, in_channels * patch_size[0] * patch_size[1])
        x_unembed = self.unembed(x_decoded)

        # Reshape to (batch, global_lat, global_lon, local_lat, local_lon, in_channels * patch_size[0] * patch_size[1])
        assert x_unembed.shape[0] == batch_size
        assert x_unembed.shape[1] == self.global_shape_mu[0] * self.global_shape_mu[1]
        assert x_unembed.shape[2] == self.local_shape_mu[0] * self.local_shape_mu[1]
        assert (
            x_unembed.shape[3]
            == self.in_channels * self.patch_size_px[0] * self.patch_size_px[1]
        )

        x_out = self.from_patching(x_unembed)

        # Pixel shuffle to (batch, in_channels, lat, lon)
        x_out = F.pixel_shuffle(x_out, self.patch_size_px[0])

        if not apply_residual:
            return x_out

        if self.residual == "temporal":
            x_out = self.output_scalers * x_out + x_hat
        elif self.residual == "climate":
            x_out = self.output_scalers * x_out + batch["climate"]
        elif self.residual == "none":
            x_out = self.output_scalers * x_out + self.input_scalers_mu.reshape(
                1, -1, 1, 1
            )

        return x_out


class MultiheadCrossAttention(nn.Module):
    """
    Multi-head cross attention module for integrating observations into the PrithviWxC model.

    This layer allows a group of latent PrithviWxC pixels to attend to the spatially collocated observations.
    """

    def __init__(
            self,
            features_target: int,
            features_source: int,
            n_heads: int, dropout: float,
    ) -> None:
        """
        Args:
            features_target: The number of features of the PrithviWxC encoding.
            features_source: The number of features of the encoded observations.
            n_heads: Number of attention heads.
            obs_path_size: The size of the observation patches in pixels
        """
        super().__init__()

        self.features_target = features_target
        self.features_source = features_source
        self.n_heads = n_heads
        self.dropout = dropout

        self.q_layer = torch.nn.Linear(features_target, features_source, bias=False)
        self.k_layer = torch.nn.Linear(features_source, features_source, bias=False)
        self.v_layer = torch.nn.Linear(features_source, features_source, bias=False)
        self.w_layer = torch.nn.Linear(features_source, features_target, bias=False)

    def forward(
            self,
            x_target: torch.Tensor, x_source: torch.Tensor,) -> torch.Tensor:
        """
        Args:
            args: A tuple ``(x, obs, obs_mask)`` containing the latent model state ``x``, the encoded observations
                ``obs``, and the ``obs_mask`` indicating which observation pixel contain valid observations.
        Returns:
            The result of the cross attention between the latent model state and the observations.
        """  # noqa: E501

        # Target sequence: [B x G x L x C]
        B, L, GR, C = x_target.shape
        B, L, GF, C = x_source.shape

        q = self.q_layer(x_target.flatten(0, 2))
        q = q.reshape(B * L, GR, self.n_heads, self.features_source // self.n_heads).transpose(1, 2).contiguous()

        k = self.k_layer(x_source.flatten(0, 2))
        k = k.reshape(B * L, GF, self.n_heads, self.features_source // self.n_heads).transpose(1, 2).contiguous()

        v = self.v_layer(x_source.flatten(0, 2))
        v = v.reshape(B * L, GF, self.n_heads, self.features_source // self.n_heads).transpose(1, 2).contiguous()


        # Let us enforce either flash (A100+) or memory efficient attention.
        if TORCH_VERSION > "2.3.0":
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                # x [B, H, S, C//H]
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout
                )
        else:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            ):
                # x [B, H, S, C//H]
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout
                )

        # x [B, L, C]
        x = x.transpose(1, 2).contiguous().reshape(B * L, GR, self.features_source)

        # x [B, L, C]
        x = self.w_layer(x)

        x = x.view(B, L, GR, self.features_target)

        return x


class CrossTransformer(nn.Module):
    """
    Transformer for inputs of shape [..., S, features].
    """

    def __init__(
        self,
        features_target: int,
        features_source: int,
        mlp_multiplier: int,
        n_heads: int,
        dropout: float,
        drop_path: float,
    ) -> None:
        """
        Args:
            features: Number of features for inputs to the layer.
            mlp_multiplier: Model will use features*mlp_multiplier hidden units.
            n_heads: Number of attention heads. Should be a factor of features.
                (I.e. the layer uses features // n_heads.)
            dropout: Dropout.
            drop_path: DropPath.
        """
        super().__init__()

        self.features_target = features_target
        self.features_source = features_source
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.target_norm = nn.LayerNorm(features_target)
        self.source_norm = nn.LayerNorm(features_target)
        self.attention = MultiheadCrossAttention(features_target, features_source, n_heads, dropout)
        #self.attention = mdl.MultiheadAttention(features_target, n_heads, dropout)

        self.ff = nn.Sequential(
            nn.LayerNorm(features_target),
            Mlp(
                features=features_target,
                hidden_features=features_target * mlp_multiplier,
                dropout=dropout,
            ),
        )

    def forward(self, x_target: torch.Tensor, x_source: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [..., sequence, features]
        Returns:
            Tensor of shape [..., sequence, features]
        """
        #attention_x = self.attention(self.target_norm(x_target), self.source_norm(x_source))
        attention_x = self.attention(self.target_norm(x_target), self.source_norm(x_source))

        x = x_target + self.drop_path(attention_x)
        x = x + self.drop_path(self.ff(x))

        return x


class LocalGlobalLocalCrossAttentionBlock(nn.Module):
    """
    Applies alternating block and grid attention. Given a parameter n_blocks, the entire
    module contains 2*n_blocks+1 transformer blocks. The first, third, ..., last apply
    local (block) attention. The second, fourth, ... global (grid) attention.

    This is heavily inspired by Tu et al. "MaxViT: Multi-Axis Vision Transformer"
    (https://arxiv.org/abs/2204.01697).
    """

    def __init__(
        self,
        features_target: int,
        features_source: int,
        mlp_multiplier: int,
        n_heads: int,
        dropout: float,
        n_blocks: int,
        drop_path: float,
        shifter: nn.Module | None = None,
        checkpoint: list[int]=[],
    ) -> None:
        """
        Args:
            features_target: Number of features for the target sequence.
            features_source: Number of features for input sequence.
            mlp_multiplier: Model will use features*mlp_multiplier hidden units.
            n_heads: Number of attention heads. Should be a factor of features.
            (I.e. the layer uses features // n_heads.)
            dropout: Dropout.
            drop_path: DropPath.
            n_blocks: Number of local-global transformer pairs.
        """
        super().__init__()

        self.features_target = features_target
        self.features_source = features_source
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self.drop_path = drop_path
        self.n_blocks = n_blocks
        self._checkpoint = checkpoint

        if len(checkpoint) > 0:
            if min(checkpoint) < 0 or max(checkpoint) >= 2 * n_blocks + 1:
                raise ValueError(f'Checkpoints should satisfy 0 <= i < 2*n_blocks+1. We have {checkpoint}.')

        blocks = []
        for _ in range(n_blocks):
            blocks += [
                Transformer(
                    features=features_target,
                    mlp_multiplier=mlp_multiplier,
                    n_heads=n_heads,
                    dropout=dropout,
                    drop_path=drop_path,
                ),
                CrossTransformer(
                    features_target=features_target,
                    features_source=features_source,
                    mlp_multiplier=mlp_multiplier,
                    n_heads=n_heads,
                    dropout=dropout,
                    drop_path=drop_path,
                )
            ]
        blocks += [
            Transformer(
                features=features_target,
                mlp_multiplier=mlp_multiplier,
                n_heads=n_heads,
                dropout=dropout,
                drop_path=drop_path,
            ),
        ]
        self.transformers = nn.ModuleList(blocks)

        self.evaluator = [
            self._checkpoint_wrapper if i in checkpoint else lambda m, *args: m(*args)
            for i, _ in enumerate(self.transformers)
        ]

        self.shifter = shifter or _Shift()

    @staticmethod
    def _checkpoint_wrapper(model, *args):
        return checkpoint(model, *args, use_reentrant=False)

    def forward(
            self,
            x_target: torch.Tensor,
            x_source: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, global_sequence, local_sequence, features]
        Returns:
            Tensor of shape [batch, global_sequence, local_sequence, features]
        """
        if x_target.shape[-1] != self.features_target:
            raise ValueError(
                f"Expecting tensor with last dimension of size {self.features_target}."
            )
        if x_source.shape[-1] != self.features_source:
            raise ValueError(
                f"Expecting tensor with last dimension of size {self.features_source}."
            )
        if x_target.ndim != 4:
            raise ValueError(
                f"Expecting tensor with exactly four dimensions. Input has shape {x_target.shape}."
            )
        if x_source.ndim != 4:
            raise ValueError(
                f"Expecting tensor with exactly four dimensions. Input has shape {x_source.shape}."
            )

        self.shifter.reset()
        local: bool = True
        attn_mask = {True: None, False: None}

        transformer_iter = zip(self.evaluator, self.transformers)

        # First local block

        evaluator, transformer = next(transformer_iter)
        #x_target = evaluator(transformer, (x_target, None))
        x_target = evaluator(transformer, (x_target, None))
        x_source = x_source.transpose(1, 2)

        cntr = 1
        for evaluator, transformer in transformer_iter:
            local = not local
            # We are making exactly 2*n_blocks transposes.
            # So the output has the same shape as input.
            x_target = x_target.transpose(1, 2)

            if local:
                x_target = evaluator(transformer, (x_target, None))
            else:
                x_target = evaluator(transformer, x_target, x_source)

            if not local:
                x_target, attn_mask = self.shifter(x_target)

        return x_target


class PrithviWxCRegional(nn.Module):
    """
    Modification of the Prithvi-WxC model for regional fine tuning.
    """
    def __init__(
        self,
        in_channels: int,
        input_size_time: int,
        in_channels_static: int,
        input_scalers_mu: torch.Tensor,
        input_scalers_sigma: torch.Tensor,
        input_scalers_epsilon: float,
        static_input_scalers_mu: torch.Tensor,
        static_input_scalers_sigma: torch.Tensor,
        static_input_scalers_epsilon: float,
        output_scalers: torch.Tensor,
        n_lats_px: int,
        n_lons_px: int,
        patch_size_px: tuple[int],
        mask_unit_size_px: tuple[int],
        mask_ratio_inputs: float,
        mask_ratio_targets: float,
        embed_dim: int,
        n_blocks_encoder: int,
        n_blocks_decoder: int,
        mlp_multiplier: float,
        n_heads: int,
        dropout: float,
        drop_path: float,
        parameter_dropout: float,
        residual: str,
        masking_mode: str,
        positional_encoding: str,
        encoder_shifting: bool = False,
        decoder_shifting: bool = False,
        checkpoint_encoder: list[int]=[],
        checkpoint_decoder: list[int]=[],
    ) -> None:
        """
        Args:
            in_channels: Number of input channels.
            input_size_time: Number of timestamps in input.
            in_channels_static: Number of input channels for static data.
            input_scalers_mu: Tensor of size (in_channels,). Used to rescale
                input.
            input_scalers_sigma: Tensor of size (in_channels,). Used to rescale
                input.
            input_scalers_epsilon: Float. Used to rescale input.
            static_input_scalers_mu: Tensor of size (in_channels_static). Used
                to rescale static inputs.
            static_input_scalers_sigma: Tensor of size (in_channels_static).
                Used to rescale static inputs.
            static_input_scalers_epsilon: Float. Used to rescale static inputs.
            output_scalers: Tensor of shape (in_channels,). Used to rescale
                output.
            n_lats_px: Total latitudes in data. In pixels.
            n_lons_px: Total longitudes in data. In pixels.
            patch_size_px: Patch size for tokenization. In pixels lat/lon.
            mask_unit_size_px: Size of each mask unit. In pixels lat/lon.
            mask_ratio_inputs: Masking ratio for inputs. 0 to 1.
            mask_ratio_targets: Masking ratio for targets. 0 to 1.
            embed_dim: Embedding dimension
            n_blocks_encoder: Number of local-global transformer pairs in
                encoder.
            n_blocks_decoder: Number of local-global transformer pairs in
                decoder.
            mlp_multiplier: MLP multiplier for hidden features in feed forward
                networks.
            n_heads: Number of attention heads.
            dropout: Dropout.
            drop_path: DropPath.
            parameter_dropout: Dropout applied to parameters.
            residual: Indicates whether and how model should work as residual
                model. Accepted values are 'climate', 'temporal' and 'none'
            positional_encoding: possible values are ['absolute' (default), 'fourier'].
                'absolute'  lat lon encoded in 3 dimensions using sine and cosine
                'fourier' lat/lon to be encoded using various frequencies
            masking_mode: String ['local', 'global', 'both'] that controls the
                type of masking used.
            checkpoint_encoder: List of integers controlling if gradient checkpointing is used on encoder.
                Format: [] for no gradient checkpointing. [3, 7] for checkpointing after 4th and 8th layer etc.
            checkpoint_decoder: List of integers controlling if gradient checkpointing is used on decoder.
                Format: See `checkpoint_encoder`.
            masking_mode: The type of masking to use {'global', 'local', 'both'}
            encoder_shifting: Whether to use swin shifting in the encoder.
            decoder_shifting: Whether to use swin shifting in the decoder.
        """
        super().__init__()

        if mask_ratio_targets > 0.0:
            raise NotImplementedError("Target masking is not implemented.")

        self.in_channels = in_channels
        self.input_size_time = input_size_time
        self.in_channels_static = in_channels_static
        self.n_lats_px = n_lats_px
        self.n_lons_px = n_lons_px
        self.patch_size_px = patch_size_px
        self.mask_unit_size_px = mask_unit_size_px
        self.mask_ratio_inputs = mask_ratio_inputs
        self.mask_ratio_targets = mask_ratio_targets
        self.embed_dim = embed_dim
        self.n_blocks_encoder = n_blocks_encoder
        self.n_blocks_decoder = n_blocks_decoder
        self.mlp_multiplier = mlp_multiplier
        self.n_heads = n_heads
        self.dropout = dropout
        self.drop_path = drop_path
        self.residual = residual
        self._encoder_shift = encoder_shifting
        self._decoder_shift = decoder_shifting
        self.positional_encoding = positional_encoding
        self._checkpoint_encoder = checkpoint_encoder
        self._checkpoint_decoder = checkpoint_decoder

        assert self.n_lats_px % self.mask_unit_size_px[0] == 0
        assert self.n_lons_px % self.mask_unit_size_px[1] == 0
        assert self.mask_unit_size_px[0] % self.patch_size_px[0] == 0
        assert self.mask_unit_size_px[1] % self.patch_size_px[1] == 0

        if self.patch_size_px[0] != self.patch_size_px[1]:
            raise NotImplementedError(
                "Current pixel shuffle implementation assumes same patch size along both dimensions."
            )

        self.local_shape_mu = (
            self.mask_unit_size_px[0] // self.patch_size_px[0],
            self.mask_unit_size_px[1] // self.patch_size_px[1],
        )
        self.global_shape_mu = (
            self.n_lats_px // self.mask_unit_size_px[0],
            self.n_lons_px // self.mask_unit_size_px[1],
        )

        assert input_scalers_mu.shape == (in_channels,)
        assert input_scalers_sigma.shape == (in_channels,)
        assert output_scalers.shape == (in_channels,)

        if self.positional_encoding != 'fourier':
            assert static_input_scalers_mu.shape == (in_channels_static,)
            assert static_input_scalers_sigma.shape == (in_channels_static,)

        # Input shape [batch, time, parameter, lat, lon]
        self.input_scalers_epsilon = input_scalers_epsilon
        self.register_buffer('input_scalers_mu', input_scalers_mu.reshape(1, 1, -1, 1, 1))
        self.register_buffer('input_scalers_sigma', input_scalers_sigma.reshape(1, 1, -1, 1, 1))

        # Static inputs shape [batch, parameter, lat, lon]
        self.static_input_scalers_epsilon = static_input_scalers_epsilon
        self.register_buffer('static_input_scalers_mu', static_input_scalers_mu.reshape(1, -1, 1, 1))
        self.register_buffer('static_input_scalers_sigma', static_input_scalers_sigma.reshape(1, -1, 1, 1))

        # Output shape [batch, parameter, lat, lon]
        self.register_buffer('output_scalers', output_scalers.reshape(1, -1, 1, 1))

        self.parameter_dropout = nn.Dropout2d(p=parameter_dropout)

        self.patch_embedding = PatchEmbed(
            patch_size=patch_size_px,
            channels=in_channels * input_size_time,
            embed_dim=embed_dim,
        )

        if self.residual == "climate":
            self.patch_embedding_static = PatchEmbed(
                patch_size=patch_size_px,
                channels=in_channels + in_channels_static,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embedding_static = PatchEmbed(
                patch_size=patch_size_px,
                channels=in_channels_static,
                embed_dim=embed_dim,
            )

        self.input_time_embedding = nn.Linear(1, embed_dim//4, bias=True)
        self.lead_time_embedding = nn.Linear(1, embed_dim//4, bias=True)

        self._nglobal_mu = np.prod(self.global_shape_mu)
        self._global_idx = torch.arange(self._nglobal_mu)

        self._nlocal_mu = np.prod(self.local_shape_mu)
        self._local_idx = torch.arange(self._nlocal_mu)

        if self._encoder_shift:
            self.encoder_shifter = e_shifter = SWINShiftNoBuffer(
                self.mask_unit_size_px,
                self.global_shape_mu,
                self.local_shape_mu,
                self.patch_size_px,
                n_context_tokens=0,
            )
        else:
            self.encoder_shifter = e_shifter = None
        self.encoder = PrithviWxCEncoderDecoder(
            embed_dim=embed_dim,
            n_blocks=n_blocks_encoder,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            dropout=dropout,
            drop_path=drop_path,
            shifter=e_shifter,
            transformer_cp=checkpoint_encoder,
        )

        if self._decoder_shift:
            self.decoder_shifter = d_shifter = SWINShift(
                self.mask_unit_size_px,
                self.global_shape_mu,
                self.local_shape_mu,
                self.patch_size_px,
                n_context_tokens=0,
            )
        else:
            self.decoder_shifter = d_shifter = None

        self.decoder = LocalGlobalLocalCrossAttentionBlock(
            features_target=embed_dim,
            features_source=embed_dim,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            n_blocks=n_blocks_decoder,
            dropout=dropout,
            drop_path=0.,
            shifter=d_shifter,
            checkpoint=checkpoint_decoder,
        )

        self.unembed = nn.Linear(
            self.embed_dim,
            self.in_channels * self.patch_size_px[0] * self.patch_size_px[1],
            bias=True,
        )

        self.masking_mode = masking_mode.lower()
        match self.masking_mode:
            case "local":
                self.generate_mask = self._gen_mask_local
            case "global":
                self.generate_mask = self._gen_mask_global
            case "both":
                self._mask_both_local: bool = True
                self.generate_mask = self._gen_mask_both
            case _:
                raise ValueError(f"Masking mode '{masking_mode}' not supported")

    def swap_masking(self) -> None:
        if hasattr(self, '_mask_both_local'):
            self._mask_both_local = not self._mask_both_local

    @cached_property
    def n_masked_global(self):
        return int(self.mask_ratio_inputs * np.prod(self.global_shape_mu))

    @cached_property
    def n_masked_local(self):
        return int(self.mask_ratio_inputs * np.prod(self.local_shape_mu))

    @staticmethod
    def _shuffle_along_axis(a, axis):
        # https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
        idx = torch.argsort(input=torch.rand(*a.shape), dim=axis)
        return torch.gather(a, dim=axis, index=idx)

    def _gen_mask_local(self, sizes: tuple[int]) -> tuple[torch.Tensor]:
        """
        Args:
            batch_size: Number of elements in batch
        Returns:
            Tuple of torch tensors. [indices masked, indices unmasked].
            Each of these is a tensor of shape (batch, global sequene)
        """
        # We identifies which indices (values) should be masked

        maskable_indices = self._local_idx.view(1, -1).expand(*sizes[:2], -1)

        if self.n_masked_local > 0:
            maskable_indices = self._shuffle_along_axis(maskable_indices, 2)

        # `...` cannot be jit'd :-(
        indices_masked = maskable_indices[:, :, : self.n_masked_local]
        indices_unmasked = maskable_indices[:, :, self.n_masked_local :]

        return indices_masked, indices_unmasked

    def _gen_mask_global(self, sizes: tuple[int]) -> tuple[torch.Tensor]:
        """
        Args:
            batch_size: Number of elements in batch
        Returns:
            Tuple of torch tensors. [indices masked, indices unmasked].
            Each of these is a tensor of shape (batch, global sequene)
        """
        # We identifies which indices (values) should be masked

        maskable_indices = self._global_idx.view(1, -1).expand(*sizes[:1], -1)
        if self.n_masked_global > 0:
            maskable_indices = self._shuffle_along_axis(maskable_indices, 1)

        indices_masked = maskable_indices[:, : self.n_masked_global]
        indices_unmasked = maskable_indices[:, self.n_masked_global :]

        return indices_masked, indices_unmasked

    def _gen_mask_both(self, sizes: tuple[int]) -> tuple[torch.Tensor]:
        if self._mask_both_local:
            return self._gen_mask_local(sizes)
        else:
            return self._gen_mask_global(sizes)

    @staticmethod
    def reconstruct_batch(
        idx_masked: torch.Tensor,
        idx_unmasked: torch.Tensor,
        data_masked: torch.Tensor,
        data_unmasked: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstructs a tensor along the mask unit dimension. Batched version.

        Args:
            idx_masked: Tensor of shape `batch, mask unit sequence`.
            idx_unmasked: Tensor of shape `batch, mask unit sequence`.
            data_masked: Tensor of shape `batch, mask unit sequence, ...`.
                Should have same size along mask unit sequence dimension as
                idx_masked. Dimensions beyond the first two, marked here as ...
                will typically be `local_sequence, channel` or `channel, lat, lon`.
                  These dimensions should agree with data_unmasked.
            data_unmasked: Tensor of shape `batch, mask unit sequence, ...`.
                Should have same size along mask unit sequence dimension as
                idx_unmasked. Dimensions beyond the first two, marked here as
                ... will typically be `local_sequence, channel` or `channel,
                lat, lon`. These dimensions should agree with data_masked.
        Returns:
            Tensor of same shape as inputs data_masked and data_unmasked. I.e.
            `batch, mask unit sequence, ...`. Index for the total data composed
            of the masked and the unmasked part
        """
        dim: int = idx_masked.ndim

        idx_total = torch.argsort(torch.cat([idx_masked, idx_unmasked], dim=-1), dim=-1)
        idx_total = idx_total.view(*idx_total.shape, *[1] * (data_unmasked.ndim - dim))
        idx_total = idx_total.expand(*idx_total.shape[:dim], *data_unmasked.shape[dim:])

        data = torch.cat([data_masked, data_unmasked], dim=dim - 1)
        data = torch.gather(data, dim=dim - 1, index=idx_total)

        return data, idx_total

    def fourier_pos_encoding(self, x_static):
        """
        Args
            x_static: B x C x H x W. first two channels are lat, and lon respectively
        Returns
            Tensor of shape B x E x H x W where E is the embedding dimension.
        """

        # B x C x H x W -> B x 1 x H/P x W/P
        latitudes_patch = F.avg_pool2d(x_static[:, [0]], kernel_size=self.patch_size_px, stride=self.patch_size_px)
        longitudes_patch = F.avg_pool2d(x_static[:, [1]], kernel_size=self.patch_size_px, stride=self.patch_size_px)

        modes = torch.arange(self.embed_dim//4, device=x_static.device).view(1, -1, 1, 1) + 1.
        pos_encoding = torch.cat(
            (
                torch.sin(latitudes_patch*modes),
                torch.sin(longitudes_patch*modes),
                torch.cos(latitudes_patch*modes),
                torch.cos(longitudes_patch*modes),
            ),
            axis=1
        )

        return pos_encoding # B x E x H/P x W/P

    def time_encoding(self, input_time, lead_time):
        '''
        Args:
            input_time: Tensor of shape [batch].
            lead_time: Tensor of shape [batch].
        Returns:
            Tensor of shape [batch, embed_dim, 1, 1]
        '''
        input_time = self.input_time_embedding(input_time.view(-1, 1, 1, 1))
        lead_time = self.lead_time_embedding(lead_time.view(-1, 1, 1, 1))

        time_encoding = torch.cat(
            (
                torch.cos(input_time),
                torch.cos(lead_time),
                torch.sin(input_time),
                torch.sin(lead_time),
            ),
            axis=3
        )
        return time_encoding

    def to_patching(self, x: torch.Tensor) -> torch.Tensor:
        """Transform data from lat/lon space to two axis patching

        Args: ->
            x: Tesnor in lat/lon space (N, C, Nlat//P_0, Nlon//P_1)

        Returns:
            Tensor in patch space (N, G, L, C)
        """
        n_batch = x.shape[0]

        lat_shape_mu = x.shape[2] // self.local_shape_mu[0]
        lon_shape_mu = x.shape[3] // self.local_shape_mu[1]

        x = x.view(
            n_batch,
            self.embed_dim,
            lat_shape_mu,
            self.local_shape_mu[0],
            lon_shape_mu,
            self.local_shape_mu[1],
        )
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()

        s = x.shape
        return x.view(n_batch, s[1] * s[2], s[3] * s[4], -1)

    def from_patching(self, x: torch.Tensor, global_shape: Optional[Tuple[int, int]]) -> torch.Tensor:
        """Transform data from two axis patching to lat/lon space

        Args:
            x: Tensor in patch space with shape (N, G, L, C*P_0*P_1)

        Returns:
            Tensor in lat/lon space (N, C*P_0*P_1, Nlat//P_0, Nlon // P_1)
        """
        n_batch = x.shape[0]

        lat_shape_mu = x.shape[1] // self.local_shape_mu[0]
        lon_shape_mu = x.shape[2] // self.local_shape_mu[1]

        x = x.view(
            n_batch,
            self.global_shape_mu[0] if global_shape is None else global_shape[0],
            self.global_shape_mu[1] if global_shape is None else global_shape[1],
            self.local_shape_mu[0],
            self.local_shape_mu[1],
            -1,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()

        s = x.shape
        return x.view(n_batch, -1, s[2]*s[3], s[4]*s[5])

    def forward(self, batch: dict[str, torch.Tensor], apply_residual: bool = True) -> torch.Tensor:
        """
        Args:
            batch: Dictionary containing the keys 'x', 'y', 'input_time',
                'lead_time' and 'static'. The associated torch tensors have the
                following shapes:
                x: Tensor of shape [batch, time, parameter, lat, lon]
                y: Tensor of shape [batch, parameter, lat, lon]
                static: Tensor of shape [batch, channel_static, lat, lon]
                climate: Optional tensor of shape [batch, parameter, lat, lon]
                input_time: Tensor of shape [batch]. Or none.
                lead_time: Tensor of shape [batch]. Or none.
        Returns:
            Tensor of shape [batch, parameter, lat, lon].
        """
        assert batch["x"].shape[2] == self.in_channels
        assert batch["x"].shape[3] == self.n_lats_px
        assert batch["x"].shape[4] == self.n_lons_px

        if self.positional_encoding == 'fourier':
            # the first two features (lat, lon) are encoded separately
            assert batch['static'].shape[1] - 2 == self.in_channels_static, "When setting self.positional_encoding to fourier, the number of static params change in the dataset. So, in the config, reduce num_static_channels (e.g., 4 instead of 7)."
        else:
            assert batch['static'].shape[1] == self.in_channels_static
        assert batch["static"].shape[2] == self.n_lats_px
        assert batch["static"].shape[3] == self.n_lons_px

        x_rescaled = (batch["x"] - self.input_scalers_mu) / (
            self.input_scalers_sigma + self.input_scalers_epsilon
        )
        batch_size = x_rescaled.shape[0]

        if self.positional_encoding == 'fourier':
            x_static_pos = self.fourier_pos_encoding(batch['static']) # B, embed_dim, lat / patch_size, lon / patch_size
            x_static = (batch['static'][:, 2:] - self.static_input_scalers_mu[:, 3:]) / ( # The first two channels in batch['static'] are used in positional encoding
                self.static_input_scalers_sigma[:, 3:] + self.static_input_scalers_epsilon # This translates to the first three channels in 'static_input_scalers_mu'
            )
        else:
            x_static = (batch["static"] - self.static_input_scalers_mu) / (
                self.static_input_scalers_sigma + self.static_input_scalers_epsilon
            )

        if self.residual == "temporal":
            # We create a residual of same shape as y
            index = torch.where(batch["lead_time"] > 0, batch["x"].shape[1] - 1, 0)
            index = index.view(-1, 1, 1, 1, 1)
            index = index.expand(batch_size, 1, *batch["x"].shape[2:])
            x_hat = torch.gather(batch["x"], dim=1, index=index)
            x_hat = x_hat.squeeze(1)
            assert (
                batch["y"].shape == x_hat.shape
            ), f'Shapes {batch["y"].shape} and {x_hat.shape} do not agree.'
        elif self.residual == "climate":
            climate_scaled = (
                batch["climate"] - self.input_scalers_mu.view(1, -1, 1, 1)
            ) / (
                self.input_scalers_sigma.view(1, -1, 1, 1) + self.input_scalers_epsilon
            )

        # [batch, time, parameter, lat, lon] -> [batch, time x parameter, lat, lon]
        x_rescaled = x_rescaled.flatten(1, 2)
        # Parameter dropout
        x_rescaled = self.parameter_dropout(x_rescaled)

        x_embedded = self.patch_embedding(x_rescaled)
        assert x_embedded.shape[1] == self.embed_dim

        if self.residual == "climate":
            static_embedded = self.patch_embedding_static(
                torch.cat((x_static, climate_scaled), dim=1)
            )
        else:
            static_embedded = self.patch_embedding_static(x_static)
        assert static_embedded.shape[1] == self.embed_dim

        if self.positional_encoding == 'fourier':
            static_embedded += x_static_pos

        x_embedded = self.to_patching(x_embedded)
        static_embedded = self.to_patching(static_embedded)

        time_encoding = self.time_encoding(batch['input_time'], batch['lead_time'])

        #tokens = static_embedded + time_encoding #x_embedded + static_embedded + time_encoding
        tokens = x_embedded + static_embedded + time_encoding
        #tokens = static_embedded + time_encoding

        # Now we generate masks based on masking_mode
        indices_masked, indices_unmasked = self.generate_mask(
            (batch_size, self._nglobal_mu)
        )
        indices_masked = indices_masked.to(device=tokens.device)
        indices_unmasked = indices_unmasked.to(device=tokens.device)
        maskdim: int = indices_masked.ndim


        # Unmasking
        unmask_view = (*indices_unmasked.shape, *[1] * (tokens.ndim - maskdim))
        unmasked = torch.gather(
            tokens,
            dim=maskdim - 1,
            index=indices_unmasked.view(*unmask_view).expand(
                *indices_unmasked.shape, *tokens.shape[maskdim:]
            ),
        )
        x_encoded = self.encoder(unmasked)


        #
        # Encode regional data
        #

        x_rescaled = (batch["x_regional"] - self.input_scalers_mu) / (
            self.input_scalers_sigma + self.input_scalers_epsilon
        )
        batch_size = x_rescaled.shape[0]

        if self.positional_encoding == 'fourier':
            x_static_pos = self.fourier_pos_encoding(batch['static_regional']) # B, embed_dim, lat / patch_size, lon / patch_size
            x_static = (batch['static_regional'][:, 2:] - self.static_input_scalers_mu[:, 3:]) / ( # The first two channels in batch['static'] are used in positional encoding
                self.static_input_scalers_sigma[:, 3:] + self.static_input_scalers_epsilon # This translates to the first three channels in 'static_input_scalers_mu'
            )
        else:
            x_static = (batch["static_regional"] - self.static_input_scalers_mu) / (
                self.static_input_scalers_sigma + self.static_input_scalers_epsilon
            )

        if self.residual == "temporal":
            # We create a residual of same shape as y
            index = torch.where(batch["lead_time"] > 0, batch["x"].shape[1] - 1, 0)
            index = index.view(-1, 1, 1, 1, 1)
            index = index.expand(batch_size, 1, *batch["x"].shape[2:])
            x_hat = torch.gather(batch["x"], dim=1, index=index)
            x_hat = x_hat.squeeze(1)
        elif self.residual == "climate":
            climate_scaled = (
                batch["climate_regional"] - self.input_scalers_mu.view(1, -1, 1, 1)
            ) / (
                self.input_scalers_sigma.view(1, -1, 1, 1) + self.input_scalers_epsilon
            )

        x_rescaled = x_rescaled.flatten(1, 2)
        x_rescaled = self.parameter_dropout(x_rescaled)
        x_embedded = self.patch_embedding(x_rescaled)
        global_shape = x_embedded.shape[-2:]
        global_shape = (
            global_shape[0] // self.local_shape_mu[0],
            global_shape[1] // self.local_shape_mu[1]
        )

        assert x_embedded.shape[1] == self.embed_dim

        if self.residual == "climate":
            static_embedded = self.patch_embedding_static(
                torch.cat((x_static, climate_scaled), dim=1)
            )
        else:
            static_embedded = self.patch_embedding_static(x_static)
        assert static_embedded.shape[1] == self.embed_dim

        if self.positional_encoding == 'fourier':
            static_embedded += x_static_pos

        x_embedded = self.to_patching(x_embedded)
        static_embedded = self.to_patching(static_embedded)
        time_encoding = self.time_encoding(batch['input_time'], batch['lead_time'])
        tokens = static_embedded + time_encoding
        x_decoded = self.decoder(tokens, x_encoded)
        # Output: (batch, global sequence, local sequence, in_channels * patch_size[0] * patch_size[1])
        x_unembed = self.unembed(x_decoded)

        x_out = self.from_patching(x_unembed, global_shape=global_shape)

        # Pixel shuffle to (batch, in_channels, lat, lon)
        x_out = F.pixel_shuffle(x_out, self.patch_size_px[0])

        if not apply_residual:
            return x_out

        if self.residual == "temporal":
            x_out = self.output_scalers * x_out + x_hat
        elif self.residual == "climate":
            x_out = self.output_scalers * x_out + batch["climate"]
        elif self.residual == "none":
            x_out = self.output_scalers * x_out + self.input_scalers_mu.reshape(
                1, -1, 1, 1
            )

        return x_out
