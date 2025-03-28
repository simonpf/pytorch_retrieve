"""
pytorch_retrieve.models.prithvi_wxc
===================================

Provides extensions of the PrithviWxC foundation model.

NOTE: Requires the PrithviWxC package to be installed.
"""
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from pytorch_retrieve.modules.normalization import LayerNormFirst
from pytorch_retrieve.modules.conv.blocks import ResNeXtBlock
from pytorch_retrieve.modules.conv.padding import Reflect

try:
    import PrithviWxC
    from PrithviWxC.model import PrithviWxC
except ImportError:
    raise ImportError(
        "Could not import the 'PrithviWxC' package. Please make sure it is installed."
    )


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
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.cross_ff = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        self.self_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
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
        if input_mask.dim() < 3:
            input_mask = input_mask[:, None].repeat_interleave(latent.shape[1], 1)

        cross_attn_out, _ = self.cross_attn(
            query=latent,
            key=input_data,
            value=input_data,
            attn_mask=input_mask
        )
        latent = latent + cross_attn_out
        latent = latent + self.cross_ff(self.norm1(latent))

        # Self-attention: within latent
        self_attn_out, _ = self.self_attn(query=latent, key=latent, value=latent)
        latent = latent + self_attn_out
        latent = latent + self.self_ff(self.norm2(latent))

        return latent


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
        decoder_shifting: bool = False,
        checkpoint_encoder: list[int] | None = None,
        checkpoint_decoder: list[int] | None = None,
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
            decoder_shifting=decoder_shifting,
            checkpoint_encoder=checkpoint_encoder,
            checkpoint_decoder=checkpoint_decoder
        )

        channels = (16, 32, obs_features)
        self.obs_patch_size = obs_patch_size
        self.obs_features = obs_features
        self.obs_latent = 256

        self.obs_encoder = ObservationEncoder(
            n_meta_features=8,
            obs_patch_size=obs_patch_size,
            channels=channels
        )
        self.perceiver = PerceiverBlock(self.obs_latent, obs_features)
        self.temporal_encoder = nn.Sequential(
            ResNeXtBlock(
                input_size_time * self.obs_latent,
                self.embed_dim,
                activation_factory=nn.GELU,
                normalization_factory=LayerNormFirst
            ),
            nn.Upsample(scale_factor=self.obs_patch_size, mode="biliner")
        )


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
            obs_enc, obs_mask_enc, meta_enc = self.obs_encoder(obs, mask, meta)

            # obs_enc: [B x T x GY x GX x O x C x LY x LX]
            obs_enc = obs_enc + meta_enc
            B, T, GY, GX, O, C, LY, LX = obs_enc.shape

            # obs_enc: B x T x GY x GX x LY x LX x O x C
            obs_enc = torch.permute(obs_enc, (0, 1, 2, 3, 6, 7, 4, 5)).view(-1, O, C)
            obs_mask = torch.permute(obs_mask, (0, 1, 2, 3, 5, 6, 4)).view(-1, O)
            latent = self.obs_latent[None].repeat_interleave(obs_enc.shape[0], 0)
            obs_latent = self.perceiver(latent, obs_enc, input_mask=obs_mask)

            obs_latent = obs.latent.reshape((B, T, GY, GX, LY, LX, self.obs_latent_dim))
            obs_latent = torch.permute(obs_latent, (0, 2, 3, 1, 6, 4, 5)).reshape(-1, C * T, LY, LX)
            obs_latent = self.temporal_encoder(obs_latent).reshape(B, GY, GX, self.embed_dim, LY, LX)
            obs_latent = torch.permute(obs_latent, (0, 1, 2, 4, 5, 3)).reshape(GY *  GX, LY * LX, -1)


        # Encoder
        #return unmasked, obs_enc, obs_mask_enc

        print("ENCODING")
        x_encoded = self.encoder(unmasked + obs_latent)

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
        diff = recon - x_encoded
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
