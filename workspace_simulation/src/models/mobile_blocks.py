import torch.nn as nn


NORMALIZATION_TYPES = {"batch_norm", "group_norm", "layer_norm", "none"}
GROUP_NORM_GROUPS = 8


class LayerNorm2d(nn.LayerNorm):
    """Channel-wise LayerNorm for NCHW feature maps."""

    def __init__(self, channels):
        super().__init__(channels)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"LayerNorm2d expects a 4D NCHW tensor, got {x.ndim}D")
        return super().forward(x.movedim(1, -1)).movedim(-1, 1)


def build_normalization_2d(normalization, channels):
    if normalization not in NORMALIZATION_TYPES:
        raise ValueError(
            f"Unsupported normalization [{normalization}], "
            f"expected one of {sorted(NORMALIZATION_TYPES)}"
        )
    if normalization == "batch_norm":
        return nn.BatchNorm2d(channels)
    if normalization == "group_norm":
        if channels % GROUP_NORM_GROUPS:
            raise ValueError(
                f"GroupNorm channels [{channels}] must be divisible by "
                f"the fixed group count [{GROUP_NORM_GROUPS}]"
            )
        return nn.GroupNorm(GROUP_NORM_GROUPS, channels)
    if normalization == "layer_norm":
        return LayerNorm2d(channels)
    return nn.Identity()


def build_normalization_1d(normalization, features):
    if normalization not in NORMALIZATION_TYPES:
        raise ValueError(
            f"Unsupported normalization [{normalization}], "
            f"expected one of {sorted(NORMALIZATION_TYPES)}"
        )
    if normalization == "batch_norm":
        return nn.BatchNorm1d(features)
    if normalization == "group_norm":
        if features % GROUP_NORM_GROUPS:
            raise ValueError(
                f"GroupNorm features [{features}] must be divisible by "
                f"the fixed group count [{GROUP_NORM_GROUPS}]"
            )
        return nn.GroupNorm(GROUP_NORM_GROUPS, features)
    if normalization == "layer_norm":
        return nn.LayerNorm(features)
    return nn.Identity()


class ConvBN(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=None,
            normalization="batch_norm",
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        conv_bias = normalization == "none"
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                bias=conv_bias,
            ),
            build_normalization_2d(normalization, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            normalization="batch_norm",
    ):
        super().__init__()
        conv_bias = normalization == "none"
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=conv_bias,
        )
        self.norm1 = build_normalization_2d(normalization, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=conv_bias,
        )
        self.norm2 = build_normalization_2d(normalization, out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.norm2(x)
        return self.relu2(x)


class SeparableConvBn(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            normalization="batch_norm",
    ):
        super().__init__()
        conv_bias = normalization == "none"
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=conv_bias,
        )
        self.pw_conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            1,
            bias=conv_bias,
        )
        self.bn = build_normalization_2d(normalization, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.bn(x)
        return self.relu(x)


class InvertedBottleNeck(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            expansion_ratio=6,
            stride=1,
            expansion_source="in",
            normalization="batch_norm",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_ratio = expansion_ratio
        self.stride = stride
        self.identical = in_channels == out_channels and stride == 1

        if expansion_source == "in":
            middle_channels = int(in_channels * expansion_ratio)
        elif expansion_source == "out":
            middle_channels = int(out_channels * expansion_ratio)
        else:
            raise ValueError("expansion_source must be 'in' or 'out'")
        self.middle_channels = middle_channels
        conv_bias = normalization == "none"

        self.expansion_pw_conv = nn.Conv2d(
            in_channels,
            middle_channels,
            kernel_size=1,
            stride=1,
            bias=conv_bias,
        )
        self.expansion_pw_bn = build_normalization_2d(
            normalization,
            middle_channels,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.middle_dw_conv = nn.Conv2d(
            middle_channels,
            middle_channels,
            kernel_size=3,
            stride=stride,
            bias=conv_bias,
            groups=middle_channels,
            padding=1,
        )
        self.middle_dw_bn = build_normalization_2d(
            normalization,
            middle_channels,
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.out_pw_conv = nn.Conv2d(
            middle_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=conv_bias,
        )
        self.out_pw_bn = build_normalization_2d(normalization, out_channels)

    def forward(self, x):
        if self.identical:
            residual = x
        x = self.expansion_pw_conv(x)
        x = self.expansion_pw_bn(x)
        x = self.relu1(x)
        x = self.middle_dw_conv(x)
        x = self.middle_dw_bn(x)
        x = self.relu2(x)
        x = self.out_pw_conv(x)
        x = self.out_pw_bn(x)

        if self.identical:
            return x + residual
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        middle_channels = channels // reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, middle_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        att = self.se(x)
        return x * att


class UniversalInvertedBottleneck(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            expand_ratio,
            start_dw_kernel_size=0,
            middle_dw_kernel_size=0,
            stride=1,
            middle_dw_downsample=True,
            se=False,
            use_layer_scale=False,
            layer_scale_init_value=1e-5,
            normalization="batch_norm",
    ):
        super().__init__()
        conv_bias = normalization == "none"
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            self.start_dw_conv = nn.Conv2d(
                in_channels,
                in_channels,
                start_dw_kernel_size,
                stride if not middle_dw_downsample else 1,
                padding=(start_dw_kernel_size - 1) // 2,
                groups=in_channels,
                bias=conv_bias,
            )
            self.start_dw_bn = build_normalization_2d(
                normalization,
                in_channels,
            )
            self.start_dw_act = nn.ReLU(inplace=True)

        expand_channels = int(in_channels * expand_ratio) if in_channels * expand_ratio > 8 else 8
        self.expand_pw_conv = nn.Conv2d(
            in_channels,
            expand_channels,
            1,
            padding=0,
            bias=conv_bias,
        )
        self.expand_bn = build_normalization_2d(
            normalization,
            expand_channels,
        )
        self.expand_act = nn.ReLU(inplace=True)

        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            self.middle_dw_conv = nn.Conv2d(
                expand_channels,
                expand_channels,
                middle_dw_kernel_size,
                stride if middle_dw_downsample else 1,
                padding=(middle_dw_kernel_size - 1) // 2,
                groups=expand_channels,
                bias=conv_bias,
            )
            self.middle_dw_bn = build_normalization_2d(
                normalization,
                expand_channels,
            )
            self.middle_dw_act = nn.ReLU(inplace=True)

        self.se = se
        if self.se:
            self.se_block = SEBlock(expand_channels)

        self.out_bw_conv = nn.Conv2d(
            expand_channels,
            out_channels,
            1,
            padding=0,
            bias=conv_bias,
        )
        self.out_bw_bn = build_normalization_2d(normalization, out_channels)
        self.identity = in_channels == out_channels and stride == 1

    def forward(self, x):
        if self.identity:
            residual = x
        if self.start_dw_kernel_size:
            x = self.start_dw_conv(x)
            x = self.start_dw_bn(x)
            x = self.start_dw_act(x)
        x = self.expand_pw_conv(x)
        x = self.expand_bn(x)
        x = self.expand_act(x)
        if self.middle_dw_kernel_size:
            x = self.middle_dw_conv(x)
            x = self.middle_dw_bn(x)
            x = self.middle_dw_act(x)
        if self.se:
            x = self.se_block(x)
        x = self.out_bw_conv(x)
        out = self.out_bw_bn(x)
        if self.identity:
            return residual + out
        return out
