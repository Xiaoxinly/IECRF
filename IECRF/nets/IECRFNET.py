import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import sys

sys.path.append("..")

import utils.geom
import utils.vox
import utils.misc
import utils.basic

from timm.layers import DropPath

from torchvision.models.resnet import resnet18
from efficientnet_pytorch import EfficientNet
from functools import partial

EPS = 1e-4

from timm.models.layers import trunc_normal_
import math
from torch.nn import init


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum

class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip

class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, predict_future_flow):
        super().__init__()
        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.predict_future_flow = predict_future_flow

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=1, padding=0),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

    def forward(self, x, bev_flip_indices=None):
        b, c, h, w = x.shape

        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        x = self.layer3(x)
        x = self.up3_skip(x, skip_x['3'])
        x = self.up2_skip(x, skip_x['2'])
        x = self.up1_skip(x, skip_x['1'])

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2])  # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        feat_output = self.feat_head(x)
        segmentation_output = self.segmentation_head(x)
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None

        return {
            'raw_feat': x,
            'feat': feat_output.view(b, *feat_output.shape[1:]),
            'segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            'instance_flow': instance_future_output.view(b, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
        }

class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):

        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class Encoder_eff(nn.Module):
    def __init__(self, C, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8
        self.version = version

        if self.version == 'b0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.version == 'b4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        x = self.upsampling_layer(input_1, input_2)
        return x

    def forward(self, x):
        x = self.get_features(x)
        x = self.depth_layer(x)
        return x

class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights

class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights

class CSM(nn.Module):
    def __init__(self, dim, reduction=1):
        super(CSM, self).__init__()
        self.dim = dim
        self.ca = ChannelWeights(dim, reduction)
        self.sa = SpatialWeights(dim, reduction)

    def forward(self, x1, x2):
        B, _, H, W = x1.shape

        # Channel weighting
        channel_weights = self.ca(x1, x2)
        x1_weighted = x1 * channel_weights[0]
        x2_weighted = x2 * channel_weights[1]

        # Spatial weighting
        spatial_weights = self.sa(x1, x2)
        x1_weighted = x1_weighted * spatial_weights[0]
        x2_weighted = x2_weighted * spatial_weights[1]

        # Concatenate and reshape final output
        out = torch.cat((x1_weighted, x2_weighted), dim=1)
        out = out.view(B, 2, self.dim, H, W).permute(1, 0, 2, 3, 4)

        return out

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.in_channels = 2 * in_channels
        self.channels = in_channels // 8

        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.mlp = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        self.SoftMax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        b, _, h, w = x1.size()
        x = torch.cat((x1, x2), dim=1)

        query = self.ConvQuery(x)
        query_H = query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
        query_W = query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)

        key = self.ConvKey(x)
        key_H = key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        key_W = key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        value = self.ConvValue(x)
        value_H = value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        value_W = value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        energy_H = (torch.bmm(query_H, key_H) + self.INF(b, h, w)).view(b, w, h, h).permute(0, 2, 1, 3)
        energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)

        concate = self.SoftMax(torch.cat([energy_H, energy_W], 3))

        attention_H = concate[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
        attention_W = concate[:, :, :, h:h + w].contiguous().view(b * h, w, w)

        out_H = torch.bmm(value_H, attention_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        out_W = torch.bmm(value_W, attention_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)
        out = self.gamma * (out_H + out_W) + x
        out = self.mlp(out).view(b, 2, self.in_channels // 2, h, w).permute(1, 0, 2, 3, 4)

        return out

class FAM(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_1=.5):
        super(FAM, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_1 = lambda_1
        self.cba = CSM(dim=dim, reduction=1)
        self.weights = CrissCrossAttention(in_channels=dim)

    def forward(self, x1, x2):
        channel_weights = self.cba(x1, x2)
        weights = self.weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_1 * weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_1 * weights[0] * x1
        return out_x1, out_x2

class FFM(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FFM, self).__init__()
        self.conv1=nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x1,x2):
        x1=self.conv1(x1)
        x1=self.relu(x1)
        x2 = self.conv1(x2)
        x_x=x1*x2
        x_x=self.relu(x_x)
        x_x=x_x+x2
        x_x=x_x*x1
        x_x=self.relu(x_x)
        return x_x

class ConvNextV2Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=2, bias=None, modulation=False):

        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)

        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:

            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)

            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def _get_p_n(self, N, dtype):

        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))

        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)

        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)

        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        p_n = self._get_p_n(N, dtype)

        p_0 = self._get_p_0(h, w, N, dtype)

        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):

        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)

        x = x.contiguous().view(b, c, -1)

        index = q[..., :N] * padded_w + q[..., N:]

        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):

        b, c, h, w, N = x_offset.size()

        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset

    def forward(self, x):

        offset = self.p_conv(x)
        if self.modulation:

            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2
        if self.padding:
            x = self.zero_padding(x)

        p = self._get_p(offset, dtype)

        p = p.contiguous().permute(0, 2, 3, 1)

        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)

        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:

            m = m.contiguous().permute(0, 2, 3, 1)

            m = m.unsqueeze(dim=1)

            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)

        out = self.conv(x_offset)

        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1,
                                                 dilation=1)
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels):
        super(DownBlock, self).__init__()
        self.dcn = DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.block1 = ConvNextV2Block(in_channels)
        self.block2 = ConvNextV2Block(in_channels)

    def forward(self, x):
        x = self.dcn(x)
        x = self.block1(x)
        x = self.block2(x)
        return x

class AggregationModule(nn.Module):
    def __init__(self, in_channels):
        super(AggregationModule, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class MSRE(nn.Module):
    def __init__(self, in_channels):
        super(MSRE, self).__init__()
        self.down1 = DownBlock(in_channels)
        self.up1 = UpBlock(in_channels)
        self.down2 = DownBlock(in_channels)
        self.down3 = DownBlock(in_channels)
        self.down4 = DownBlock(in_channels)
        self.down5 = DownBlock(in_channels)
        self.up2 = UpBlock(in_channels)
        self.up3 = UpBlock(in_channels)
        self.up4 = UpBlock(in_channels)
        self.up5 = UpBlock(in_channels)
        self.agg1 = AggregationModule(in_channels)
        self.agg2 = AggregationModule(in_channels)
        self.agg3 = AggregationModule(in_channels)
        self.agg4 = AggregationModule(in_channels)
        self.agg5 = AggregationModule(in_channels)
        self.agg6 = AggregationModule(in_channels)
        self.up6 = UpBlock(in_channels)

    def forward(self, x):
        down1_out = self.down1(x)
        up1_out = self.up1(down1_out)
        agg1_out = self.agg1(x, up1_out)
        down2_out = self.down2(down1_out)
        up2_out = self.up2(down2_out)
        down3_out = self.down3(agg1_out)
        agg2_out = self.agg2(down3_out, up2_out)
        down4_out = self.down4(down2_out)
        up3_out = self.up3(down4_out)
        down5_out = self.down5(agg2_out)
        agg3_out = self.agg3(down5_out, up3_out)
        up4_out = self.up4(agg3_out)
        agg4_out = self.agg4(agg2_out, up4_out)
        up5_out = self.up5(agg4_out)
        up6_out = self.up6(agg2_out)
        agg6_out = self.agg6(agg1_out, up6_out)
        out = self.agg5(agg6_out, up5_out)

        return out

class Segnet(nn.Module):
    def __init__(self, Z, Y, X, vox_util=None,
                 use_radar=True,
                 use_lidar=False,
                 use_metaradar=True,
                 do_rgbcompress=True,
                 rand_flip=True,
                 latent_dim=128,
                 encoder_type="res101"):
        super(Segnet, self).__init__()
        assert (encoder_type in ["res101", "res50", "effb0", "effb4"])

        self.Z, self.Y, self.X = Z, Y, X
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.use_metaradar = use_metaradar
        self.do_rgbcompress = do_rgbcompress
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        self.frm1 = FAM(dim=128, reduction=1, lambda_c=0.5, lambda_1=0.5)
        self.ffm1 = FFM(128, 128)
        self.frm2 = FAM(dim=128, reduction=1, lambda_c=0.5, lambda_1=0.5)
        self.ffm2 = FFM(128, 128)
        self.cma = MSRE(in_channels=128)

        self.conv_rad1 = nn.Sequential(
            nn.Conv2d(128, latent_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(latent_dim),
            nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(latent_dim),
            nn.GELU(),
        )
        self.conv_rad2 = nn.Sequential(
            nn.Conv2d(16 * Y, latent_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(latent_dim),
            nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(latent_dim),
            nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(latent_dim),
            nn.GELU(),
        )
        self.conv_rad3 = nn.Sequential(
            nn.Conv2d(256, latent_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(latent_dim),
            nn.GELU(),
        )

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).float().cuda()

        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == "effb0":
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        else:
            # effb4
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        if self.use_radar:
            if self.use_metaradar:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim * Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim * Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
        elif self.use_lidar:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim * Y + Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
        else:
            if self.do_rgbcompress:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim * Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:

                pass

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False
        )

        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        if vox_util is not None:
            self.xyz_memA = utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
            self.xyz_camA = vox_util.Mem2Ref(self.xyz_memA, Z, Y, X, assert_cube=False)
        else:
            self.xyz_camA = None

    def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util, rad_occ_mem0=None):

        B, S, C, H, W = rgb_camXs.shape
        assert (C == 3)

        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_camXs_ = __p(rgb_camXs)
        pix_T_cams_ = __p(pix_T_cams)
        cam0_T_camXs_ = __p(cam0_T_camXs)
        camXs_T_cam0_ = utils.geom.safe_inverse(cam0_T_camXs_)

        device = rgb_camXs_.device
        rgb_camXs_ = (rgb_camXs_ + 0.5 - self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            B0, _, _, _ = rgb_camXs_.shape
            self.rgb_flip_index = np.random.choice([0, 1], B0).astype(bool)
            rgb_camXs_[self.rgb_flip_index] = torch.flip(rgb_camXs_[self.rgb_flip_index], [-1])
        feat_camXs_ = self.encoder(rgb_camXs_)
        if self.rand_flip:
            feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
        _, C, Hf, Wf = feat_camXs_.shape

        sy = Hf / float(H)
        sx = Wf / float(W)
        Z, Y, X = self.Z, self.Y, self.X

        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)
        if self.xyz_camA is not None:
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B * S, 1, 1)
        else:
            xyz_camA = None
        feat_mems_ = vox_util.unproject_image_to_mem(
            feat_camXs_,
            utils.basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, Z, Y, X,
            xyz_camA=xyz_camA)
        feat_mems = __u(feat_mems_)

        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1)

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0, 1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0, 1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

            if rad_occ_mem0 is not None:
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])


        if self.use_radar:
            assert (rad_occ_mem0 is not None)
            if not self.use_metaradar:  
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim * Y, Z, X)
                rad_bev = torch.sum(rad_occ_mem0, 3).clamp(0, 1)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim * Y, Z, X)
                rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16 * Y, Z, X)

                feat_bev_1 = self.bev_compressor(feat_bev_)
                rad_bev_1 = self.conv_rad2(rad_bev_)
                rad_bev_2, feat_bev_2 = self.frm1(rad_bev_1, feat_bev_1)
                rad_bev_3 = self.cma(rad_bev_2)
                f1 = self.ffm1(rad_bev_3, feat_bev_2)

                rad_bev_4 = self.conv_rad2(rad_bev_3)
                rad_bev_5, feat_bev_5 = self.frm2(rad_bev_4, feat_bev_2)
                f2 = self.ffm2(rad_bev_5, feat_bev_5)

                feat_bev = torch.cat([f1, f2], dim=1)
                feat_bev = self.conv_rad3(feat_bev)

        elif self.use_lidar:
            assert (rad_occ_mem0 is not None)
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim * Y, Z, X)
            rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
            feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
            feat_bev = self.bev_compressor(feat_bev_)
        else:
            if self.do_rgbcompress:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim * Y, Z, X)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev = torch.sum(feat_mem, dim=3)


        out_dict = self.decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        raw_e = out_dict['raw_feat']
        feat_e = out_dict['feat']
        seg_e = out_dict['segmentation']
        center_e = out_dict['instance_center']
        offset_e = out_dict['instance_offset']

        return raw_e, feat_e, seg_e, center_e, offset_e, rad_bev_1, rad_bev_2, rad_bev_3, f1, f2, feat_bev, feat_bev_1, feat_bev_2

