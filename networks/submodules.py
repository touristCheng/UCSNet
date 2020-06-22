import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-12

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

def uncertainty_aware_samples(cur_depth, exp_var, ndepth, device, dtype, shape):
    if cur_depth.dim() == 2:
        #must be the first stage
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )
        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) # (B, D)
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) # (B, D, H, W)
    else:
        low_bound = -torch.min(cur_depth, exp_var)
        high_bound = exp_var

        # assert exp_var.min() >= 0, exp_var.min()
        assert ndepth > 1

        step = (high_bound - low_bound) / (float(ndepth) - 1)
        new_samps = []
        for i in range(int(ndepth)):
            new_samps.append(cur_depth + low_bound + step * i + eps)

        depth_range_samples = torch.cat(new_samps, 1)
        # assert depth_range_samples.min() >= 0, depth_range_samples.min()
    return depth_range_samples

def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth

class Conv2dUnit(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv2dUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Deconv2dUnit(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Deconv2dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Conv3dUnit(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3dUnit, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Deconv3dUnit(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(Deconv2dBlock, self).__init__()

        self.deconv = Deconv2dUnit(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                                   bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2dUnit(2 * out_channels, out_channels, kernel_size, stride=1, padding=1,
                               bn=bn, relu=relu, bn_momentum=bn_momentum)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x

class FeatExtNet(nn.Module):
    def __init__(self, base_channels, num_stage=3,):
        super(FeatExtNet, self).__init__()

        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2dUnit(3, base_channels, 3, 1, padding=1),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2dUnit(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2dUnit(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2dUnit(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2dUnit(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        if num_stage == 3:
            self.deconv1 = Deconv2dBlock(base_channels * 4, base_channels * 2, 3)
            self.deconv2 = Deconv2dBlock(base_channels * 2, base_channels, 3)

            self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
            self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
            self.out_channels.append(2 * base_channels)
            self.out_channels.append(base_channels)

        elif num_stage == 2:
            self.deconv1 = Deconv2dBlock(base_channels * 4, base_channels * 2, 3)

            self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
            self.out_channels.append(2 * base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)

        outputs["stage1"] = out
        if self.num_stage == 3:
            intra_feat = self.deconv1(conv1, intra_feat)
            out = self.out2(intra_feat)
            outputs["stage2"] = out

            intra_feat = self.deconv2(conv0, intra_feat)
            out = self.out3(intra_feat)
            outputs["stage3"] = out

        elif self.num_stage == 2:
            intra_feat = self.deconv1(conv1, intra_feat)
            out = self.out2(intra_feat)
            outputs["stage2"] = out

        return outputs

class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3dUnit(in_channels, base_channels, padding=1)

        self.conv1 = Conv3dUnit(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3dUnit(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3dUnit(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3dUnit(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3dUnit(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3dUnit(base_channels * 8, base_channels * 8, padding=1)

        self.deconv7 = Deconv3dUnit(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.deconv8 = Deconv3dUnit(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.deconv9 = Deconv3dUnit(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.deconv7(x)
        x = conv2 + self.deconv8(x)
        x = conv0 + self.deconv9(x)
        x = self.prob(x)
        return x

