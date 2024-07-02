import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import argparse
import os
import os.path
from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8, point_fill

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def myconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=1, bias=True),
        nn.LeakyReLU(0.1))


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small == 'True':
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small == 'True':
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return corr

        return flow_predictions


class RaftCalib(nn.Module):
    def __init__(self, image_size, args):

        super(RaftCalib, self).__init__()

        self.raft = RAFT(args)
        state_dict = torch.load(args.model)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        self.raft.load_state_dict(new_state_dict)

        dd = np.cumsum([128, 128, 96, 64, 32])
        self.conv6_0 = myconv(324, 128, kernel_size=3, stride=1)
        self.conv6_1 = myconv(324 + dd[0], 128, kernel_size=3, stride=2)
        self.conv6_2 = myconv(128, 96, kernel_size=3, stride=2)
        self.conv6_3 = myconv(96, 64, kernel_size=3, stride=2)
        # self.conv6_4 = myconv(324 + dd[3], 32, kernel_size=3, stride=1)

        self.leakyRELU = nn.LeakyReLU(0.1)
        fc_size = 64 * math.ceil((image_size[0]//8) / 8) * math.ceil((image_size[1]//8) / 8)
        self.fc1 = nn.Linear(fc_size, 512)

        self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)

        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)

        # self.dropout = nn.Dropout(0.0)


    def forward(self, rgb, lidar):
        with torch.no_grad():
            corr = self.raft(lidar, rgb, iters=12, test_mode=True)
        # print("flow_low.shape: ", corr.shape)
        x = torch.cat((self.conv6_0(corr), corr), 1)
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        x = self.conv6_3(x)
        # x = torch.cat((self.conv6_4(x), x), 1)
        # print("x.shape: ", x.shape)
        x = x.view(x.shape[0], -1)
        # x = self.dropout(x)
        # print(x.shape)
        x = self.leakyRELU(self.fc1(x))

        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))
        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)

        # print("transl: ", transl, "rot: ", rot)
        return transl, rot