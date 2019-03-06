import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
from utils import *
import sys
sys.path.append('..')


class ThaiCheckersNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.input_channels = 34
        self.args = args
        self.track_running_stats = True

        super(ThaiCheckersNNet, self).__init__()
        self.conv1 = nn.Conv2d(self.input_channels,
                               args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(
            args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, sAndValids):
        # s: batch_size x board_x x board_y
        # batch_size x 1 x board_x x board_y
        s, valids = sAndValids
        s = s.view(-1, self.input_channels, self.board_x, self.board_y)
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))
        # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn3(self.conv3(s)))
        # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.args.num_channels *
                   (self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout,
                      training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout,
                      training=self.training)  # batch_size x 512

        # batch_size x action_size
        pi = self.fc3(s)
        pi -= (1-valids)*1000
        pi = torch.log_softmax(pi, dim=1)

        # batch_size x 1
        v = self.fc4(s)

        return pi, F.tanh(v)


class ResBlock(nn.Module):
    def __init__(self, filters=256, kernel=3):
        super(ResBlock, self).__init__()
        if kernel % 2 != 1:
            raise ValueError('kernel must be odd, got %d' % kernel)
        pad = int(np.floor(kernel/2))

        self.conv1 = nn.Conv2d(
            filters, filters, kernel_size=kernel, padding=pad)
        self.bn1 = nn.BatchNorm2d(
            filters, track_running_stats=self.track_running_stats)
        self.conv2 = nn.Conv2d(
            filters, filters, kernel_size=kernel, padding=pad)
        self.bn2 = nn.BatchNorm2d(
            filters, track_running_stats=self.track_running_stats)

    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + inp
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    """The alphazero model."""

    def __init__(self, game,
                 block_filters=128,
                 block_kernel=3,
                 blocks=20,
                 policy_filters=32,
                 value_filters=32,
                 value_hidden=256):

        super(ResNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.input_channels = 34
        # self.args = args
        self.board_size = [self.input_channels, self.board_x, self.board_y]

        # padding to get the same size output only works for odd kernel sizes
        if block_kernel % 2 != 1:
            raise ValueError('block_kernel must be odd, got %d' % block_kernel)
        pad = int(np.floor(block_kernel/2))

        # the starting conv block
        self.conv_block = nn.Conv2d(
            self.board_size[0], block_filters, kernel_size=block_kernel, stride=1, padding=pad)
        self.conv_block_bn = nn.BatchNorm2d(
            block_filters, track_running_stats=self.track_running_stats)

        # the residual blocks
        self.blocks = blocks
        self.res_blocks = nn.ModuleList(
            [ResBlock(block_filters, block_kernel) for i in range(blocks-1)])

        # policy head
        self.policy_conv = nn.Conv2d(
            block_filters, policy_filters, kernel_size=1)
        self.policy_conv_bn = nn.BatchNorm2d(
            policy_filters, track_running_stats=self.track_running_stats)
        # calculate policy output shape to flatten
        pol_shape = (policy_filters, self.board_size[1], self.board_size[2])
        self.policy_flat = int(np.prod(pol_shape))
        # policy layers
        # self.policy_bn = nn.BatchNorm1d(num_features=policy_filters,track_running_stats=False)
        self.policy = nn.Linear(self.policy_flat, self.action_size)
        #self.policy_softmax = torch.nn.Softmax()

        # value head
        self.value_conv = nn.Conv2d(
            block_filters, value_filters, kernel_size=1)
        self.value_conv_bn = nn.BatchNorm2d(
            value_filters, track_running_stats=self.track_running_stats)
        # calculate value shape to flatten
        val_shape = (value_filters, self.board_size[1], self.board_size[2])
        self.value_flat = int(np.prod(val_shape))
        # value layers
        self.value_hidden = nn.Linear(self.value_flat, value_hidden)
        self.value = nn.Linear(value_hidden, 1)

    def forward(self, sAndValids):
        s, valids = sAndValids
        s = s.view(-1, self.input_channels, self.board_x, self.board_y)

        x = F.relu(self.conv_block_bn(self.conv_block(s)))

        for i in range(self.blocks-1):
            x = self.res_blocks[i](x)

        # policy head
        x_pi = self.policy_conv(x)
        x_pi = self.policy_conv_bn(x_pi)
        x_pi = x_pi.view(-1, self.policy_flat)
        x_pi = F.relu(x_pi)
        x_pi = self.policy(x_pi)
        x_pi -= (1-valids)*1000
        # .reshape(self.action_size)  # torch.log_softmax(x_pi, dim=1)
        x_pi = torch.log_softmax(x_pi, dim=1)

        # value head
        x_v = self.value_conv(x)
        x_v = self.value_conv_bn(x_v)
        x_v = F.relu(x_v)
        x_v = x_v.view(-1, self.value_flat)
        x_v = self.value_hidden(x_v)
        x_v = F.relu(x_v)
        x_v = self.value(x_v)
        x_v = torch.tanh(x_v)

        return x_pi, x_v
