from .ThaiCheckersNNet import ThaiCheckersNNet as thcheckersnet
from .ThaiCheckersNNet import ResNet
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from NeuralNet import NeuralNet
from pytorch_classification.utils import Bar, AverageMeter
from utils import *
import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
from tqdm import tqdm
sys.path.append('../../')


args = dotdict({
    'lr': 0.02,
    'dropout': 0.3,
    'epochs': 2,
    'batch_size': 1024,  # 4096
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,  # 256
    'num_blocks': 20
})

#torch.backends.cudnn.benchmark = True


class NNetWrapper(NeuralNet):
    def __init__(self, game, gpu_num):
        self.gpu_num = gpu_num

        self.device = f'cuda:{self.gpu_num}'

        torch.cuda.set_device(self.device)
        self.game = game
        self.nnet = ResNet(game, block_filters=args.num_channels,
                           block_kernel=3, blocks=args.num_blocks).to(self.device).eval()

        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.optimizer = optim.Adam(
            self.nnet.parameters(), lr=args.lr, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[50, 150, 200, 250, 300], gamma=0.5)
        self.min_batch_size = 100
        # if args.cuda:
        #     self.nnet.cuda()
        # self.nnet.cuda()
        # self.nnet.eval()
        self.nnet.share_memory()

    def train(self, past_examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # self.nnet.train()
        self.scheduler.step()
        for epoch in range(args.epochs):
            examples = random.sample(past_examples, len(past_examples)//8)
            print('EPOCH ::: ' + str(epoch+1))

            # data_time = AverageMeter()
            # batch_time = AverageMeter()
            # pi_losses = AverageMeter()
            # v_losses = AverageMeter()
            # end = time.time()

            # bar = Bar('Training Net', max=int(len(examples)/args.batch_size)+1)
            number_of_batches = int(math.ceil(len(examples)/args.batch_size))
            bar = tqdm(total=number_of_batches)
            batch_idx = 0

            train_pi_loss = 0
            train_v_loss = 0

            while (batch_idx < number_of_batches):
                start = batch_idx*args.batch_size
                if (batch_idx+1)*args.batch_size >= len(examples):
                    end = len(examples)-1
                else:
                    end = (batch_idx+1)*args.batch_size

                if end-start < self.min_batch_size:  # minimum size of a batch
                    number_of_batches -= 1
                    break

                else:
                    boards, pis, vs, turns, stales, valids = list(
                        zip(*examples[start:end]))

                # print(boards)
                # print(turns)
                # print(stales)
                stacked_board = []
                for i in range(len(boards)):
                    stacked_board.append(self.convertToModelInput(
                        boards[i], turns[i], stales[i]))

                boards = torch.as_tensor(
                    np.array(stacked_board), dtype=torch.float32).to(self.device)

                target_pis = torch.as_tensor(
                    np.array(pis), dtype=torch.float32).to(self.device)
                target_vs = torch.as_tensor(
                    np.array(vs), dtype=torch.float32).to(self.device)
                valids = torch.as_tensor(
                    np.array(valids), dtype=torch.float32).to(self.device)

                # compute output
                out_pi, out_v = self.nnet((boards, valids))
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                train_pi_loss += l_pi.item()
                train_v_loss += l_v.item()

                # record loss
                # pi_losses.update(l_pi.item(), boards.size(0))
                # v_losses.update(l_v.item(), boards.size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # measure elapsed time
                # batch_time.update(time.time() - end)
                # end = time.time()

                batch_idx += 1

                # plot progress
                bar.update(1)
            #     bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
            #         batch=batch_idx,
            #         size=int(len(examples)/args.batch_size)+1,
            #         data=data_time.avg,
            #         bt=batch_time.avg,
            #         total=bar.elapsed_td,
            #         eta=bar.eta_td,
            #         lpi=pi_losses.avg,
            #         lv=v_losses.avg,
            #     )
            #     bar.next()
            # bar.finish()
            bar.close()
            print('Training Pi loss:', train_pi_loss/number_of_batches,
                  'Training V loss:', train_v_loss/number_of_batches)

        # Validation
        val_pi_loss = 0
        val_v_loss = 0

        # self.nnet.eval()

        val_examples = random.sample(past_examples, len(past_examples)//2)

        batch_idx = 0
        number_of_batches = int(
            math.ceil(len(val_examples)/args.batch_size))

        while batch_idx < number_of_batches:

            start = batch_idx*args.batch_size
            if (batch_idx+1)*args.batch_size >= len(val_examples):
                end = len(val_examples)-1
            else:
                end = (batch_idx+1)*args.batch_size

            if end-start < self.min_batch_size:  # minimum size of a batch
                number_of_batches -= 1
                break

            else:
                boards, pis, vs, turns, stales, valids = list(
                    zip(*val_examples[start:end]))

            stacked_board = []
            for i in range(len(boards)):
                stacked_board.append(self.convertToModelInput(
                    boards[i], turns[i], stales[i]))

            boards = torch.as_tensor(
                np.array(stacked_board), dtype=torch.float32).cuda()

            target_pis = torch.as_tensor(
                np.array(pis), dtype=torch.float32).cuda()
            target_vs = torch.as_tensor(
                np.array(vs), dtype=torch.float32).cuda()
            valids = torch.as_tensor(
                np.array(valids), dtype=torch.float32).cuda()

            # compute output

            with torch.no_grad():
                out_pi, out_v = self.nnet((boards, valids))
                val_pi_loss += self.loss_pi(target_pis, out_pi).item()
                val_v_loss += self.loss_v(target_vs, out_v).item()

            batch_idx += 1

        print('Val Pi loss:', val_pi_loss/number_of_batches,
              'Val V loss:', val_v_loss/number_of_batches)
        print()

    def predict(self, board, turn, stale, valids):
        """
        board: np array with board
        """

        board = self.convertToModelInput(board, turn, stale)
        # preparing input
        board = torch.as_tensor(board, dtype=torch.float32).to(self.device)
        valids = torch.as_tensor(valids, dtype=torch.float32).to(self.device)
        # self.nnet.eval()
        pi, v = self.nnet((board, valids))

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print("No model in path {}".format(filepath))
            raise ValueError("No model in path {}".format(filepath))

        # , map_location=torch.device('cuda')
        checkpoint = torch.load(filepath)

        self.nnet.load_state_dict(checkpoint['state_dict'], strict=False)
        # self.nnet.to(self.device).eval()
        # self.nnet.share_memory()

        # self.optimizer = optim.Adam(
        #     self.nnet.parameters(), lr=args.lr, weight_decay=0.0001)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, milestones=[100, 500, 1000], gamma=0.1)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        print('Latest model loaded')
        # print(self.optimizer.state_dict())
        # print(self.scheduler.state_dict())

    def convertToModelInput(self, board, turn, stale):

        out = []

        for i in range(len(board)):
            currentplayer_pieces = np.zeros((8, 8))
            currentplayer_kings = np.zeros((8, 8))
            other_pieces = np.zeros((8, 8))
            other_kings = np.zeros((8, 8))
            currentplayer_pieces[board[i] == 1] = 1
            currentplayer_kings[board[i] == 3] = 1
            other_pieces[board[i] == -1] = 1
            other_kings[board[i] == -3] = 1
            out.append(currentplayer_pieces)
            out.append(currentplayer_kings)
            out.append(other_pieces)
            out.append(other_kings)

        turn_plane = np.zeros((8, 8))
        stale_plane = np.zeros((8, 8))
        stale_plane[:, :] = stale
        turn_plane[:, :] = turn
        out.append(stale_plane)
        out.append(turn_plane)

        return np.array(out)
