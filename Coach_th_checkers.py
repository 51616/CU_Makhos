from collections import deque
from Arena import Arena
from MCTS_th_checkers import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
from ThaiCheckers.ThaiCheckersGame import ThaiCheckersGame as Game
from ThaiCheckers.pytorch.NNet import NNetWrapper as nn
from ThaiCheckers.ThaiCheckersPlayers import minimaxAI
import time
import os
import sys
from pickle import Pickler, Unpickler
import pickle
from random import shuffle
from torch import multiprocessing
import torch
from tqdm import tqdm
import random
import copy
from utils_examples_global_avg import build_unique_examples

mp = multiprocessing.get_context('spawn')


def AsyncSelfPlay(nnet, game, args, iter_num):  # , bar

    # bar.suffix = "iter:{i}/{x} | Total: {total:} | ETA: {eta:}".format(
    #     i=iter_num+1, x=iterr, total=bar.elapsed_td, eta=bar.eta_td)

    # set gpu
    if(args.multiGPU):
        if(iter_num % 3 == 0):
            #os.environ["CUDA_VISIBLE_DEVICES"] = '1'
            torch.cuda.device('cuda:1')
        elif (iter_num % 3 == 1):
            #os.environ["CUDA_VISIBLE_DEVICES"] = '2'
            torch.cuda.device('cuda:2')
        else:
            #os.environ["CUDA_VISIBLE_DEVICES"] = '3'
            torch.cuda.device('cuda:3')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

    # create nn and load

    # net = nn(game, (iter_num % 2) + 1)
    # net.nnet.load_state_dict(nnet.nnet.state_dict())

    mcts = MCTS(game, nnet, args)
    # try:
    #     net.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
    # except:
    #     pass
    # boardHistory = deque(np.zeros((8, 8, 8), dtype='int'), maxlen=8)
    # histIdx = 0
    trainExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0

    while True:
        episodeStep += 1
        canonicalBoard = game.getCanonicalForm(board, curPlayer)
        # boardHistory.append(canonicalBoard)
        # temp = int(episodeStep < args.tempThreshold)
        # print('canonical Board')
        # print(canonicalBoard)
        pi = mcts.getActionProb(canonicalBoard, temp=1)
        valids = game.getValidMoves(canonicalBoard, 1)
        # bs, ps = zip(*game.getSymmetries(canonicalBoard, pi))
        # _, valids_sym = zip(
        #     *game.getSymmetries(canonicalBoard, valids))
        # sym = zip(bs, ps, valids_sym)
        # for b, p, valid in data:
        #     trainExamples.append([b, curPlayer, p, game.gameState.turn, game.gameState.stale, valid])
        # data = zip(canonicalBoard, pi, valids)
        # boardHistory[histIdx] = [canonicalBoard, curPlayer, pi,
        #                         game.gameState.turn, game.gameState.stale, valids]
        # if histIdx<7:
        #     histIdx+=1

        trainExamples.append([canonicalBoard, curPlayer, pi,
                              game.gameState.turn, game.gameState.stale, valids])

        action = random.choices(np.arange(0, len(pi)), weights=pi)[0]
        board, curPlayer = game.getNextState(
            board, curPlayer, action)

        # print()
        # print('next_board')
        # print(board)
        # print()
        r = game.getGameEnded(board, curPlayer)  # winner

        if r != 0:
            # print([(x[0], x[2], r*((-1)**(x[1] != curPlayer)))
            #        for x in trainExamples])
            # print([(x[0], r*x[1])
            #        for x in trainExamples])
            # bar.update(1)
            # global draw_count
            # global win_loss_count
            # if r == 1e-4:
            #     draw_count += 1
            # else:
            #     win_loss_count += 1

            return [(x[0], x[2], r*x[1], x[3], x[4], x[5]) for x in trainExamples], r


def AsyncMinimaxPlay(game, args):

    minimax = minimaxAI(game)

    #boardHistory = deque(np.zeros((8, 8, 8), dtype='int'), maxlen=8)
    trainExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0

    while True:
        episodeStep += 1
        canonicalBoard = game.getCanonicalForm(board, curPlayer)
        # boardHistory.append(canonicalBoard)

        pi = minimax.get_pi(canonicalBoard)
        valids = game.getValidMoves(canonicalBoard, 1)

        trainExamples.append([canonicalBoard, curPlayer, pi,
                              game.gameState.turn, game.gameState.stale, valids])

        action = random.choices(np.arange(0, len(pi)), weights=pi)[0]
        board, curPlayer = game.getNextState(
            board, curPlayer, action)

        r = game.getGameEnded(board, curPlayer)  # winner

        if r != 0:

            return [(x[0], x[2], r*x[1], x[3], x[4], x[5]) for x in trainExamples], r


def TrainNetwork(nnet, game, args, iter_num, trainhistory):
    # set gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU
    # create network for training
    # nnet = nn(game)
    # try:
    #     nnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
    # except:
    #     pass
    # ---load history file---
    modelFile = os.path.join(args.checkpoint, "trainhistory.pth.tar")
    examplesFile = modelFile+".examples"
    if not os.path.isfile(examplesFile):
        print('Train history not found')
    else:
        # print("File with trainExamples found. Read it.")
        old_history = pickle.load(open(examplesFile, "rb"))
        for iter_samples in old_history:
            trainhistory.append(iter_samples)
        # f.closed
    # ----------------------
    # ---delete if over limit---
    if len(trainhistory) > args.numItersForTrainExamplesHistory:
        print("len(trainExamplesHistory) =", len(trainhistory),
              " => remove the oldest trainExamples")
        #del trainhistory[len(trainhistory)-1]
        trainhistory = trainhistory[:args.numItersForTrainExamplesHistory]
        print('Length after remove:', len(trainhistory))
    # -------------------
    # ---extend history---
    trainExamples = build_unique_examples(trainhistory)

    # trainExamples = []
    # for e in unique_train_history:
    #     trainExamples.extend(e)
    shuffle(trainExamples)
    print('Total train samples (moves):', len(trainExamples))
    # ---save history---
    folder = args.checkpoint
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, 'trainhistory.pth.tar'+".examples")
    pickle.dump(trainhistory, open(filename, "wb"))

    # with open(filename, "wb") as f:
    #     Pickler(f).dump(trainhistory)
    #     # f.closed
    # ------------------
    nnet.train(trainExamples)

    nnet.save_checkpoint(folder=args.checkpoint,
                         filename='train_iter_' + str(iter_num) + '.pth.tar')


def AsyncAgainst(nnet, game, args, iter_num):

    # set gpu
    if(args.multiGPU):
        if(iter_num % 3 == 0):
            #os.environ["CUDA_VISIBLE_DEVICES"] = '1'
            torch.cuda.device('cuda:1')
        elif (iter_num % 3 == 1):
            #os.environ["CUDA_VISIBLE_DEVICES"] = '2'
            torch.cuda.device('cuda:2')
        else:
            #os.environ["CUDA_VISIBLE_DEVICES"] = '3'
            torch.cuda.device('cuda:3')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

    # create nn and load
    minimax = minimaxAI(game)
    # try:
    #     nnet.load_checkpoint(folder=args.checkpoint,
    #                          filename='train_iter_'+str(iter_num)+'.pth.tar')
    # except:
    #     print("load train model fail")
    #     pass
    # try:
    #     pnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
    # except:
    #     print("load old model fail")
    #     pass

    mcts = MCTS(game, nnet, args, eval=True)

    arena = Arena(lambda x: np.argmax(mcts.getActionProb(x, temp=0)),
                  minimax.get_move, game)
    arena.displayBar = False
    net_win, minimax_win, draws = arena.playGames(2)
    return net_win, minimax_win, draws


# def CheckResultAndSaveNetwork(pwins, nwins, draws, nnet, game, args, iter_num):

#     if pwins+nwins > 0 and float(nwins+(0.5*draws))/(pwins+nwins+draws) < args.updateThreshold:
#         print('REJECTING NEW MODEL')
#     else:
#         print('ACCEPTING NEW MODEL')
#         nnet.load_checkpoint(folder=args.checkpoint,
#                              filename='train_iter_'+str(iter_num)+'.pth.tar')
#         nnet.save_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
#         nnet.save_checkpoint(folder=args.checkpoint,
#                              filename='checkpoint_' + str(iter_num) + '.pth.tar')


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.nnet = nn(game, gpu_num=0)
        self.nnet1 = nn(self.game, gpu_num=1)
        self.nnet2 = nn(self.game, gpu_num=2)
        self.nnet3 = nn(self.game, gpu_num=3)

        state_dict = self.nnet.nnet.state_dict()
        self.nnet1.nnet.load_state_dict(state_dict)
        self.nnet2.nnet.load_state_dict(state_dict)
        self.nnet3.nnet.load_state_dict(state_dict)

        self.trainExamplesHistory = []
        self.checkpoint_iter = 0

        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0

        self.win_games = []
        self.loss_games = []
        self.draw_games = []

    def parallel_self_play(self):
        pool = mp.Pool(processes=self.args.numSelfPlayPool)
        temp = []
        res = []
        result = []

        temp_draw_games = []
        temp_win_games = []
        temp_loss_games = []
        # bar = Bar('Self Play', max=self.args.numEps)
        # bar = tqdm(total=self.args.numEps)
        for i in range(self.args.numEps):
            if i % 3 == 0:
                net = self.nnet1
            elif i % 3 == 1:
                net = self.nnet2
            else:
                net = self.nnet3

            res.append(pool.apply_async(AsyncSelfPlay, args=(
                net, self.game, self.args, i)))

        pool.close()
        pool.join()
        # print("Done self-play")

        for i in res:
            gameplay, r = i.get()
            result.append(gameplay)
            if (r == 1e-4):
                self.draw_count += 1
                temp_draw_games.append(gameplay)
            elif r == 1:
                self.win_count += 1
                temp_win_games.append(gameplay)
            else:
                self.loss_count += 1
                temp_loss_games.append(gameplay)

        for i in result:
            temp += i
        for i in temp_draw_games:
            self.draw_games += i

        for i in temp_win_games:
            self.win_games += i

        for i in temp_loss_games:
            self.loss_games += i
        return temp

    def parallel_minimax_play(self):
        pool = mp.Pool(processes=self.args.numSelfPlayPool)
        temp = []
        res = []
        result = []

        temp_draw_games = []
        temp_win_games = []
        temp_loss_games = []
        # bar = Bar('Self Play', max=self.args.numEps)
        # bar = tqdm(total=self.args.numEps)
        for i in range(self.args.numEps):

            res.append(pool.apply_async(AsyncMinimaxPlay, args=(
                self.game, self.args)))

        pool.close()
        pool.join()
        # print("Done self-play")

        for i in res:
            gameplay, r = i.get()
            result.append(gameplay)
            if (r == 1e-4):
                self.draw_count += 1
                temp_draw_games.append(gameplay)
            elif r == 1:
                self.win_count += 1
                temp_win_games.append(gameplay)
            else:
                self.loss_count += 1
                temp_loss_games.append(gameplay)

        for i in result:
            temp += i
        for i in temp_draw_games:
            self.draw_games += i

        for i in temp_win_games:
            self.win_games += i

        for i in temp_loss_games:
            self.loss_games += i
        return temp

    def parallel_self_test_play(self, iter_num):
        pool = mp.Pool(processes=self.args.numTestPlayPool)

        res = []
        result = []
        for i in range(self.args.arenaCompare):
            if i % 3 == 0:
                net = self.nnet1
            elif i % 3 == 1:
                net = self.nnet2
            else:
                net = self.nnet3

            res.append(pool.apply_async(
                AsyncAgainst, args=(net, self.game, self.args, i)))
        pool.close()
        pool.join()

        pwins = 0
        nwins = 0
        draws = 0
        for i in res:
            result.append(i.get())
        for i in result:
            pwins += i[0]
            nwins += i[1]
            draws += i[2]

        print("NN win: "+str(pwins)+"\tMinimax win: " +
              str(nwins)+"\tDraws: "+str(draws))
        # pool = mp.Pool(processes=1)

        # pool.apply_async(CheckResultAndSaveNetwork, args=(
        #     pwins, nwins, draws, self.game, self.args, iter_num,))
        # pool.close()
        # pool.join()

        # CheckResultAndSaveNetwork(
        #     pwins, nwins, draws, self.nnet, self.game, self.args, iter_num)

    def train_network(self, iter_num):

        # print("Start train network")

        torch.cuda.set_device('cuda:0')

        TrainNetwork(self.nnet, self.game, self.args,
                     iter_num, self.trainExamplesHistory)

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        if self.args.load_model:
            try:
                self.nnet.load_checkpoint(
                    folder=self.args.checkpoint, filename='train_iter_'+str(self.args.load_iter)+'.pth.tar')
                # self.nnet1.load_state_dict(self.nnet.state_dict())
                # self.nnet2.load_state_dict(self.nnet.state_dict())

            except Exception as e:
                print(e)
                print("Create a new model")

        pytorch_total_params = sum(p.numel()
                                   for p in self.nnet.nnet.parameters() if p.requires_grad)

        print('Num trainable params:', pytorch_total_params)

        start_iter = 1
        if self.args.loadmodel:
            start_iter += self.args.load_iter
            self.args.numMCTSSims += self.args.load_iter
            self.args.numItersForTrainExamplesHistory = min(
                20, 4 + (self.args.load_iter-4)//2)

        for i in range(start_iter, self.args.numIters+1):
            if (self.args.numMCTSSims < 400):
                self.args.numMCTSSims += 1
            if ((i > 5) and (i % 2 == 0) and (self.args.numItersForTrainExamplesHistory < 20)):
                self.args.numItersForTrainExamplesHistory += 1
            print('------ITER ' + str(i) + '------' +
                  '\tMCTS sim:' + str(self.args.numMCTSSims) + '\tIter samples :' + str(self.args.numItersForTrainExamplesHistory))

            # if i > 1:
            #     try:
            #         # self.nnet = nn(self.game, gpu_num=0)
            #         self.nnet.load_checkpoint(
            #             folder=self.args.checkpoint, filename='train_iter_'+str(self.checkpoint_iter)+'.pth.tar')

            #     except Exception as e:
            #         print(e)
            #         print('train_iter_' + str(self.checkpoint_iter) + '.pth.tar')
            #         print('No checkpoint iter')

            # print('Check weights')
            # state_dict = self.nnet.nnet.policy.weight.data.cpu().numpy()
            # state_dict1 = self.nnet1.nnet.policy.weight.data.cpu().numpy()
            # print(state_dict == state_dict1)
            # state_dict2 = self.nnet2.nnet.policy.weight.data.cpu().numpy()
            # print(state_dict == state_dict2)
            # state_dict3 = self.nnet3.nnet.policy.weight.data.cpu().numpy()
            # print(state_dict == state_dict3)

            self.win_count = 0
            self.loss_count = 0
            self.draw_count = 0

            self.win_games = []
            self.loss_games = []
            self.draw_games = []

            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            temp = self.parallel_self_play()

            # iterationTrainExamples += temp
            iterationTrainExamples += self.win_games
            iterationTrainExamples += self.loss_games

            print('Win count:', self.win_count, 'Loss count:',
                  self.loss_count, 'Draw count:', self.draw_count)

            self.checkpoint_iter = i

            # games = []
            # games += self.win_games
            # games += self.loss_games

            if self.draw_count <= (self.win_count + self.loss_count):
                iterationTrainExamples += self.draw_games

            else:
                win_loss_count = len(self.win_games) + len(self.loss_games)

                sample_draw_games = random.sample(
                    self.draw_games, win_loss_count)  # get samples from draw games

                iterationTrainExamples += sample_draw_games
                print('Too much draw, add all win/loss games and ',
                      str(win_loss_count), ' draw moves')

            self.trainExamplesHistory.append(iterationTrainExamples)
            self.train_network(i)
            self.nnet1.nnet.load_state_dict(self.nnet.nnet.state_dict())
            self.nnet2.nnet.load_state_dict(self.nnet.nnet.state_dict())
            self.nnet3.nnet.load_state_dict(self.nnet.nnet.state_dict())

            self.trainExamplesHistory.clear()

            # self.trainExamplesHistory.append(iterationTrainExamples)
            # self.train_network(i)
            # self.trainExamplesHistory.clear()
            # self.parallel_self_test_play(i)

    def learn_minimax(self):

        if self.args.load_model:
            try:
                self.nnet.load_checkpoint(
                    folder=self.args.checkpoint, filename='train_iter_'+str(self.args.load_iter)+'.pth.tar')
                # self.nnet1.load_state_dict(self.nnet.state_dict())
                # self.nnet2.load_state_dict(self.nnet.state_dict())

            except Exception as e:
                print(e)
                print("Create a new model")

        pytorch_total_params = sum(p.numel()
                                   for p in self.nnet.nnet.parameters() if p.requires_grad)

        print('Num trainable params:', pytorch_total_params)

        # start_iter = 1
        # if self.args.load_model:
        #     start_iter += self.args.load_iter

        for i in range(1, 30):  # hard code for 30 iters
            print('------ITER ' + str(i) + '------')
            self.win_count = 0
            self.loss_count = 0
            self.draw_count = 0

            self.win_games = []
            self.loss_games = []
            self.draw_games = []

            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            temp = self.parallel_minimax_play()

            # iterationTrainExamples += temp
            iterationTrainExamples += self.win_games
            iterationTrainExamples += self.loss_games
            iterationTrainExamples += self.draw_games

            print('Win count:', self.win_count, 'Loss count:',
                  self.loss_count, 'Draw count:', self.draw_count)

            self.checkpoint_iter = i

            self.trainExamplesHistory.append(iterationTrainExamples)
            self.train_network(i)
            self.nnet1.nnet.load_state_dict(self.nnet.nnet.state_dict())
            self.nnet2.nnet.load_state_dict(self.nnet.nnet.state_dict())
            self.nnet3.nnet.load_state_dict(self.nnet.nnet.state_dict())
            self.parallel_self_test_play(i)
            self.trainExamplesHistory.clear()
