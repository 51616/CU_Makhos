from collections import deque
from Arena import Arena
from MCTS_th_checkers import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
from ThaiCheckers.ThaiCheckersGame import ThaiCheckersGame as Game
from ThaiCheckers.pytorch.NNet import NNetWrapper as nn
import time
import os
import sys
from pickle import Pickler, Unpickler
from random import shuffle
from torch import multiprocessing
import torch
from tqdm import tqdm
import random
import copy

mp = multiprocessing.get_context('spawn')


def AsyncSelfPlay(nnet, game, args, iter_num, iterr):  # , bar

    # bar.suffix = "iter:{i}/{x} | Total: {total:} | ETA: {eta:}".format(
    #     i=iter_num+1, x=iterr, total=bar.elapsed_td, eta=bar.eta_td)

    # #set gpu
    if(args.multiGPU):
        if(iter_num % 2 == 0):
            torch.cuda.set_device('cuda:1')
        else:
            torch.cuda.set_device('cuda:2')
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

    # create nn and load

    # net = nn(game, (iter_num % 2) + 1)
    # net.nnet.load_state_dict(nnet.nnet.state_dict())

    mcts = MCTS(game, nnet, args)
    # try:
    #     net.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
    # except:
    #     pass
    boardHistory = deque(np.zeros((8, 8, 8), dtype='int'), maxlen=8)
    # histIdx = 0
    trainExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0

    while True:
        episodeStep += 1
        canonicalBoard = game.getCanonicalForm(board, curPlayer)
        boardHistory.append(canonicalBoard)
        # temp = int(episodeStep < args.tempThreshold)
        # print('canonical Board')
        # print(canonicalBoard)
        pi = mcts.getActionProb(boardHistory, temp=1)
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

        trainExamples.append([boardHistory, curPlayer, pi,
                              game.gameState.turn, game.gameState.stale, valids])

        action = np.random.choice(len(pi), p=pi)
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
        print(examplesFile)
    else:
        # print("File with trainExamples found. Read it.")
        with open(examplesFile, "rb") as f:
            for i in Unpickler(f).load():
                trainhistory.append(i)
        # f.closed
    # ----------------------
    # ---delete if over limit---
    if len(trainhistory) > args.numItersForTrainExamplesHistory:
        print("len(trainExamplesHistory) =", len(trainhistory),
              " => remove the oldest trainExamples")
        del trainhistory[len(trainhistory)-1]
    # -------------------
    # ---extend history---
    trainExamples = []
    for e in trainhistory:
        trainExamples.extend(e)
    shuffle(trainExamples)
    # ---save history---
    folder = args.checkpoint
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, 'trainhistory.pth.tar'+".examples")
    with open(filename, "wb+") as f:
        Pickler(f).dump(trainhistory)
        # f.closed
    # ------------------
    nnet.train(trainExamples)
    nnet.save_checkpoint(folder=args.checkpoint,
                         filename='train_iter_' + str(iter_num) + '.pth.tar')


# def AsyncAgainst(game, args, iter_num, bar):
#     bar.suffix = "iter:{i}/{x} | Total: {total:} | ETA: {eta:}".format(
#         i=iter_num+1, x=args.arenaCompare, total=bar.elapsed_td, eta=bar.eta_td)
#     bar.next()

#     # set gpu
#     if(args.multiGPU):
#         if(iter_num % 2 == 0):
#             os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#         else:
#             os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     else:
#         os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

#     # create nn and load
#     nnet = nn(game)
#     pnet = nn(game)
#     try:
#         nnet.load_checkpoint(folder=args.checkpoint, filename='train.pth.tar')
#     except:
#         print("load train model fail")
#         pass
#     try:
#         pnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
#     except:
#         print("load old model fail")
#         pass
#     pmcts = MCTS(game, pnet, args)
#     nmcts = MCTS(game, nnet, args)

#     arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
#                   lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), game)
#     arena.displayBar = False
#     pwins, nwins, draws = arena.playGames(2)
#     return pwins, nwins, draws


# def CheckResultAndSaveNetwork(pwins, nwins, draws, game, args, iter_num):
#     # set gpu
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

#     if pwins+nwins > 0 and float(nwins+(0.5*draws))/(pwins+nwins+draws) < args.updateThreshold:
#         print('REJECTING NEW MODEL')
#     else:
#         print('ACCEPTING NEW MODEL')
#         self.nnet = nn(game)
#         net.load_checkpoint(folder=args.checkpoint, filename='train.pth.tar')
#         net.save_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
#         net.save_checkpoint(folder=args.checkpoint,
#                             filename='checkpoint_' + str(iter_num) + '.pth.tar')


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.nnet = nn(game, gpu_num=0)
        self.nnet1 = nn(game, gpu_num=2)
        self.nnet2 = nn(game, gpu_num=3)
        self.trainExamplesHistory = []
        self.checkpoint_iter = 0

        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0

        self.win_games = []
        self.loss_games = []
        self.draw_games = []

    # def parallel_self_play_process(self):
    #     processes = []
    #     temp = []
    #     result = []

    #     for i in range(self.args.numSelfPlayPool):
    #         p = mp.Process(target=AsyncSelfPlay, args=(
    #             self.nnet, self.game, self.args, i, self.args.numEps))
    #         p.start()
    #         processes.append(p)

    #     for p in processes:
    #         p.join()

    #     for i in processes:
    #         gameplay, r = i.get()
    #         result.append(gameplay)
    #         if (r == 1e-4):
    #             self.draw_count += 1
    #         elif r == 1:
    #             self.win_count += 1
    #         else:
    #             self.loss_count += 1

    #     for i in result:
    #         temp += i
    #     return temp

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
            if i % 2 == 0:
                net = self.nnet1
            else:
                net = self.nnet2

            res.append(pool.apply_async(AsyncSelfPlay, args=(
                net, self.game, self.args, i, self.args.numEps)))  # , bar

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

    def train_network(self, iter_num):

        # print("Start train network")

        torch.cuda.set_device('cuda:0')

        TrainNetwork(self.nnet, self.game, self.args,
                     iter_num, self.trainExamplesHistory)

    # def parallel_self_test_play(self, iter_num):
    #     mp = multiprocessing.get_context('forkserver')
    #     pool = mp.Pool(processes=self.args.numTestPlayPool)
    #     print("Start test play")
    #     bar = Bar('Test Play', max=self.args.arenaCompare)
    #     res = []
    #     result = []
    #     for i in range(self.args.arenaCompare):
    #         res.append(pool.apply_async(
    #             AsyncAgainst, args=(self.game, self.args, i, bar)))
    #     pool.close()
    #     pool.join()

    #     pwins = 0
    #     nwins = 0
    #     draws = 0
    #     for i in res:
    #         result.append(i.get())
    #     for i in result:
    #         pwins += i[0]
    #         nwins += i[1]
    #         draws += i[2]

    #     print("pwin: "+str(pwins))
    #     print("nwin: "+str(nwins))
    #     print("draw: "+str(draws))
    #     pool = mp.Pool(processes=1)
    #     pool.apply_async(CheckResultAndSaveNetwork, args=(
    #         pwins, nwins, draws, self.game, self.args, iter_num,))
    #     pool.close()
    #     pool.join()

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
                    folder=self.args.checkpoint, filename='train_iter_1.pth.tar')
                # self.nnet1.load_state_dict(self.nnet.state_dict())
                # self.nnet2.load_state_dict(self.nnet.state_dict())

            except:
                print("Create a new model")

        for i in range(1, self.args.numIters+1):
            print('------ITER ' + str(i) + '------')

            if i > 1:
                try:
                    # self.nnet = nn(self.game, gpu_num=0)
                    self.nnet.load_checkpoint(
                        folder=self.args.checkpoint, filename='train_iter_'+str(self.checkpoint_iter)+'.pth.tar')

                except Exception as e:
                    print(e)
                    print('train_iter_' + str(self.checkpoint_iter) + '.pth.tar')
                    print('No checkpoint iter')

            self.nnet1.nnet.load_state_dict(self.nnet.nnet.state_dict())
            self.nnet2.nnet.load_state_dict(self.nnet.nnet.state_dict())

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
                self.trainExamplesHistory.append(iterationTrainExamples)

            else:
                win_loss_count = len(self.win_games) + len(self.loss_games)

                sample_draw_games = random.sample(
                    self.draw_games, win_loss_count)  # get samples from draw games

                iterationTrainExamples += sample_draw_games
                print('Too much draw, add all win/loss games and ',
                      str(win_loss_count), ' draw moves')

            self.trainExamplesHistory.append(iterationTrainExamples)

            self.train_network(i)
            self.trainExamplesHistory.clear()

            # self.trainExamplesHistory.append(iterationTrainExamples)
            # self.train_network(i)
            # self.trainExamplesHistory.clear()
            # self.parallel_self_test_play(i)
