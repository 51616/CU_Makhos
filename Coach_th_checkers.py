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

mp = multiprocessing.get_context('spawn')
win_loss_count = 0
draw_count = 0


def AsyncSelfPlay(net, game, args, iter_num, iterr):  # , bar

    # bar.suffix = "iter:{i}/{x} | Total: {total:} | ETA: {eta:}".format(
    #     i=iter_num+1, x=iterr, total=bar.elapsed_td, eta=bar.eta_td)

    # #set gpu
    # if(args.multiGPU):
    #     if(iter_num%2==0):
    #         os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #     else:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

    # create nn and load
    # net = nn(game)
    mcts = MCTS(game, net, args)
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
        #temp = int(episodeStep < args.tempThreshold)
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
            global draw_count
            global win_loss_count
            if r == 1e-4:
                draw_count += 1
            else:
                win_loss_count += 1
            return [(x[0], x[2], r*x[1], x[3], x[4], x[5]) for x in trainExamples]


def TrainNetwork(nnet, game, args, iter_num, trainhistory):
    # set gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU
    # create network for training
    #nnet = nn(game)
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
        print("File with trainExamples found. Read it.")
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
    nnet.save_checkpoint(folder=args.checkpoint, filename='train.pth.tar')


def AsyncAgainst(game, args, iter_num, bar):
    bar.suffix = "iter:{i}/{x} | Total: {total:} | ETA: {eta:}".format(
        i=iter_num+1, x=args.arenaCompare, total=bar.elapsed_td, eta=bar.eta_td)
    bar.next()

    # set gpu
    if(args.multiGPU):
        if(iter_num % 2 == 0):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

    # create nn and load
    nnet = nn(game)
    pnet = nn(game)
    try:
        nnet.load_checkpoint(folder=args.checkpoint, filename='train.pth.tar')
    except:
        print("load train model fail")
        pass
    try:
        pnet.load_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
    except:
        print("load old model fail")
        pass
    pmcts = MCTS(game, pnet, args)
    nmcts = MCTS(game, nnet, args)

    arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                  lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), game)
    arena.displayBar = False
    pwins, nwins, draws = arena.playGames(2)
    return pwins, nwins, draws


def CheckResultAndSaveNetwork(pwins, nwins, draws, game, args, iter_num):
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

    if pwins+nwins > 0 and float(nwins+(0.5*draws))/(pwins+nwins+draws) < args.updateThreshold:
        print('REJECTING NEW MODEL')
    else:
        print('ACCEPTING NEW MODEL')
        self.nnet = nn(game)
        net.load_checkpoint(folder=args.checkpoint, filename='train.pth.tar')
        net.save_checkpoint(folder=args.checkpoint, filename='best.pth.tar')
        net.save_checkpoint(folder=args.checkpoint,
                            filename='checkpoint_' + str(iter_num) + '.pth.tar')


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.nnet = nn(game)
        self.trainExamplesHistory = []

    def parallel_self_play(self):
        global draw_count
        global win_loss_count
        draw_count = 0
        win_loss_count = 0

        pool = mp.Pool(processes=self.args.numSelfPlayPool)
        temp = []
        res = []
        result = []
        #bar = Bar('Self Play', max=self.args.numEps)
        with tqdm(total=self.args.numEps) as pbar:
            for i in range(self.args.numEps):
                res.append(pool.apply_async(AsyncSelfPlay, args=(
                    self.nnet, self.game, self.args, i, self.args.numEps)))  # , bar
                pbar.update()
        pool.close()
        pool.join()
        for i in res:
            result.append(i.get())
        for i in result:
            temp += i
        return temp

    def train_network(self, iter_num):
        print("Start train network")
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
                    folder=self.args.checkpoint, filename='train.pth.tar')
                print("Load old model")

            except:
                print("Create a new model")

        for i in range(1, self.args.numIters+1):
            print('------ITER ' + str(i) + '------')

            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
            temp = self.parallel_self_play()

            iterationTrainExamples += temp
            #iterationTrainExamples = list(set(iterationTrainExamples))

            print('Win/loss count:', win_loss_count)
            print('Draw Count:', draw_count)

            self.trainExamplesHistory.append(iterationTrainExamples)
            self.train_network(i)
            self.trainExamplesHistory.clear()
            # self.parallel_self_test_play(i)
