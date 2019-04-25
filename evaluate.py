
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
from utils import *

mp = multiprocessing.get_context('spawn')

CPU_NUM = 30
GAME_NUM = 100

GAME = Game()

def AsyncAgainst(nnet, game, args, iter_num):

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    minimax = minimaxAI(game,depth=7)

    local_args = dotdict({'numMCTSSims': 200, 'cpuct': 1.0})
    mcts = MCTS(game, nnet, local_args, eval=True)

    arena = Arena(lambda x: np.argmax(mcts.getActionProb(x, temp=0)),
                  minimax.get_move, game)
    arena.displayBar = False
    net_win, minimax_win, draws = arena.playGames(2)
    return net_win, minimax_win, draws

def parallel_self_test_play(iter_num):
    pool = mp.Pool(processes=CPU_NUM, maxtasksperchild=1)

    res = []
    result = []
    for i in range(GAME_NUM):
        net = nn(GAME, gpu_num=3)
        net.load_checkpoint(folder='/workspace/CU_Makhos/models_minimax/', filename='train_iter_'+str(iter_num)+'.pth.tar')
        res.append(pool.apply_async(
            AsyncAgainst, args=(net, Game(), None, i)))
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

    print("Agent " + str(iter_num) + " win: "+str(pwins)+"\tMinimax win: " +
            str(nwins)+"\tDraws: "+str(draws))

if __name__=='__main__':
    for iter_num in [i*10 for i in range(1,27)]:
        parallel_self_test_play(iter_num)