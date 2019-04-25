from Arena import Arena
from MCTS_th_checkers import MCTS
from ThaiCheckers.ThaiCheckersGame import ThaiCheckersGame, display
from ThaiCheckers.ThaiCheckersPlayers import *
from ThaiCheckers.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *
from torch import multiprocessing


if __name__ == "__main__":

    args = dotdict({
        'numMCTSSims': 10,
        'cpuct': 1,

        'multiGPU': False,  # multiGPU only support 2 GPUs.
        'setGPU': '0',
        # total num should x2, because each process play 2 games.
        'numPlayGames': 10,
        'numPlayPool': 5,   # num of processes pool.

        'model1Folder': '/workspace/CU_Makhos/models/',
        'model1FileName': 'best.pth.tar',
        'model2Folder': '/workspace/CU_Makhos/models/',
        'model2FileName': 'best.pth.tar',

    })

    g = ThaiCheckersGame()
    minimax = minimaxAI(game=g, depth=7).get_move
    # nnet players
    n1 = NNet(g, gpu_num=0)
    n1.load_checkpoint('models_minimax/','train_iter_268.pth.tar')
    args1 = dotdict({'numMCTSSims': 100, 'cpuct': 1.0})
    mcts1 = MCTS(g, n1, args1, eval=True, verbose=True)
    def n1p(x): return np.random.choice(
        32*32, p=mcts1.getActionProb(x, temp=0))

    n2 = NNet(g, gpu_num=0)
    n2.load_checkpoint('models_minimax/','train_iter_140.pth.tar')
    args2 = dotdict({'numMCTSSims': 100, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2, eval=True)
    def n2p(x): return np.random.choice(
        32*32, p=mcts2.getActionProb(x, temp=0))

    arena = Arena(n1p, n2p, g, display=display)
    print(arena.playGames(2, verbose=True))
