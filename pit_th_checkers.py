from Arena import Arena
from MCTS_th_checkers import MCTS
from ThaiCheckers.ThaiCheckersGame import ThaiCheckersGame, display
from ThaiCheckers.ThaiCheckersPlayers import *
from ThaiCheckers.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *
from torch import multiprocessing

# mp = multiprocessing.get_context('forkserver')
# pool = mp.Pool(processes=self.args.numSelfPlayPool)

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""


def Async_Play(game, args, iter_num, bar):
    # bar.suffix = "iter:{i}/{x} | Total: {total:} | ETA: {eta:}".format(i=iter_num+1,x=args.numPlayGames,total=bar.elapsed_td, eta=bar.eta_td)
    # bar.next()

    # set gpu
    # if(args.multiGPU):
    #     if(iter_num%2==0):
    #         os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #     else:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # else:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

    # create NN
    model1 = NNet(game)
    model2 = NNet(game)

    # try load weight
    try:
        model1.load_checkpoint(folder=args.model1Folder,
                               filename=args.model1FileName)
    except:
        print("load model1 fail")
        pass
    try:
        model2.load_checkpoint(folder=args.model2Folder,
                               filename=args.model2FileName)
    except:
        print("load model2 fail")
        pass

    # create MCTS
    mcts1 = MCTS(game, model1, args)
    mcts2 = MCTS(game, model2, args)

    # each process play 2 games
    arena = Arena(lambda x: np.argmax(mcts1.getActionProb(x, temp=0)),
                  lambda x: np.argmax(mcts2.getActionProb(x, temp=0)), game)
    arena.displayBar = False
    oneWon, twoWon, draws = arena.playGames(2)
    return oneWon, twoWon, draws


if __name__ == "__main__":
    """
    Before using multiprocessing, please check 2 things before use this script.
    1. The number of PlayPool should not over your CPU's core number.
    2. Make sure all Neural Network which each process created can store in VRAM at same time. Check your NN size before use this.
    """
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

    def ParallelPlay(g):
        bar = Bar('Play', max=args.numPlayGames)
        res = []
        result = []
        for i in range(args.numPlayGames):
            res.append(pool.apply_async(Async_Play, args=(g, args, i, bar)))
        pool.close()
        pool.join()

        oneWon = 0
        twoWon = 0
        draws = 0
        for i in res:
            result.append(i.get())
        for i in result:
            oneWon += i[0]
            twoWon += i[1]
            draws += i[2]
        print("Model 1 Win:", oneWon, " Model 2 Win:", twoWon, " Draw:", draws)

    g = ThaiCheckersGame()
    # parallel version
    # ParallelPlay(g)

    # single process version
    # all players
    rp = RandomPlayer(g).play
    # gp = GreedyOthelloPlayer(g).play
    # hp = HumanOthelloPlayer(g).play
    minimax = minimaxAI(game=g, depth=5).get_move
    # nnet players
    n1 = NNet(g, gpu_num=0)
    n1.load_checkpoint('/workspace/CU_Makhos/models/',
                       'train_iter_190.pth.tar')
    args1 = dotdict({'numMCTSSims': 200, 'cpuct': 1.0})
    mcts1 = MCTS(g, n1, args1, eval=True)
    def n1p(x): return np.random.choice(
        32*32, p=mcts1.getActionProb(x, temp=1))

    # n2 = NNet(g)
    # n2.load_checkpoint('temp/', 'train.pth.tar')
    # args2 = dotdict({'numMCTSSims': 400, 'cpuct': 3.0})
    # mcts2 = MCTS(g, n2, args2)
    # def n2p(x): return np.random.choice(
    #     32*32, p=mcts2.getActionProb(x, temp=1))

    # player1 = {'func': n1p, 'name': 'NNet'}
    # player2 = {'func': minimax, 'name': 'minimax'}

    arena = Arena(n1p, minimax, g, display=display)
    print(arena.playGames(2, verbose=True))
