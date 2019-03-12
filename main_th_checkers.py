from Coach_th_checkers import Coach
from ThaiCheckers.ThaiCheckersGame import ThaiCheckersGame as Game
from ThaiCheckers.pytorch.NNet import NNetWrapper as nn
from utils import *
import os

"""
Before using multiprocessing, please check 2 things before use this script.
1. The number of PlayPool should not over your CPU's core number.
2. Make sure all Neural Network which each process created can store in VRAM at same time. Check your NN size before use this.
"""

args = dotdict({
    'numIters': 1000,
    'numEps': 500,  # 25000
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 100,  # 1600 , 800
    'arenaCompare': 5,  # 400, 0
    'cpuct': 8,

    'multiGPU': True,
    'setGPU': '0',
    'numSelfPlayPool': 30,
    'numTestPlayPool': 2,

    'checkpoint': '/workspace/CU_Makhos/models_minimax/',
    'load_model': False,
    'load_iter': 160,
    'load_folder_file': '/workspace/CU_Makhos/models_minimax/',
    'numItersForTrainExamplesHistory': 4

})

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    g = Game()
    c = Coach(g, args)
    # if args.load_model:
    #print("Load trainExamples from file")
    # c.loadTrainExamples()
    c.learn_minimax()
