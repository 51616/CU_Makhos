import numpy as np
import time
import copy
from .preprocessing import move_to_index, index_to_move
from .ThaiCheckersLogic import Board
from random import randint


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanThaiCheckersPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.n), int(i % self.game.n))
        while True:
            a = input()

            x, y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x != -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class minimaxAI:

    def __init__(self, game, side=1, depth=5):
        #self.total_time_elapsed = 0
        #self.skipping_point = None
        #self.pruned = 0
        #self.num_move_called = 0
        self.side = side
        self.depth = depth
        self.game = game

    def set_side(self, side=1):
        self.side = side

    def get_move(self, checkers):
        board = Board(checkers[-1], 1, self.game.turn, self.game.stale)
        #self.num_move_called += 1
        #start_time = time.time()
        (start_point, end_point) = self.minimax_start(board, self.depth, True)
        #self.total_time_elapsed += time.time() - start_time
        return move_to_index((start_point, end_point))

    def minimax_start(self, checkers, depth, maximizing_player):
        alpha = float('-inf')
        beta = float('inf')

        possible_moves = None
        possible_moves = checkers.get_legal_moves()

        if len(possible_moves) == 0:
            return None

        heuristics = list()
        for move in possible_moves:
            temp_checkers = copy.deepcopy(checkers)
            temp_checkers.make_move(move)
            heuristics.append(self.minimax(
                temp_checkers, depth-1, not maximizing_player, alpha, beta))

        max_heuristics = float('-inf')

        for heu in heuristics:
            if heu > max_heuristics:
                max_heuristics = heu
        i = 0
        while i < len(heuristics):
            heu = heuristics[i]
            if heu < max_heuristics:
                del heuristics[i]
                del possible_moves[i]
                i -= 1
            i += 1

        return possible_moves[randint(0, len(possible_moves) - 1)]

    def minimax(self, checkers, depth, maximizing_player, alpha, beta):
        if depth == 0:
            return self.get_heuristic(checkers)

        possible_moves = checkers.get_legal_moves()

        value = 0
        temp_checkers = None
        if maximizing_player:
            value = float('-inf')
            for move in possible_moves:
                temp_checkers = copy.deepcopy(checkers)
                temp_checkers.make_move(move)
                result = self.minimax(
                    temp_checkers, depth - 1, not maximizing_player, alpha, beta)
                value = max(result, value)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break

        else:
            value = float('inf')
            for move in possible_moves:
                temp_checkers = copy.deepcopy(checkers)
                temp_checkers.make_move(move)
                result = self.minimax(
                    temp_checkers, depth - 1, not maximizing_player, alpha, beta)
                value = min(result, value)
                beta = min(beta, value)

                if alpha >= beta:
                    break

        return value

    def get_heuristic(self, checkers):
        result = checkers.calculate_score(self.side)
        return result
