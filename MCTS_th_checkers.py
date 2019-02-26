import math
import numpy as np
from collections import deque
EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, boardHistory, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        #print('len getprob:', len(boardHistory))  
        cur_move = self.game.gameState.turn
        cur_stale = self.game.gameState.stale
        for i in range(self.args.numMCTSSims):
            self.search(boardHistory)
            self.game.gameState.turn = cur_move
            self.game.gameState.stale = cur_stale

        s = self.game.stringRepresentation(boardHistory)
        #print('string rep:',s)
        counts = [self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA] = 1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs

    def search(self, boardHistory):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        # print('canonicalBoard')
        # print(canonicalBoard)
        # print()
        s = self.game.stringRepresentation(boardHistory)
        canonicalBoard = boardHistory[-1]
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)

        if self.Es[s] != 0:
            # # terminal node
            # print('terminal')
            # print('terminal')
            # print('terminal')
            # print('terminal')
            # print('terminal')
            # print('terminal')
            # print('terminal')
            # print('terminal')
            # print('terminal')
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s], v = self.nnet.predict(
                boardHistory, self.game.gameState.turn, self.game.gameState.stale, valids)
            # valids = self.game.getValidMoves(canonicalBoard, 1)
            # self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            # if sum_Ps_s > 0:
            #     self.Ps[s] /= sum_Ps_s    # renormalize
            if sum_Ps_s <= 0:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            # print('return not in ps')
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct*self.Ps[s][a] * \
                        math.sqrt(self.Ns[s])/(1+self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * \
                        self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        # print()
        # print('sim board')
        # print(next_s)
        # print()
        next_s = self.game.getCanonicalForm(next_s, next_player)
        newHistory = boardHistory.copy()
        newHistory.append(next_s)
        
        v = self.search(newHistory)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                self.Qsa[(s, a)] + v)/(self.Nsa[(s, a)]+1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        # print('return end')
        return -v
