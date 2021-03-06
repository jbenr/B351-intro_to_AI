#!/usr/bin/python3

### CSCI-B 351 / COGS-Q 351 Spring 2020
### Framework code copyright 2020 B351/Q351 instruction team.
### Do not copy or redistribute this code without permission
### and do not share your solutions outside of this class.
### Doing so constitutes academic misconduct and copyright infringement.

import math
import random
from board import Board

class BasePlayer:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    ##################
    #      TODO      #
    ##################
    # Assign integer scores to the three terminal states
    # P2_WIN_SCORE < TIE_SCORE < P1_WIN_SCORE
    # Access these with "self.TIE_SCORE", etc.
    P1_WIN_SCORE = 1000000
    P2_WIN_SCORE = -1000000
    TIE_SCORE =  0
    # Returns a heuristic for the board position
    # Good positions for 0 pieces should be positive and
    # good positions for 1 pieces should be negative
    # for all boards, P2_WIN_SCORE < heuristic(b) < P1_WIN_SCORE
    def heuristic(self, board):
        score = board.p1_pot - board.p2_pot
        for i in range(0,5):
            if board.p1_pits[i] == 0:
                for j in range(0,i):
                    if board.p1_pits[j] == (i-j):
                        score += board.p2_pits[5-i]
            if board.p2_pits[i] == 0:
                for j in range(0,i):
                    if board.p2_pits[j] == (i-j):
                        score -= board.p1_pits[5-i]

            if board.p1_pits[i] == 6-i:
                score += 1
            if board.p2_pits[i] == 6-i:
                score -= 1

            if (6-i) > board.p1_pits[i]:
                score += board.p1_pits[i] - (6-i)

        return score

    # You are not expected to implement anything here.
    def findMove(self, trace):
        raise NotImplementedError

class ManualPlayer(BasePlayer):
    def __init__(self, max_depth=None):
        BasePlayer.__init__(self, max_depth)

    def findMove(self, trace):
        board = Board(trace)
        opts = "  "
        for c in range(6):
            opts += " "+(str(c+1) if board.isValidMove(c) else ' ')+"  "

        while True:
            if(board.turn == 0):
                print("\n")
                board.printSpaced()
                print(opts)
                pit = input("Pick a pit (P1 side): ")
            else:
                print("\n")
                print(" " + opts[::-1])
                board.printSpaced()
                pit = input("Pick a pit (P2 side): ")
            try: pit = int(pit) - 1
            except ValueError: continue
            if board.isValidMove(pit):
                return pit

class RandomPlayer(BasePlayer):
    def __init__(self, max_depth=None):
        BasePlayer.__init__(self, max_depth)
        self.random = random.Random(max_depth)
    def findMove(self, trace):
        board = Board(trace)
        options = list(board.getAllValidMoves())
        return self.random.choice(options)

class RemotePlayer(BasePlayer):
    def __init__(self, max_depth=None):
        BasePlayer.__init__(self, max_depth)
        self.instructor_url = "http://silo.cs.indiana.edu:30005"
        if self.max_depth > 8:
            print("It refused to go that hard. Sorry.")
            self.max_depth = 8
    def findMove(self, trace):
        import requests
        r = requests.get(f'{self.instructor_url}/getmove/{self.max_depth},{trace}')
        move = int(r.text)
        return move


class PlayerMM(BasePlayer):
    ##################
    #      TODO      #
    ##################
    # performs minimax on board with depth.
    # returns the best move and best score as a tuple
    def minimax(self, board, depth):
        MaxVal = -self.P1_WIN_SCORE
        MinVal = -self.P2_WIN_SCORE
        move = 0

        if board.game_over:
            if board.winner == 0:
                score = board.p1_pot
                self.P1_WIN_SCORE = score
                return None, self.P1_WIN_SCORE
            else:
                score = board.p2_pot
                self.P2_WIN_SCORE = score
                return None, self.P2_WIN_SCORE

        if depth == 0 and (board.game_over == False):
            return None, self.heuristic(board)
        else:
            moves = board.getAllValidMoves()
            if board.turn == 0:
                for i in moves: 
                    board.makeMove(i)
                    val = self.minimax(board, depth-1)[1]
                    board.undoMove()
                    if val > MaxVal:
                        MaxVal = val
                        move = i
                return move, MaxVal
            else:
                for i in moves:
                    board.makeMove(i)
                    val = self.minimax(board, depth-1)[1]
                    board.undoMove()
                    if val < MinVal:
                        MinVal = val
                        move = i
                return move, MinVal

    def findMove(self, trace):
        board = Board(trace)
        move, score = self.minimax(board, self.max_depth)
        return move

class PlayerAB(BasePlayer):
    ##################
    #      TODO      #
    ##################
    # performs minimax with alpha-beta pruning on board with depth.
    # alpha represents the score of max's current strategy
    # beta  represents the score of min's current strategy
    # in a cutoff situation, return the score that resulted in the cutoff
    # returns the best move and best score as a tuple
    def alphaBeta(self, board, depth, alpha, beta):
        MaxVal = -self.P1_WIN_SCORE
        MinVal = -self.P2_WIN_SCORE
        max_move = 0
        min_move = 0

        if board.game_over:
            winner = board.winner
            if winner == 0:
                score = board.p1_pot
                self.P1_WIN_SCORE = score
                return None, self.P1_WIN_SCORE
            else:
                score = board.p2_pot
                self.P2_WIN_SCORE = score
                return None, self.P2_WIN_SCORE

        if depth == 0 and (board.game_over == False):
            return None, self.heuristic(board)
        else:
            moves = board.getAllValidMoves()
            if board.turn == 0:
                for i in moves: 
                    board.makeMove(i)
                    val = self.alphaBeta(board, depth-1, MaxVal, MinVal)[1]
                    board.undoMove()
                    if val > MaxVal:
                        MaxVal = val
                        max_move = i
                        alpha = max(alpha, MaxVal)
                        if beta <= alpha:
                            return None, MaxVal
                return max_move, MaxVal
            else:
                for i in moves:
                    board.makeMove(i)
                    val = self.alphaBeta(board, depth-1, MaxVal, MinVal)[1]
                    board.undoMove()
                    if val < MinVal:
                        MinVal = val
                        min_move = i
                        beta = min(beta, MinVal)
                        if beta <= alpha:
                            return None, MinVal
                return min_move, MinVal

    def findMove(self, trace):
        board = Board(trace)
        move, score = self.alphaBeta(board, self.max_depth, -math.inf, math.inf)
        return move

class PlayerDP(PlayerAB):
    ''' A version of PlayerAB that implements dynamic programming
        to cache values for its heuristic function, improving performance. '''
    def __init__(self, max_depth):
        PlayerAB.__init__(self, max_depth)
        self.resolved = {}

    ##################
    #      TODO      #
    ##################
    # if a saved heuristic value exists in self.resolved for board.state, returns that value
    # otherwise, uses BasePlayer.heuristic to get a heuristic value and saves it under board.state
    def heuristic(self, board):
        if board.state in self.resolved:
            return self.resolved.get(board.state)
        else:
            self.resolved.update({board.state: BasePlayer.heuristic(self, board)})
            return self.heuristic(board)

#######################################################
###########Example Subclass for Testing
#######################################################

# This will inherit your findMove from above, but will override the heuristic function with
# a new one; you can swap out the type of player by changing X in "class TestPlayer(X):"
class TestPlayer(BasePlayer):
    # define your new heuristic here
    def heuristic(self):
        pass


