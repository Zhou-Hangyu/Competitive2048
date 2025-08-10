"""
Citation:
@misc{thakoor2016learning,
  title={Learning to play othello without human knowledge},
  author={Thakoor, Shantanu and Nair, Surag and Jhunjhunwala, Megha},
  year={2016},
  publisher={Stanford University, Final Project Report}
}
"""

import torch


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board, score):
        a = torch.randint(self.game.getActionSize(), size=(1,)).item()
        valids = self.game.getValidMoves(board, score, 1)
        while valids[a] != 1:
            a = torch.randint(self.game.getActionSize(), size=(1,)).item()
        return a


class HumanComp2048Player:
    def __init__(self, game):
        self.game = game

    def play(self, board, score):
        # display(board)
        valid = self.game.getValidMoves(board, score, 1)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i / self.game.n), int(i % self.game.n), end="] ")
        while True:
            input_move = int(input())
            if input_move in range(4):
                try:
                    if valid[input_move]:
                        break
                except ValueError:
                    # Input needs to be an integer
                    "Invalid integer"
            print("Invalid move, please input a move within 0, 1, 2, 3")
        return input_move


class GreedyComp2048Player:
    def __init__(self, game):
        self.game = game

    def play(self, board, score):
        valids = self.game.getValidMoves(board, score, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            _, score, _ = self.game.getNextState(board, score, 1, a)
            candidates += [(-score[1], a)]
        candidates.sort()
        return candidates[0][1]
