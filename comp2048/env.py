from __future__ import print_function
import sys

sys.path.append("../..")
from comp2048.alphazero.Game import Game
import numpy as np
import torch


class Comp2048Game(Game):
    def __init__(self, n=4, seed=0, device="cpu"):
        self.n = n
        self.device = device
        self.seed = seed

    def getInitBoard(self):
        # return initial board (torch board) and score
        return Comp2048Board(
            self.n, seed=self.seed, device=self.device
        ).board, torch.zeros(2, device=self.device)

    def getBoardSize(self):
        # (a,b) tuple
        return self.n, self.n

    def getActionSize(self):
        # return number of actions
        return 4

    def getNextState(self, board, score, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = self._loadBoard(board, score)
        b.execute_move(action, player)

        return torch.clone(b.board), torch.clone(b.score), -player

    def getValidMoves(self, board, score, player):
        # return a fixed size binary vector (0 means invalid, 1 otherwise)
        valids = torch.zeros(self.getActionSize(), device=self.device)
        b = self._loadBoard(board, score)
        legalMoves = b.get_legal_moves()
        valids[legalMoves] = 1
        return valids

    def getGameEnded(self, board, score, player):
        # The return is the reward
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        if player == -1:
            player1 = 0
            player2 = 1
        elif player == 1:
            player1 = 1
            player2 = 0
        b = self._loadBoard(board, score)
        if b.has_legal_moves():
            return 0
        if b.score[player1] > b.score[player2]:
            return -1
        return 1

    def getCanonicalForm(self, board, player):
        # return player * board
        return board  # FIXME: in 2048 there is only one kind of board, no need to flip the board.

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert len(pi) == self.n**2 + 1  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        symmetries = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                symmetries.append((newB, list(newPi.ravel()) + [pi[-1]]))
        return symmetries

    def stringRepresentation(self, board):
        return board.cpu().numpy().tostring()

    # def stringRepresentationReadable(self, board):
    #     board_s = "".join(self.square_content[square] for row in board for square in row)
    #     return board_s

    # def getScore(self, board, score, player):
    #     b = self._loadBoard(board, score)
    #     return b.countDiff(player)

    def _loadBoard(self, board, score):
        b = Comp2048Board(self.n, seed=self.seed, device=self.device)
        b.board = board.detach().clone()
        b.score = score.detach().clone()
        return b

    # square_content = {
    #     -1: "X",
    #     +0: "-",
    #     +1: "O"
    # }
    @staticmethod
    def display(board, score):
        n = board.shape[0]
        print(f"Score: {score[0], score[1]}")
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = int(board[y][x].item())  # get the piece to print
                # print(Comp2048Game.square_content[piece], end=" ")
                print(piece, end=" ")
            print("|")

        print("-----------------------")


class Comp2048Board:
    def __init__(self, size=4, seed=None, device="cpu"):
        "Set up initial board configuration."
        self.seed = seed
        self.device = device
        if self.seed is None:
            self.generator = torch.Generator(device=device)  # random generator
        else:
            self.generator = torch.Generator(device=device).manual_seed(
                self.seed
            )  # random generator

        # Create the empty board array.
        self.board = torch.zeros((size, size), device=self.device)

        # pre-calculate the resulting board layout and score gained for each legal move
        self.board_cache = torch.zeros((4, size, size), device=self.device)
        self.score_cache = torch.zeros(4, device=self.device)

        # Set up the initial 2 tiles
        self.add_random_tile()
        self.add_random_tile()

        # Store each player's score
        self.score = torch.zeros(2, device=self.device)

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.board[index]

    def add_random_tile(self):
        # randomly select an empty tile
        indices = torch.argwhere(self.board == 0)
        random_index = torch.randperm(
            indices.shape[0], generator=self.generator, device=self.device
        )[0]

        # randomly put 2 or 4 (prob: 2->0.9, 4->0.1)
        prob = torch.tensor([0.9, 0.1]).to(device=self.device)
        value = 2 ** (torch.multinomial(prob, 1, generator=self.generator) + 1)
        if torch.sum(self.board) < 0:
            value = -value
        self.board[indices[random_index][0], indices[random_index][1]] = value

    def move(self, action=None):
        """
        Store resulting board layout to the board cache
        return True if the action is valid, otherwise return False
        """
        # start  = time.time()
        current_board = self.board.detach().clone()
        current_board = self.shift(current_board, action)
        if action == 0 or action == 1:
            # Sequential implementation of move.
            for i in range(4):
                #     current_board[i, :] = self.shift_seq(shift_row_col=current_board[i, :], direction=action)
                current_board[i, :] = self.combine_seq(
                    current_row=current_board[i, :], direction=action, action=action
                )
        else:
            current_board = current_board.T
            # Sequential implementation of move.
            for i in range(4):
                #     current_board[i, :] = self.shift_seq(shift_row_col=current_board[i, :], direction=action - 2)
                current_board[i, :] = self.combine_seq(
                    current_row=current_board[i, :], direction=action - 2, action=action
                )
            current_board = current_board.T

        self.board_cache[action, :, :] = current_board
        # print(f"move: {time.time()-start}")
        if torch.sum(current_board != self.board) == 0:
            return False, current_board
        return True, current_board

    def shift(self, board, action):
        """
        Move the tiles to remove inter-tile zero blocks.

        Decode action to move toggles
            * 0 - left - dim=1, descending=True
            * 1 - right - dim=1, descending=False
            * 2 - up - dim=0, descending=True
            * 3 - down - dim=0, descending=False
            action is even --> descending=True
            action > 1 --> dim=0
        """
        dim = (action < 2) * 1
        descending = action % 2 == 0
        index = torch.sort(
            (board != 0) * 1, dim=dim, descending=descending, stable=True
        )[1]
        return board.gather(dim, index)

    def combine(self, board, action):
        if action >= 2:
            board = board.T
        zeros = torch.zeros((4, 1), dtype=int, device=self.device)
        delimiters = -torch.ones((4, 1), dtype=int, device=self.device)
        board = torch.cat((zeros, board, delimiters), dim=1)
        values, count = torch.unique_consecutive(board, return_counts=True)

        if action >= 2:
            board = board.T

    def shift_seq(self, shift_row_col, direction):
        """
        Sequential implementation of shift function.
        """
        zeros = torch.argwhere(shift_row_col == 0)
        new_row_col = torch.zeros(4, device=self.device)
        for i in range(4):
            if shift_row_col[i] != 0:
                if direction == 0:
                    shifts = (zeros < i).sum()
                    new_row_col[i - shifts] = shift_row_col[i]
                if direction == 1:
                    shifts = (zeros > i).sum()
                    new_row_col[i + shifts] = shift_row_col[i]
        return new_row_col

    def combine_seq(self, current_row, direction, action):
        """
        Sequential implementation of combine function.
        """
        current_row = current_row.detach().clone()
        combined = torch.zeros(4, device=self.device)
        # if right, reverse row and do left combining and then reverse back at the end
        if direction == 1:
            current_row = torch.flip(current_row, dims=(0,))
        i = 0
        j = 0
        while i < 3:
            num = current_row[i]
            next_num = current_row[i + 1]
            if num == next_num != 0:
                self.score_cache[action] += (
                    2 * num
                )  # store the score gained from each action
                combined[j] = num * 2
                i += 2
                j += 1
            else:
                combined[j] = num
                i += 1
                j += 1
        if i == 3:
            combined[j] = current_row[i]
        # Get rid of extra empty cells
        # combined = self.shift(combined, 0)
        if direction == 1:
            combined = torch.flip(combined, dims=(0,))
        return combined

    def get_legal_moves(self):
        """Returns all the legal moves"""
        legal_moves = []
        for i in range(4):
            valid, _ = self.move(action=i)
            if valid:
                legal_moves.append(i)
        return legal_moves

    def has_legal_moves(self):
        if len(self.get_legal_moves()) == 0:
            return False
        return True

    def execute_move(self, action, player):
        """Perform the given move on the board; flips pieces as necessary."""
        if player == -1:
            player = 0
        # self.board = self.board_cache[action, :, :].detach().clone()
        _, self.board = self.move(action=action)

        self.score[player] += self.score_cache[
            action
        ]  # add score to the corresponding player
        self.add_random_tile()  # add a random tile

        # reset the board and score cache
        self.board_cache = torch.zeros((4, 4, 4), device=self.device)
        self.score_cache = torch.zeros(4, device=self.device)
