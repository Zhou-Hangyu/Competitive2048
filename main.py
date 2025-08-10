from tkinter import Tk, Frame, Label, Canvas, StringVar
from comp2048 import Comp2048Game, Comp2048Board
from comp2048.models.NNet import NNetWrapper as NNet
from comp2048.alphazero.train import args
from comp2048.alphazero import MCTS
from comp2048.utils.utils import dotdict
import torch
import time

KEY2NUM = {"Left": 0, "Right": 1, "Up": 2, "Down": 3}
DURATION = 0


class User:
    def __init__(self, game):
        self.game = game

    def play(self, board, score, event=None):
        valid = self.game.getValidMoves(board, score, 1)
        while True:
            input_move = int(input())
            if input_move in range(4) and valid[input_move]:
                break
        return input_move


class Comp2048(Tk):
    def __init__(self):
        Tk.__init__(self)

        self.game_board = Comp2048Board(seed=args.seed, device=args.device)
        self.game = Comp2048Game(n=4, seed=args.seed, device=args.device)

        # Set up
        self.agent_score = StringVar()
        self.player_score = StringVar()
        self.move_initiated = False

        # get two players ready
        self.player1 = User(self.game).play
        nnet = NNet(self.game, args)
        if args.device == "cpu":
            nnet.load_checkpoint("comp2048/alphazero/temp_cpu/", "best_cpu.pth.tar")
        else:
            nnet.load_checkpoint("comp2048/alphazero/temp_gpu/", "best_gpu.pth.tar")
        args2 = dotdict({"numMCTSSims": 250, "cpuct": 1.0})
        # TODO: visual bar to change numMCTSSims to control difficulty level.
        mcts = MCTS.MCTS(self.game, nnet, args2, seed=args.seed, device=args.device)
        self.player2 = lambda x, score: torch.argmax(
            mcts.getActionProb(x, score, temp=0)
        ).item()

        # display
        # TODO: resizable window
        self.cellwidth = 105
        self.cellheight = 105
        self.header_frame = Frame(self)
        self.header_frame.grid(row=2, column=0, columnspan=4)
        # Button(self.header_frame, text="Let AI Agent Move", command=self.agentMove).grid(row=0, column=0)
        self.header_frame.pack(side="top")

        Label(self.header_frame, text="Player:", font=("times new roman", 15)).grid(
            row=0, column=1
        )
        Label(
            self.header_frame,
            textvariable=self.player_score,
            font=("times new roman", 15),
        ).grid(row=0, column=2)
        Label(self.header_frame, text="Agent:", font=("times new roman", 15)).grid(
            row=0, column=3
        )
        Label(
            self.header_frame,
            textvariable=self.agent_score,
            font=("times new roman", 15),
        ).grid(row=0, column=4)

        self.canvas = Canvas(
            self, width=410, height=410, borderwidth=5, highlightthickness=0
        )
        self.canvas.pack(side="top", fill="both", expand="false")

        # ==== create new game
        self.new_game()

    def playGame(self, event):
        """
        One turn. Always start with User, because this function is called by keyboard event.
        """
        # check game ended or not
        end = self.game.getGameEnded(
            self.game_board.board, self.game_board.score, player=1
        )
        if end == 0:
            if self.move_initiated:
                return
            self.move_initiated = True
            # User makes a move
            key = event.keysym
            # check if the input is valid or not
            if key in KEY2NUM.keys():
                valids = self.game.getValidMoves(
                    self.game_board.board, self.game_board.score, 1
                )
                move = KEY2NUM[event.keysym]
                # check if the move is legal or not
                if valids[move] == 1:
                    self.game_board.board, self.game_board.score, self.curPlayer = (
                        self.game.getNextState(
                            self.game_board.board, self.game_board.score, -1, move
                        )
                    )
                    self.update_board()
                    # display updated scores
                    self.agent_score.set(str(int(self.game_board.score[1].item())))
                    self.player_score.set(str(int(self.game_board.score[0].item())))
                    # update board appearance
                    self.update_idletasks()
                    self.update()
                    time.sleep(DURATION)
                    # AI agent makes a move
                    move = self.player2(self.game_board.board, self.game_board.score)
                    self.game_board.board, self.game_board.score, self.curPlayer = (
                        self.game.getNextState(
                            self.game_board.board, self.game_board.score, 1, move
                        )
                    )
                    self.update_board()
                    # display updated scores
                    self.agent_score.set(str(int(self.game_board.score[1].item())))
                    self.player_score.set(str(int(self.game_board.score[0].item())))
                    # update board appearance
                    self.update_idletasks()
                    self.update()
                self.move_initiated = False
        # handle game ending
        else:
            self.game_ended()
            self.update_idletasks()
            self.update()
            time.sleep(3)
            if end == 1:
                self.game_won()
            elif end == -1:
                self.game_lose()

    # update board layout
    def update_board(self):
        for column in range(4):
            for row in range(4):
                x1 = column * self.cellwidth
                y1 = row * self.cellheight
                x2 = x1 + self.cellwidth - 5
                y2 = y1 + self.cellheight - 5
                num = int(self.game_board.board[row, column].item())
                if num == 0:
                    self.show_number0(x1, y1, x2, y2)
                else:
                    self.show_number(x1, y1, x2, y2, num)

    # show board block when it is empty
    def show_number0(self, a, b, c, d):
        self.canvas.create_rectangle(
            a, b, c, d, fill="#f5f5f5", tags="rect", outline=""
        )

    # show board number
    def show_number(self, a, b, c, d, num):
        bg_color = {
            "2": "#eee4da",
            "4": "#ede0c8",
            "8": "#edc850",
            "16": "#edc53f",
            "32": "#f67c5f",
            "64": "#f65e3b",
            "128": "#edcf72",
            "256": "#edcc61",
            "512": "#f2b179",
            "1024": "#f59563",
            "2048": "#edc22e",
        }
        color = {
            "2": "#776e65",
            "4": "#f9f6f2",
            "8": "#f9f6f2",
            "16": "#f9f6f2",
            "32": "#f9f6f2",
            "64": "#f9f6f2",
            "128": "#f9f6f2",
            "256": "#f9f6f2",
            "512": "#776e65",
            "1024": "#f9f6f2",
            "2048": "#f9f6f2",
        }
        self.canvas.create_rectangle(
            a, b, c, d, fill=bg_color[str(num)], tags="rect", outline=""
        )
        self.canvas.create_text(
            (a + c) / 2,
            (b + d) / 2,
            font=("Arial", 36),
            fill=color[str(num)],
            text=str(num),
        )

    # create a new game
    def new_game(self):
        self.game_board.board, self.game_board.score = self.game.getInitBoard()
        self.update_board()

    # ==== check for game over
    def game_ended(self):
        gameover = [
            [
                "G",
                "A",
                "M",
                "E",
            ],
            ["", "", "", ""],
            ["O", "V", "E", "R"],
            ["", "", "", ""],
        ]

        for column in range(4):
            for row in range(4):
                a = column * self.cellwidth
                b = row * self.cellheight
                c = a + self.cellwidth - 5
                d = b + self.cellheight - 5
                self.canvas.create_rectangle(
                    a, b, c, d, fill="#ede0c8", tags="rect", outline=""
                )
                self.canvas.create_text(
                    (a + c) / 2,
                    (b + d) / 2,
                    font=("Arial", 36),
                    fill="#494949",
                    text=gameover[row][column],
                )

    def game_won(self):
        congrat = [
            [
                "Y",
                "O",
                "U",
                "",
            ],
            ["", "", "", ""],
            ["W", "O", "N", "!"],
            ["", "", "", ""],
        ]
        for column in range(4):
            for row in range(4):
                a = column * self.cellwidth
                b = row * self.cellheight
                c = a + self.cellwidth - 5
                d = b + self.cellheight - 5
                self.canvas.create_rectangle(
                    a, b, c, d, fill="#ede0c8", tags="rect", outline=""
                )
                self.canvas.create_text(
                    (a + c) / 2,
                    (b + d) / 2,
                    font=("Arial", 36),
                    fill="#494949",
                    text=congrat[row][column],
                )

    def game_lose(self):
        congrat = [
            [
                "Y",
                "O",
                "U",
                "",
            ],
            ["", "", "", ""],
            ["L", "O", "S", "E", "!"],
            ["", "", "", ""],
        ]
        for column in range(4):
            for row in range(4):
                a = column * self.cellwidth
                b = row * self.cellheight
                c = a + self.cellwidth - 5
                d = b + self.cellheight - 5
                self.canvas.create_rectangle(
                    a, b, c, d, fill="#ede0c8", tags="rect", outline=""
                )
                self.canvas.create_text(
                    (a + c) / 2,
                    (b + d) / 2,
                    font=("Arial", 36),
                    fill="#494949",
                    text=congrat[row][column],
                )


if __name__ == "__main__":
    app = Comp2048()
    app.wm_title("Competitive 2048")
    app.minsize(430, 470)
    while True:
        app.bind_all("<Key>", app.playGame)
        app.update_idletasks()
        app.update()
