from comp2048.alphazero import Arena
from MCTS import MCTS
from comp2048.env import Comp2048Game
from comp2048.models.Comp2048Players import (
    RandomPlayer,
    GreedyComp2048Player,
    HumanComp2048Player,
)
from comp2048.models.NNet import NNetWrapper as NNet
from comp2048.alphazero.train import args

import torch
from comp2048.utils.utils import dotdict

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
mini_2048 = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True
# human_vs_cpu = False

if mini_2048:
    g = Comp2048Game(4, seed=args.seed, device=args.device)
else:
    g = Comp2048Game(4, seed=args.seed, device=args.device)

# all players
rp = RandomPlayer(g).play
gp = GreedyComp2048Player(g).play
hp = HumanComp2048Player(g).play


# nnet players
n1 = NNet(g, args)
if mini_2048:
    n1.load_checkpoint("./pretrained_models/comp2048/", "6x100x25_best.pth.tar")
else:
    n1.load_checkpoint("./temp_gpu/", "temp.pth.tar")
args1 = dotdict({"numMCTSSims": 50, "cpuct": 1.0})
mcts1 = MCTS(g, n1, args1, seed=args.seed, device=args.device)


def n1p(x, score):
    return torch.argmax(mcts1.getActionProb(x, score, temp=0)).item()


if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g, args)
    n2.load_checkpoint("./temp_gpu/", "temp.pth.tar")
    args2 = dotdict({"numMCTSSims": 50, "cpuct": 1.0})
    mcts2 = MCTS(g, n2, args2, seed=args.seed, device=args.device)

    def n2p(x, score):
        return torch.argmax(mcts2.getActionProb(x, score, temp=0)).item()

    player2 = gp  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, g, display=Comp2048Game.display)
arena.playGames(2, verbose=True)
# print(arena.playGames(2, verbose=True))
