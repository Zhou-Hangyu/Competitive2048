import logging
from json import dumps
import coloredlogs

from comp2048.alphazero.Coach import Coach
from comp2048 import Comp2048Game as Game
from comp2048.models.NNet import NNetWrapper as nn
from comp2048.utils.utils import *
import sys
sys.setrecursionlimit(1000)

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 200,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 50,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of MCTS simulations (till reach a leaf node) getActionProb will perform each time.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    # 'numIters': 20,
    # 'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    # 'tempThreshold': 30,        #
    # 'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    # 'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    # 'numMCTSSims': 5,          # Number of MCTS simulations (till reach a leaf node) getActionProb will perform each time.
    # 'arenaCompare': 6,         # Number of games to play during arena play to determine if new net will be accepted.
    # 'cpuct': 1,

    # sanity test
    # 'numIters': 2,
    # 'numEps': 1,              # Number of complete self-play games to simulate during a new iteration.
    # 'tempThreshold': 15,        #
    # 'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    # 'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    # 'numMCTSSims': 2,          # Number of MCTS simulations (till reach a leaf node) getActionProb will perform each time.
    # 'arenaCompare': 4,         # Number of games to play during arena play to determine if new net will be accepted.
    # 'cpuct': 1,

    'checkpoint': './temp_gpu/',
    'load_model': False,
    'load_folder_file': ('./temp_gpu/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 30,  # Number of iterations' training examples to save.

    'verbose': False,        # Toggle for terminal game display

    'lr': 0.001,    # neural network hyperparameters
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    # 'cuda': True,
    'num_channels': 512,

    # 'seed': 0,
    'seed': None,
    # 'device': 'cuda',
    'device': 'cpu',

    'baseline': True,  # whether to compare with baseline models or not during training.

})

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(n=4, seed=args.seed, device=args.device)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g, args)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args, seed=args.seed, device=args.device)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
