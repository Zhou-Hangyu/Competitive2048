import logging
import os
import sys
from json import dumps
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import torch
from tqdm import tqdm

from .Arena import Arena
from .MCTS import MCTS
from comp2048.models.Comp2048Players import *

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in train.py.
    """

    def __init__(self, game, nnet, args, seed=None, device='cpu'):
        self.game = game
        self.nnet = nnet  # the new network
        self.pnet = self.nnet.__class__(self.game, args)  # the previous network
        self.args = args
        self.seed = seed
        if self.seed == None:
            self.generator = torch.Generator(device=device)  # random generator
        else:
            self.generator = torch.Generator(device=device).manual_seed(self.seed)  # random generator

        self.device = device
        self.mcts = MCTS(self.game, self.nnet, self.args, seed=self.seed, device=self.device)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        trainExamples: (canonicalBoard, currPlayer, pi,v)
        Returns:
            a list of examples of the form (canonicalBoard, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board, score = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer) # FIXME
            # canonicalBoard = board
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, score, temp=temp)
            # sym = self.game.getSymmetries(canonicalBoard, pi)
            # for b, p in sym:
            # trainExamples.append([b, self.curPlayer, p, None])
            trainExamples.append([canonicalBoard, self.curPlayer, pi, None])

            # action = np.random.choice(len(pi), p=pi)
            action = torch.multinomial(pi, 1, generator=self.generator).item()
            # board, score, self.curPlayer = self.game.getNextState(canonicalBoard, score, self.curPlayer, action) # FIXME
            board, score, self.curPlayer = self.game.getNextState(board, score, self.curPlayer, action)

            r = self.game.getGameEnded(board, score, self.curPlayer)

            # add game results to the training data
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        record_train = {}  # record data for analysis
        record_train['general_training_setting'] = {
            'numIters': self.args.numIters,
            'numEps': self.args.numEps,  # Number of complete self-play games to simulate during a new iteration.
            'tempThreshold': self.args.tempThreshold,  #
            'updateThreshold': self.args.updateThreshold,
            # During arena playoff, new neural net will be accepted if threshold or more of games are won.
            'maxlenOfQueue': self.args.maxlenOfQueue,  # Number of game examples to train the neural networks.
            'numMCTSSims': self.args.numMCTSSims,
            # Number of MCTS simulations (till reach a leaf node) getActionProb will perform each time.
            'arenaCompare': self.args.arenaCompare,
            # Number of games to play during arena play to determine if new net will be accepted.
            'cpuct': self.args.cpuct,
            'checkpoint': self.args.checkpoint,
            'load_model': self.args.load_model,
            'load_folder_file': self.args.load_folder_file,
            'numItersForTrainExamplesHistory': self.args.numItersForTrainExamplesHistory,
            # Number of iterations' training examples to save.
            'verbose': self.args.verbose,  # Toggle for terminal game display
            'seed': self.args.seed,
            'device': self.args.device,
        }
        record_train['nn_training_setting'] = {
            'lr': self.args.lr,
            'dropout': self.args.dropout,
            'epochs': self.args.epochs,
            'batch_size': self.args.batch_size,
            'num_channels': self.args.num_channels
        }
        record_train['train_info'] = []
        # baseline comparison
        record_train['win_rate_self'] = []  # win rate against old model
        record_train['win_rate_random'] = []  # win rate against random baseline
        record_train['win_rate_greedy'] = []  # win rate against greedy baseline
        record_train['win_rate_mcts'] = []  # win rate against original mcts baseline TODO
        rp = RandomPlayer(self.game).play
        gp = GreedyComp2048Player(self.game).play

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            record_iter_all = {}
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    # reset the MCTS tree for a new episode (a new self-play game)
                    self.mcts = MCTS(self.game, self.nnet, self.args, seed=self.seed,
                                     device=self.device)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # the examples were collected using the models from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)  # flatten the examples
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args, seed=self.seed, device=self.device)

            record_iter_train = self.nnet.train(trainExamples)
            record_train['train_info'].append(record_iter_train)
            nmcts = MCTS(self.game, self.nnet, self.args, seed=self.seed, device=self.device)

            # baseline comparison
            if self.args.baseline:
                log.info('PITTING AGAINST RANDOM POLICY')
                arena = Arena(lambda x, score: torch.argmax(nmcts.getActionProb(x, score, temp=0)).item(),
                              rp,
                              self.game,
                              display=lambda board: self.game.display(board))
                nwins, rpwins, draws = arena.playGames(self.args.arenaCompare, verbose=self.args.verbose)

                log.info('NEW/RANDOM WINS : %d / %d ; DRAWS : %d' % (nwins, rpwins, draws))
                record_train['win_rate_random'].append(nwins / self.args.arenaCompare)

                log.info('PITTING AGAINST GREEDY POLICY')
                arena = Arena(lambda x, score: torch.argmax(nmcts.getActionProb(x, score, temp=0)).item(),
                              gp,
                              self.game,
                              display=lambda board: self.game.display(board))
                nwins, gpwins, draws = arena.playGames(self.args.arenaCompare, verbose=self.args.verbose)

                log.info('NEW/GREEDY WINS : %d / %d ; DRAWS : %d' % (nwins, gpwins, draws))
                record_train['win_rate_greedy'].append(nwins / self.args.arenaCompare)

            # model update check
            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x, score: torch.argmax(pmcts.getActionProb(x, score, temp=0)).item(),
                          lambda x, score: torch.argmax(nmcts.getActionProb(x, score, temp=0)).item(),
                          self.game,
                          display=lambda board: self.game.display(board))
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare, verbose=self.args.verbose)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            record_train['win_rate_self'].append(float(nwins) / (pwins + nwins))

            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            if i % 5 == 0:
                json = dumps(record_train)
                f = open("temp_gpu/train_data_"+str(i)+".json", "w")
                f.write(json)
                f.close()

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the models were already collected (loaded)
            self.skipFirstSelfPlay = True
