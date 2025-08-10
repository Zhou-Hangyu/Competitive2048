# @misc{thakoor2016learning,
#   title={Learning to play othello without human knowledge},
#   author={Thakoor, Shantanu and Nair, Surag and Jhunjhunwala, Megha},
#   year={2016},
#   publisher={Stanford University, Final Project Report}
# }

import os
import sys

from tqdm import tqdm

sys.path.append("../../../")
from comp2048.utils.utils import AverageMeter
from comp2048.alphazero.NeuralNet import NeuralNet

import torch
import torch.optim as optim

from comp2048.models.Comp2048NNet import Comp2048NNet as cnnet


class NNetWrapper(NeuralNet):
    def __init__(self, game, args):
        self.args = args
        self.nnet = cnnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.seed = self.args.seed
        self.device = self.args.device
        if self.seed is None:
            self.generator = torch.Generator(device=self.device)  # random generator
        else:
            self.generator = torch.Generator(device=self.device).manual_seed(
                self.seed
            )  # random generator

        self.nnet.to(device=self.device)

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())
        # record training info
        record_iter_train = {}
        record_iter_train["batch_size"] = self.args.batch_size
        record_iter_train["batch_count"] = []
        record_iter_train["pi_losses"] = []
        record_iter_train["v_losses"] = []

        for epoch in range(self.args.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.args.batch_size)

            t = tqdm(range(batch_count), desc="Training Net")

            for _ in t:
                sample_ids = torch.randint(
                    high=len(examples),
                    size=(self.args.batch_size,),
                    device=self.device,
                    generator=self.generator,
                )  # randomly select a batch
                boards, pis, vs = list(
                    zip(*[examples[i] for i in sample_ids])
                )  # TODO: torch implementation FIXME
                # print(f"board{torch.FloatTensor(np.array(boards))}")
                # boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                boards = torch.stack(boards, dim=0)
                # target_pis = torch.FloatTensor(np.array(pis))
                # target_vs = torch.FloatTensor(np.array(vs))
                target_pis = torch.stack(pis, dim=0)
                target_vs = torch.cuda.FloatTensor(vs)

                # predict
                boards, target_pis, target_vs = (
                    boards.contiguous().to(device=self.device),
                    target_pis.contiguous().to(device=self.device),
                    target_vs.contiguous().to(device=self.device),
                )

                # compute output
                out_pi, out_v = self.nnet(boards)  # boards is a batch of board layout
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            # collecting training info
            record_iter_train["batch_count"].append(batch_count)
            record_iter_train["pi_losses"].append(pi_losses.avg)
            record_iter_train["v_losses"].append(v_losses.avg)
        return record_iter_train

    def predict(self, board):
        # preparing input
        # board = torch.FloatTensor(board.asType(np.float64))
        board.contiguous().to(self.device)
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        return torch.exp(pi).data.cpu()[0], v.data.cpu()[0]

    def loss_pi(self, targets, outputs):
        """
        Cosine-similarity
        """
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        """
        MSE
        """
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists!")
        torch.save(
            {
                "state_dict": self.nnet.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print("No models in path {}".format(filepath))
            raise ("No models in path {}".format(filepath))
        # map_location = None if self.args.cuda else 'cpu'
        map_location = self.device
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint["state_dict"])
