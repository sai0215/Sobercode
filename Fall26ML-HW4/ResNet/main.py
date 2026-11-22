from ImageUtils import parse_record
from DataReader import load_data, train_vaild_split
from Model import Cifar, VARIANTS

import os
import random
import argparse
import numpy as np
import torch

def configure():
    parser = argparse.ArgumentParser()
    ### YOUR CODE HERE
    parser.add_argument("--batch_size", type=int, default=128, help='training batch size')
    parser.add_argument("--num_classes", type=int, default=10, help='number of classes')
    parser.add_argument("--save_interval", type=int, default=10,
                        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--weight_decay", type=float, default=2e-4, help='weight decay rate')
    parser.add_argument("--modeldir", type=str, default=None,
                        help='model directory (default: runs/<variant>_lr<lr>)')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--variant", type=str, default="resnet18", choices=list(VARIANTS),
                        help='resnet18 | no_residual | no_batchnorm')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=10,
                        help='compute validation / test loss every this many epochs')
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "data", "cifar-10-batches-py"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="continue from <modeldir>/train_state.pt if it exists")
    ### YOUR CODE HERE
    config = parser.parse_args()
    if config.modeldir is None:
        config.modeldir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                                       "{}_lr{}".format(config.variant, config.lr))
    return config

def main(config):
    print("--- Preparing Data ---")

    ### YOUR CODE HERE
    data_dir = config.data_dir
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    ### YOUR CODE HERE

    x_train, y_train, x_test, y_test = load_data(data_dir)
    x_train_new, y_train_new, x_valid, y_valid = train_vaild_split(x_train, y_train)

    model = Cifar(config)

    ### YOUR CODE HERE
    # Train on 45k images; validation (5k held-out) is used to choose the lr,
    # test loss is only logged (every 10 epochs) for the curves in part (e).
    model.train(x_train_new, y_train_new, config.epochs,
                eval_sets={"valid": (x_valid, y_valid), "test": (x_test, y_test)})
    if not model.history["diverged"]:
        model.test_or_validate(x_valid, y_valid, [config.epochs])
        model.test_or_validate(x_test, y_test, [config.epochs])

    ### END CODE HERE

if __name__ == "__main__":
    config = configure()
    # Don't override the GPU that a scheduler (e.g. Slurm on HPRC) already assigned.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(config.gpu))
    main(config)
