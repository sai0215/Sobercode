import os
import json
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from NetWork import ResNet, resnet18
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

VARIANTS = {
    "resnet18": dict(residual=True, batchnorm=True),
    "no_residual": dict(residual=False, batchnorm=True),
    "no_batchnorm": dict(residual=True, batchnorm=False),
}


def make_batch(x, training):
    """Run parse_record on every row of x and stack into a [N, 3, 32, 32] tensor."""
    return torch.from_numpy(np.stack([parse_record(r, training) for r in x]).astype(np.float32))


class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = resnet18(**VARIANTS[config.variant]).to(DEVICE)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.network.parameters(),  \
                             self.config.lr, weight_decay=self.config.weight_decay)
        # lr scheduler: every 10 epochs the learning rate is divided by 1.5
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=1 / 1.5)
        self.history = {"config": vars(config), "train_loss": [], "eval_epochs": [],
                        "valid_loss": [], "valid_acc": [], "test_loss": [], "test_acc": [],
                        "lr": [], "diverged": False}

    def train(self, x_train, y_train, max_epoch, eval_sets=None):
        """eval_sets: optional {"valid": (x, y), "test": (x, y)}; their loss and
        accuracy are computed every `eval_interval` epochs (and never used for training)."""
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size

        print('### Training... ###')
        start_epoch = self.try_resume() if getattr(self.config, "resume", False) else 0
        for epoch in range(start_epoch + 1, max_epoch + 1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Manually update or use scheduler from pytorch
            # (StepLR is stepped once at the end of every epoch, see below)
            lr = self.optimizer.param_groups[0]["lr"]
            self.network.train()
            epoch_losses = []
            ### YOUR CODE HERE

            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                lo, hi = i * self.config.batch_size, (i + 1) * self.config.batch_size
                x_batch = make_batch(curr_x_train[lo:hi], training=True).to(DEVICE)
                y_batch = torch.from_numpy(curr_y_train[lo:hi].astype(np.int64)).to(DEVICE)
                loss = self.loss(self.network(x_batch), y_batch)
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss = loss.item()
                epoch_losses.append(loss)
                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)

            self.scheduler.step()
            mean_loss = float(np.mean(epoch_losses))
            self.history["train_loss"].append(mean_loss)
            self.history["lr"].append(lr)
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} lr {:.6f} Duration {:.3f} seconds.'.format(
                epoch, mean_loss, lr, duration), flush=True)

            if not math.isfinite(mean_loss):
                # Training blew up (can happen without BN / residuals at large lr):
                # the weights are NaN from here on, so further epochs are pointless.
                print('Loss is not finite -- stopping this run.', flush=True)
                self.history["diverged"] = True
                break

            if eval_sets and epoch % self.config.eval_interval == 0:
                self.history["eval_epochs"].append(epoch)
                msg = []
                for name, (x, y) in eval_sets.items():
                    l, a = self.evaluate(x, y)
                    self.history[name + "_loss"].append(l)
                    self.history[name + "_acc"].append(a)
                    msg.append('{} loss {:.4f} acc {:.4f}'.format(name, l, a))
                print('  [eval @ epoch {:d}] {}'.format(epoch, ' | '.join(msg)), flush=True)
                self.save_history()

            if epoch % self.config.save_interval == 0 or epoch == max_epoch:
                self.save(epoch)
                self.save_state(epoch)

        self.save_history()
        ### YOUR CODE HERE
        #Plot the curves
        self.plot_curves()
        ### YOUR CODE HERE

    @torch.no_grad()
    def evaluate(self, x, y, batch_size=1000):
        """Mean cross-entropy loss and accuracy of the current network on (x, y)."""
        self.network.eval()
        total_loss, correct = 0.0, 0
        for lo in range(0, x.shape[0], batch_size):
            xb = make_batch(x[lo:lo + batch_size], training=False).to(DEVICE)
            yb = torch.from_numpy(y[lo:lo + batch_size].astype(np.int64)).to(DEVICE)
            logits = self.network(xb)
            total_loss += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
            correct += (logits.argmax(1) == yb).sum().item()
        self.network.train()
        return total_loss / x.shape[0], correct / x.shape[0]

    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        accs = {}
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)

            ### YOUR CODE HERE
            _, acc = self.evaluate(x, y)
            ### END CODE HERE

            print('Test accuracy: {:.4f}'.format(acc))
            accs[checkpoint_num] = acc
        return accs

    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
        # Keep only the last few checkpoints to save disk space.
        old = os.path.join(self.config.modeldir, 'model-%d.ckpt' % (epoch - 3 * self.config.save_interval))
        if os.path.exists(old):
            os.remove(old)

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        self.network.to(DEVICE)
        print("Restored model parameters from {}".format(checkpoint_name))

    # ---- full training state, so a crashed / time-limited run can continue (--resume) ----
    def _state_path(self):
        return os.path.join(self.config.modeldir, 'train_state.pt')

    def save_state(self, epoch):
        os.makedirs(self.config.modeldir, exist_ok=True)
        tmp = self._state_path() + '.tmp'
        torch.save({"epoch": epoch,
                    "network": self.network.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "history": self.history,
                    "np_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, self._state_path())  # atomic: a crash never leaves a half-written file

    def try_resume(self):
        """Load train_state.pt if present; return the last completed epoch (0 = fresh start)."""
        path = self._state_path()
        if not os.path.exists(path):
            return 0
        s = torch.load(path, map_location="cpu", weights_only=False)
        self.network.load_state_dict(s["network"])
        self.network.to(DEVICE)
        self.optimizer.load_state_dict(s["optimizer"])
        self.scheduler.load_state_dict(s["scheduler"])
        self.history = s["history"]
        np.random.set_state(s["np_rng"])
        torch.set_rng_state(s["torch_rng"])
        print("Resumed from {} (epoch {})".format(path, s["epoch"]), flush=True)
        return s["epoch"]

    def save_history(self):
        os.makedirs(self.config.modeldir, exist_ok=True)
        with open(os.path.join(self.config.modeldir, 'history.json'), 'w') as f:
            json.dump(self.history, f)

    def plot_curves(self):
        h = self.history
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], label="train loss (every epoch)")
        if h["eval_epochs"]:
            ax.plot(h["eval_epochs"], h["test_loss"], "o--", label="test loss (every 10 epochs)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy loss")
        ax.set_title("{} lr={}".format(self.config.variant, self.config.lr))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.config.modeldir, 'loss_curves.png'), dpi=150)
        plt.close(fig)
