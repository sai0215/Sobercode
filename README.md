# HW4 Question 4: ResNet on CIFAR-10, how to run on TAMU HPRC

This README covers **only Question 4** of the homework (Deep Residual Networks for CIFAR-10).

## Where the Question 4 code is

All Question 4 code is in **one folder of this repo: `Fall26ML-HW4/ResNet/`**. Every step below runs from inside that folder.

```
HW-4_ML/                       <- repo root
└── Fall26ML-HW4/
    └── ResNet/                <- QUESTION 4 (cd here)
        ├── main.py            entry point: trains + evaluates one (model, lr) run
        ├── Model.py           training / testing loop (Adam, lr /1.5 every 10 epochs)
        ├── NetWork.py         ResNet18 + "no residual" and "no BatchNorm" variants
        ├── DataReader.py      loads CIFAR-10, 45k train / 5k validation split
        ├── ImageUtils.py      augmentation: pad + random crop, flip, per-channel normalization
        ├── report.py          best lr per model, loss-curve plots, results table
        ├── hprc_job.slurm     the HPRC job script (runs all 9 trainings)
        ├── requirements.txt   Python packages
        └── runs/              results from the local (Mac) runs, for comparison
```

The job trains 9 models: 3 architectures × 3 learning rates, 100 epochs each.

| Model | Learning rates |
|---|---|
| ResNet18 | 0.003, 0.01, 0.03 |
| ResNet18 without residual connections | 0.003, 0.01, 0.03 |
| ResNet18 without BatchNorm | 0.003, 0.01, 0.03 |

Your HPRC results go to a new folder, `runs_hprc/`, next to `runs/`.

---

## Steps

All commands run on the HPRC **login node**. Login nodes have internet access; compute nodes do not, so steps 1–3 must be done there before submitting.

### 1. Log in to HPRC and get the Question 4 code

```bash
ssh <NetID>@grace.hprc.tamu.edu          # on campus or via the TAMU VPN; Duo 2FA required

cd $SCRATCH                              # use scratch, the home quota is too small
git clone <REPO_URL> HW-4_ML             # the repo owner sends you <REPO_URL> + a collaborator invite
cd HW-4_ML/Fall26ML-HW4/ResNet           # <- the Question 4 folder
ls                                       # you should see main.py, hprc_job.slurm, ...
```

The repo is private. If `git clone` asks for a password, use a GitHub *personal access token* (GitHub → Settings → Developer settings → Tokens) instead of your password.

### 2. Create the Python environment (once)

```bash
module purge
module load GCCcore/12.3.0 Python/3.11.3     # check names with: module spider Python
export PIP_CACHE_DIR=$SCRATCH/.pip-cache
python -m venv $SCRATCH/hw4-venv
source $SCRATCH/hw4-venv/bin/activate
pip install -r requirements.txt
```

If you load a different module line, put the same line in the `EDIT ME` block of `hprc_job.slurm`.

### 3. Download CIFAR-10 (once, ~170 MB)

```bash
mkdir -p data && cd data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xzf cifar-10-python.tar.gz && rm cifar-10-python.tar.gz
cd ..                                    # back in Fall26ML-HW4/ResNet
```

### 4. Quick test (1 epoch, about 5 minutes)

```bash
mkdir -p logs
EPOCHS=1 sbatch --array=0 --time=00:30:00 hprc_job.slurm
squeue -u $USER                          # wait until the job is gone
cat logs/*.out                           # should show the GPU name, "Epoch 1 Loss ...", a test accuracy
rm -rf runs_hprc logs/*                  # clean up
```

### 5. Run Question 4

```bash
sbatch hprc_job.slurm                    # submits all 9 runs (1 GPU each)
squeue -u $USER                          # check status
tail -f logs/hw4-resnet_<JOBID>_0.out    # follow one run
```

Each run takes roughly 1 hour on an A100. Task ids 0–8 map to (model, lr) as listed at the top of `hprc_job.slurm`.

Some runs are **expected to fail**. Without BatchNorm at lr 0.01 or 0.03, accuracy stays at 10% (chance level). That is a result to report, not a setup problem.

### 6. Make the report and push the results

```bash
source $SCRATCH/hw4-venv/bin/activate
python report.py --run_dir runs_hprc     # -> runs_hprc/results.md + runs_hprc/figures/
git add runs_hprc && git commit -m "Q4: HPRC runs" && git push
```

Model checkpoints (`*.ckpt`) are excluded by `.gitignore`, so only the small result files are pushed.

### 7. Compare with the local results

| | local (Mac) | HPRC |
|---|---|---|
| results table | `results.md` | `runs_hprc/results.md` |
| loss curves | `figures/` | `runs_hprc/figures/` |

The code, seed and settings are the same on both, so the numbers should be close, but not identical, because the GPUs differ.

---

**If something goes wrong**
- *Job rejected because of the account:* uncomment the `--account` line in `hprc_job.slurm`; `myproject` shows your account number.
- *Waiting a long time for A100s:* change `--gres=gpu:a100:1` to `gpu:t4:1` or `gpu:rtx:1` in `hprc_job.slurm`.
- *`No such file ... cifar-10-batches-py`:* step 3 was not run inside `Fall26ML-HW4/ResNet`.
- *A run stopped early (crash or time limit):* resubmit just that task, e.g. `sbatch --array=2 hprc_job.slurm`. It resumes from its last 10-epoch checkpoint (`train_state.pt`) instead of starting over.
