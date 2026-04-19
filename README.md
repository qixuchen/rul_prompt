# FedLER

This repository contains code for running FedLER on CMAPSS.
This README focuses on the **main experiment only**: federated prompt training with `train_fed.py`.

## 1) Environment Setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Data Setup

Place CMAPSS files under `data/CMAPSSData/`:

- `train_FD001.txt`, `train_FD002.txt`, `train_FD003.txt`, `train_FD004.txt`
- `test_FD001.txt`, `test_FD002.txt`, `test_FD003.txt`, `test_FD004.txt`
- `RUL_FD001.txt`, `RUL_FD002.txt`, `RUL_FD003.txt`, `RUL_FD004.txt`

For `train_fed.py`, prompt feature files are also required in `feats/`:

- `clip_feature_ts_forcasting.pkl` (when `net.llm: clip`)
- `siglip.pkl` (when `net.llm: siglip`)

## 3) Run the Main Experiment (`train_fed`)

Run one federated experiment with a config in `exps/clip_bilstm/` or `exps/clip_pe_net/`:

```bash
python train_fed.py --cfg exps/clip_bilstm/fed_FD001.yaml
```

You can override the random seed from command line:

```bash
python train_fed.py --cfg exps/clip_bilstm/fed_FD001.yaml --seed 4000
```

Optional: run multiple seeds and save logs:

```bash
mkdir -p exp_results/bilstm_clip
for seed in 4000 5000 6000 7000; do
  python train_fed.py --cfg exps/clip_bilstm/fed_FD001.yaml --seed "$seed" \
    > "exp_results/bilstm_clip/fed_FD001_seed${seed}.txt"
done
```
