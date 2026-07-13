...............

Notes:

Pipeline was run on Runpod.io for 0.36/hr per RTX 4090 GPU on the community portal

...............

How to run ca_gnne.py example:

...............

Install Libraries:

pip install --upgrade pip
pip install pandas
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install albumentations opencv-python scikit-learn matplotlib tqdm requests timm
sudo apt-get update && sudo apt-get install -y libgl1
pip install git+https://github.com/jacobgil/pytorch-grad-cam.git
pip install optuna
pip install --force-reinstall numpy==1.26.4
pip install "albumentations<1.4.0"
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pynvml

...............

Prepare Data (Using orchid_pipeline.py and image prepared from the first benchmark):

First prepare model weights:

Models/:
convnext  deit  efficientnet  regnety  resnet

Models/convnext:
best.pt

Models/deit:
best.pt

Models/efficientnet:
best.pt

Models/regnety:
best.pt

Models/resnet:
best.pt

Then run this to create the data for the GNN:

#!/bin/bash

# List of models
MODELS=("convnext" "resnet" "efficientnet" "deit" "regnety")

# Create main data directory
mkdir -p data

for M in "${MODELS[@]}"; do
  echo "[INFO] Processing $M ..."
  mkdir -p data/$M

  for SPLIT in train val test; do
    echo "  -> Evaluating $SPLIT split"
    CUDA_VISIBLE_DEVICES=0 python orchid_pipeline.py \
      --model $M \
      --weights Models/$M/best.pt \
      --eval-only --split $SPLIT

    # Move the predictions CSV into the right folder with a clear name
    mv predictions/${SPLIT}_${M}.csv data/$M/${SPLIT}_${M}.csv
  done
done

echo "[INFO] All CSVs saved under ./data/<model>/"

...............

Hyperparameters tuning using Optuna:

mkdir -p logs

seq 1 20 | xargs -P7 -I{} sh -c '
  CUDA_VISIBLE_DEVICES=0 python CA-GNN-E_pipeline.py \
    --out_dir outputs_gnne \
    --trials 1 \
    --storage sqlite:///cagne_optuna.db \
    --study-name gnne_study > logs/trial{}.log 2>&1'

...............

View logging of Optuna session example:

while true; do
  line=$(tail -n 1 trial3.log)
  echo -ne "\rGPU0 | $line"
  sleep 1
done

...............

View progress of Optuna session:

watch -n 30 "optuna trials --storage sqlite:///cagne_optuna.db --study-name gnne_study"

...............

Training with Optuna set parameters:

python CA-GNN-E_pipeline.py --out_dir outputs_gnne --params '{"hidden_dim": 64, "dropout": 0.4145042658920881, "lr": 0.007410199847025845, "alpha": 0.6787553989153896, "beta": 0.6428542931143773, "lam": 0.3523423755549125, "batch_size": 64}'

...............

Pod Environment:

GPU: RTX 4090

vCPU: 12

Memory: 31 GB

Container Disk: 20 GB

Pod Volume Size: 150 GB

Mount Path: /workspace

Container:

Image: runpod-torch-v220

Network:

Location: US

...............