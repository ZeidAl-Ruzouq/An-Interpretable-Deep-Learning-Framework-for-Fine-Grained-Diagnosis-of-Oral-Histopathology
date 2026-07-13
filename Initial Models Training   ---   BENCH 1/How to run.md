...............

Notes:

- Pipeline was run on Runpod.io for 0.36/hr per RTX 4090 GPU on the community portal

...............

How to run orchid_pipeline for ResNet50 as an example:

...............

Install Libraries:

pip install --upgrade pip
pip install pandas
pip install albumentations opencv-python scikit-learn matplotlib tqdm requests timm
sudo apt-get update && sudo apt-get install -y libgl1
pip install git+https://github.com/jacobgil/pytorch-grad-cam.git
pip install optuna
pip install --force-reinstall numpy==1.26.4
pip install "albumentations<1.4.0"
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pynvml

...............

Prepare Data:

python preparedata.py

...............

Hyperparameters tuning with Optuna:

for i in 0 1 2 3 4; do
  for j in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$i python orchid_pipeline.py \
      --model resnet \
      --epochs 30 \
      --tune \
      --n-trials 1 \
      --device cuda:0 \
      --storage sqlite:///orchid_optuna.db \
      --study-name orchid_study > gpu${i}_trial${j}.log 2>&1 &
    sleep 10
  done
done
wait

...............

View logging of Optuna session example:

while true; do
  line=$(tail -n 1 gpu0.log)
  echo -ne "\rGPU0 | $line"
  sleep 1
done

...............

View progress of Optuna session:

watch -n 30 "optuna trials --storage sqlite:///orchid_optuna.db --study-name orchid_study"

...............

Evauluate performance ( if training session cut off ):

python orchid_pipeline.py \
  --model resnet \
  --bs 16 \
  --eval-only \
  --weights best.pt

...............

Training and further details can be read in the "Readme.md" of each model directory.

........................................................................................................................

Extra Code snippets used:

....................   Resume training from Optuna best trial instead of fresh start: (example) ......................... 

python orchid_pipeline.py \
  --model efficientnet \
  --epochs 100 \
  --bs 16 \
  --lr 0.0001302094624670934 \
  --dropout 0.480084831585933 \
  --wd 1e-4 \
  --resume last.pt

....................   Used to Restart and continue tuning is error occured:   ................................. 

python - <<'EOF'
import optuna
from optuna.trial import TrialState

# Load old study
old_study = optuna.load_study(
    study_name="orchid_study",
    storage="sqlite:///orchid_optuna.db"
)

# Create new clean study
new_storage = "sqlite:///orchid_optuna_clean.db"
new_study = optuna.create_study(
    study_name="orchid_study",
    storage=new_storage,
    direction=old_study.direction
)

# Copy COMPLETE + PRUNED trials
copied = 0
for t in old_study.get_trials(deepcopy=False):
    if t.state in (TrialState.COMPLETE, TrialState.PRUNED):
        new_study.add_trial(t)
        copied += 1

print(f"[INFO] Copied {copied} trials "
      f"(COMPLETE={len([t for t in new_study.trials if t.state == TrialState.COMPLETE])}, "
      f"PRUNED={len([t for t in new_study.trials if t.state == TrialState.PRUNED])}) "
      f"into orchid_optuna_clean.db")
EOF

...............................................................................................................

...............

Pod Environment:

- GPU: RTX 4090 (1x)
- vCPU: 12
- Memory: 31 GB
- Container Disk: 20 GB

Container:

- Image: runpod-torch-v220

- Pod Volume Size: 150 GB

- Mount Path: /workspace

Network:

- Location: US

...............