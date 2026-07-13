...............

Notes:

Not all logging was captured due to session interruption from the main website

Pipeline was run on Runpod.io for 0.36/hr per RTX 4090 GPU on the community portal

This is the slides we used for our test:

Slides:
MDOSCC  Normal  PDOSCC  WDOSCC

Slides/MDOSCC:
C3L-00997-21.svs  C3L-00997-24.svs

Slides/Normal:
3846cc4e-d0d0-4780-afb0-4bb0649ac62e.svs  60949963-7d24-4a7f-81c9-1502d08fe7d4.svs  7694ac39-5b57-44ad-a256-ff4b063c0521.svs  d6a027ee-4efa-49fc-9c46-d9f1f5f5371d.svs
388da34b-1c39-4572-8e31-6f7df8422a2f.svs  61f7212f-176f-4cc8-a4a7-7e2fe2fd4aa4.svs  78752dcb-8a25-496d-8c73-d1c804cc557a.svs  eb4dbda3-7bcd-47c9-b4f4-8e17fcdc3879.svs
39f8ea94-1611-483d-956d-5718a9f0385b.svs  6a5b5f72-da3d-48c5-8eab-3c14d524f9fd.svs  a3312371-5d88-4b7b-8ce2-eb155da18910.svs  ee8520ce-96f4-4153-96ca-d3812828fd12.svs
485c0785-38b7-4573-aa1e-5727d8b285af.svs  6b46dfa4-20c8-4b60-b421-13fa6b9634bc.svs  b66aa295-4cef-4105-aede-07352a1cab32.svs  fec4bef6-dd5e-4a1b-9d35-6c1727db9706.svs

Slides/PDOSCC:
C3L-02621-21.svs  C3L-02621-24.svs

Slides/WDOSCC:
C3L-00977-22.svs  C3L-00977-26.svs  C3L-00995-22.svs

...............

Install Libraries:

...............

pip install torch torchvision torchaudio timm albumentations opencv-python openslide-python pandas numpy scikit-learn matplotlib tqdm requests
apt-get update && apt-get install -y libopenslide0 libopenslide-dev && pip install openslide-python
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
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

Running generalization test: (Note: you need to download OSCC slides from "https://www.cancerimagingarchive.net/collection/cptac-hnscc/" and place them into Slides/{CLASS}, use cptac_case_labels.csv to locate and find case_ids)

python TCGA_Test.py all

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