## 2-view Novel View Synthesis Test

Two synchronized cameras (back view and side view) are packed into a dataset for [Neural Body](https://github.com/zju3dv/neuralbody). The player mesh comes from SAM 3D Body. The basketball is saved separately and composited after rendering.

Longer notes: [docs/nvs_two_view.md](docs/nvs_two_view.md).

### 1. Check the pipeline
From the repo root, with Python 3.11:
```bash
python -m pip install -e ".[dev,nvs]"
python preprocess.py --self-check
``` 

### 2. One-time model setup
Use a CUDA build of PyTorch. Then clone the two models next to the code and download the weights: 
```bash
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
python -m pip install -e third_party/sam2

git clone https://github.com/facebookresearch/sam-3d-body.git third_party/sam-3d-body
```

Follow third_party/sam-3d-body/INSTALL.md for that repo’s Python packages. Skip appnope on Windows.

Request access to https://huggingface.co/facebook/sam-3d-body-dinov3, then:
```bash
hf auth login
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
```

SAM 2 base-plus checkpoint:
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

Save it as checkpoints/sam2.1_hiera_base_plus.pt.

Neural Body is a separate Linux environment. Clone it only when you are ready to train. Do not install it into this environment.

### 3. Calibrate both cameras
Film a checkerboard that both cameras see at the same time. Lay it flat on the court. --square-mm is the size of one square. --board-cols and --board-rows are inner corners.
```bash
python scripts/calibrate_stereo.py --video-back data/raw/calib_back.mp4 --video-side data/raw/calib_side.mp4 --square-mm 40 --board-cols 9 --board-rows 6 --output config/cameras.json
```
config/cameras.example.json is only a format sample.

### 4. Preprocess a shot
The two videos must start together and use the same frame rate.
```bash
python preprocess.py --video-back data/raw/back.mp4 --video-side data/raw/side.mp4 --cameras config/cameras.json --output data/nvs/shot --sam2-checkpoint checkpoints/sam2.1_hiera_base_plus.pt --sam3d-checkpoint checkpoints/sam-3d-body-dinov3/model.ckpt --mhr-path checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

If the GPU runs out of memory, split it:
```bash
python preprocess.py --video-back data/raw/back.mp4 --video-side data/raw/side.mp4 --cameras config/cameras.json --output data/nvs/shot --skip-mesh --sam2-checkpoint checkpoints/sam2.1_hiera_base_plus.pt

python preprocess.py --mesh-from-export --output data/nvs/shot --sam3d-checkpoint checkpoints/sam-3d-body-dinov3/model.ckpt --mhr-path checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

Camera 0 is the back view. Camera 1 is the side view. Open data/nvs/shot/vertices/000000.ply in MeshLab, and read summary.json before training.

### 5. Put the ball back on a render
```bash
python scripts/composite_ball.py --render orbit_000010.png --ball data/nvs/shot/ball/000010.npz --cameras data/nvs/shot/cameras_export.json --camera-id cam_back --output composited_000010.png
```

