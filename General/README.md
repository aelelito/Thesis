# SAM3 → SAM-3D-Objects Pipeline

3D object dimension estimation from monocular camera images, built on top of SAM3 segmentation masks from an automotive nuScenes-format dataset.

---

## Overview

```
nuScenes frames (CAM_FRONT)
        │
        ▼
[Step 1] SAM3 video segmentation
        │  conda env: sam3
        │  script:    SAM3/get_sam3_masks_for_nuscenes_trainval.py
        │  output:    .npy files  (one per class per scene)
        ▼
[Step 2] SAM-3D-Objects reconstruction
        │  conda env: sam3d-objects
        │  script:    SAM3D/run_sam3d_objects.py
        │  output:    3D bounding box dimensions  (+optional .ply Gaussian Splat)
        ▼
[Future] LiDAR scale anchoring  →  metric bounding boxes
[Future] SAM-3D-Body for pedestrians
[Future] Temporal accumulation across frames
```

---

## Directory structure

```
Thesis/Development/
│
├── SAM3/
│   ├── get_sam3_masks_for_nuscenes_trainval.py   ← Step 1 (run SAM3)
│   ├── visualize_sam3_results.py
│   └── print_sam3_masks.py
│
└── SAM3D/
    ├── run_sam3d_objects.py                       ← Step 2 (run 3D reconstruction)
    ├── README.md                                  ← this file
    ├── setup_sam3d_objects.sh                     ← one-time install script
    └── sam-3d-objects/                            ← cloned Meta repo (do not edit)
        ├── checkpoints/hf/                        ← model weights (downloaded by setup)
        ├── notebook/
        │   └── inference.py                       ← Inference class used by run script
        └── sam3d_objects/                         ← model source code
```

---

## Conda environments

| Env | Python | PyTorch | Used for |
|-----|--------|---------|----------|
| `sam3` | 3.12 | 2.7.0+cu126 | Step 1 — SAM3 video segmentation |
| `sam3d-objects` | 3.11 | 2.5.1+cu121 | Step 2 — SAM-3D-Objects reconstruction |

The two envs are intentionally separate because pytorch3d / flash_attn / kaolin are pinned to torch 2.5.1/cu121 and are incompatible with the torch 2.7 in the SAM3 env.

---

## Step 1 — Run SAM3 segmentation

### Configuration (edit at top of script)

```python
results_root       = '/media/.../SAM3_Visualizations/Strassbourg'
data_root          = '/media/.../output2/ecp2nuscenes'
scene_token_filter = 'scene-euro-citystrasbourg-...'   # which scene to process
TEXT_PROMPTS       = ["car"]   # one prompt per run: "car", "pedestrian", etc.
start_frame        = 1400      # which frame to start from
num_frames_to_process = 300    # how many frames to process in one go
```

### Run

```bash
conda activate sam3
python SAM3/get_sam3_masks_for_nuscenes_trainval.py
```

### Output

```
{results_root}/scene_{idx}_cam_0_sam3_outputs__{prompt}.npy
```

Each `.npy` file is a Python dict saved with `np.save(..., allow_pickle=True)`:

```python
data = np.load("scene_10_cam_0_sam3_outputs__car.npy", allow_pickle=True).item()
# data is a dict:  { frame_idx (int) : outputs_dict }

outputs = data[500]
outputs["out_obj_ids"]      # (N,)      int64  — instance track IDs
outputs["out_probs"]        # (N,)      float32 — detection confidence 0..1
outputs["out_boxes_xywh"]   # (N, 4)   float32 — normalised [cx, cy, w, h]
outputs["out_binary_masks"] # (N, H, W) bool   — one binary mask per instance
```

`N` = number of objects detected in this frame. Image resolution: H=1024, W=1920.

---

## Step 2 — Run SAM-3D-Objects reconstruction

### Configuration (edit at top of script)

```python
SAM3_MASKS_FILE    = "/media/.../scene_10_cam_0_sam3_outputs__car.npy"
NUSCENES_SCENE_IDX = 10      # nusc.scene[] index matching the masks file
START_FRAME        = 547     # first frame to process (None = from beginning)
N_FRAMES           = 3       # number of frames to process
MIN_PROB           = 0.6     # minimum SAM3 detection confidence

SAVE_PLY           = False   # True: save .ply for every object (large: ~24 MB each)
SAVE_PLY_FRAME     = 547     # save .ply only for this one frame; None to disable
OUTPUT_DIR         = Path("/media/.../SAM3D_Outputs/Strassbourg")
```

To match `SAM3_MASKS_FILE` to the correct scene:

| masks file | NUSCENES_SCENE_IDX |
|---|---|
| `scene_10_cam_0_sam3_outputs__car.npy` | 10 |
| `scene_11_cam_0_sam3_outputs__car.npy` | 11 |
| `scene_15_cam_0_sam3_outputs__car.npy` | 15 |

### Run

```bash
conda activate sam3d-objects
python /home/lleba/Thesis/Development/SAM3D/run_sam3d_objects.py
```

### Key functions

#### `build_image_path_index(scene_idx) → dict`
Walks the nuScenes CAM_FRONT linked list **once** and returns `{frame_idx: image_path}`.
Called once at startup to avoid re-traversing on every frame.

#### `bbox_from_gaussian_splat(gs) → dict`
Reads the `.means` tensor (N, 3) directly from the in-memory `GaussianSplat` object.
Returns X/Y/Z min, max, and extent — no `.ply` write needed.

```python
dims = bbox_from_gaussian_splat(output["gs"])
dims["width_x"]   # X extent (scene units, not metres yet)
dims["height_y"]  # Y extent
dims["depth_z"]   # Z extent
dims["center"]    # [x, y, z] centroid
dims["n_gaussians"]
```

#### `main()`
Full pipeline in one function:
1. Load `.npy` masks, filter by `MIN_PROB` and `START_FRAME`
2. Call `build_image_path_index()` to get image paths
3. Load the `Inference` model from `sam-3d-objects/notebook/inference.py`
4. For each frame → for each detected object:
   - Load RGB image from disk
   - Call `inference(rgb, binary_mask, seed=42)` — returns `{"gs": GaussianSplat, ...}`
   - Call `bbox_from_gaussian_splat()` and print results
   - Optionally save `.ply`
5. Print summary table

### Output

Terminal — one block per object:
```
Frame 00547  |  3 object(s) detected
  obj 0:
    SAM3 prob     : 0.860
    mask pixels   : 24161
    image bbox    : cx=0.653 cy=0.397 w=0.105 h=0.151 (normalised)
    Running SAM-3D-Objects...
    Gaussians     : 348032
    3D bbox (scene units, not metric):
      X : [-0.302, +0.304]  width  = 0.606
      Y : [-0.497, +0.497]  height = 0.994
      Z : [-0.241, +0.242]  depth  = 0.483
```

Optional `.ply` files at `OUTPUT_DIR/frame{NNNNN}_obj{NN}.ply` — open in MeshLab, CloudCompare, or Blender.

---

## Note on dimensions

The X/Y/Z values are in SAM-3D-Objects' **internal normalised scene units**, not real-world metres. The model reconstructs geometry from a single view with no absolute scale reference. To convert to metres, you need to anchor to LiDAR — that's the planned next step (LiDAR fusion).

As a rough sanity check: for a typical car the model should produce a consistent aspect ratio (e.g. width > depth > height... or height > depth depending on axis convention) even if the absolute scale is off.

---

## Data locations

| What | Path |
|---|---|
| nuScenes dataset | `/media/lleba/ECP_Nuscenes_01/output2/ecp2nuscenes` |
| SAM3 mask outputs | `/media/lleba/ECP_Nuscenes_01/SAM3_Visualizations/Strassbourg/` |
| SAM3 visualizations | `…/scene_10_cam_0_car/frame_NNNNN.jpg` |
| SAM-3D-Objects .ply output | `/media/lleba/ECP_Nuscenes_01/SAM3D_Outputs/Strassbourg/` |
| SAM-3D-Objects checkpoints | `SAM3D/sam-3d-objects/checkpoints/hf/` |
| SAM3.1 checkpoint (cached) | `~/.cache/huggingface/hub/models--facebook--sam3.1/…/sam3.1_multiplex.pt` |
