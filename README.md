

# CVAT YOLOv8 & YOLOv12 Segmentation Auto-Annotation

Deploy YOLOv8 or YOLOv12 segmentation models as serverless Nuclio functions for automatic annotation in [CVAT](https://www.cvat.ai/).

## Table of Contents

- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Model Configuration](#model-configuration)
- [How It Works](#how-it-works)
- [YOLOv8 vs YOLOv12](#yolov8-vs-yolov12)
- [Customization](#customization)
  - [Spec Labels](#spec-labels)
  - [Confidence Threshold](#confidence-threshold)
  - [Polygon Approximation Tolerance](#polygon-approximation-tolerance)
  - [Image Size & Retina Masks](#image-size--retina-masks)
- [Two-Step Annotation Workflow](#two-step-annotation-workflow)
- [Flash Attention (YOLOv12)](#flash-attention-yolov12)
- [Troubleshooting](#troubleshooting)

## Repository Structure

```
├── yolov8-seg/
│   ├── main.py              # Nuclio handler: inference, contour extraction, CVAT mask
│   └── function-gpu.yaml    # Nuclio function definition (YOLOv8)
├── yolov12-seg/
│   ├── main.py              # Nuclio handler: same pipeline with error handling
│   └── function-gpu.yaml    # Nuclio function definition (YOLOv12)
└── README.md
```

Each folder contains everything needed to deploy a Nuclio function: a `function-gpu.yaml` that defines the Docker build, runtime, and GPU resources, and a `main.py` that runs the inference and returns CVAT-compatible annotations.

## Prerequisites

- **CVAT** installed via Docker Compose (provides the Nuclio dashboard)
- **GPU** with NVIDIA drivers and `nvidia-container-toolkit` (CPU-only also works, see [Deployment](#quick-start))
- **Docker**

Refer to the [CVAT serverless documentation](https://docs.cvat.ai/docs/administration/advanced/installation_automatic_annotation/) for initial setup of the Nuclio platform.

## Quick Start

1. Clone this repository alongside your CVAT installation.

2. Copy or symlink the model folder into your CVAT serverless directory:

   ```bash
   # For YOLOv8
   cp -r yolov8-seg ~/cvat/serverless/pytorch/ultralytics/

   # For YOLOv12
   cp -r yolov12-seg ~/cvat/serverless/pytorch/ultralytics/
   ```

3. **(YOLOv12 only)** Adjust the flash attention wheel URL in `function-gpu.yaml` to match your CUDA and Python version (see [Flash Attention](#flash-attention-yolov12)).

4. Deploy the function from the CVAT directory:

   ```bash
   # YOLOv8
   ./serverless/deploy_gpu.sh serverless/pytorch/ultralytics/yolov8-seg

   # YOLOv12
   ./serverless/deploy_gpu.sh serverless/pytorch/ultralytics/yolov12-seg
   ```

   For CPU-only, use `deploy_cpu.sh` instead.

5. In the CVAT UI, the model will appear under **Automatic Annotation** in the task view.

## Model Configuration

By default, both handlers try to load a custom model file from the working directory, falling back to a pretrained model from Ultralytics:

| Variant  | Custom path                      | Fallback              |
|----------|----------------------------------|-----------------------|
| YOLOv8   | `your-custom-yolov8-model.pt`    | `yolov8s-seg.pt`      |
| YOLOv12  | `your-custom-yolov12-seg-model.pt` | `yolov12s-seg.pt`   |

To use a **custom model**, place your `.pt` file in the respective folder and update the `model_path` variable in `main.py` to match the filename. The function will load it on startup.

## How It Works

```
[CVAT] → HTTP POST (base64 image, threshold)
         │
         ▼
    main.py:handler()
         │
         ├─ Decode base64 image → PIL Image
         ├─ Run YOLO inference (imgsz=640, retina_masks=True)
         ├─ Filter detections by confidence threshold
         │
         For each detection:
         ├─ Extract bounding box (xyxy)
         ├─ Extract binary mask → convert to CVAT mask format
         ├─ Find contours with skimage → approximate polygon (tolerance=1)
         └─ Return {"confidence", "label", "type": "mask", "points", "mask"}
         │
         ▼
[CVAT] ← JSON response with polygon points and mask data
```

The handler returns both polygon points (for the CVAT shape) and a flat mask array for precise segmentation overlay within the bounding box region.

## YOLOv8 vs YOLOv12

| Feature              | YOLOv8                                      | YOLOv12                                                   |
|----------------------|---------------------------------------------|-----------------------------------------------------------|
| Base image           | `ultralytics/ultralytics:latest`            | `nvcr.io/nvidia/pytorch:25.05-py3`                        |
| Python runtime       | 3.9                                         | 3.11                                                      |
| Dependencies         | supervision, scikit-image (via ultralytics) | torch, supervision, scikit-image, huggingface_hub, msgpack, nuclio-sdk |
| Flash attention      | Not required                                | Required (wheel install)                                  |
| Ultralytics source   | Standard `pip install ultralytics`          | Custom fork: `sunsmarterjie/yolov12` seg branch           |
| Namespace            | `pth-ultralytics-yolov8-segment`            | `yolov12-segment`                                         |
| Error handling       | None                                        | Try/except around input parsing and inference             |

## Customization

### Spec Labels

Replace the placeholder labels in `function-gpu.yaml` with your model's actual class names:

```yaml
annotations:
  spec: |
    [
      {"id": 0, "name": "person"},
      {"id": 1, "name": "car"},
      {"id": 2, "name": "bicycle"}
    ]
```

These labels appear in the CVAT UI when selecting the model and are used to map class IDs to human-readable names.

### Confidence Threshold

The default threshold is `0.5`. You can pass a different value from CVAT's automatic annotation dialog. Detections below the threshold are filtered out.

### Polygon Approximation Tolerance

Contours extracted from segmentation masks are simplified using `skimage.measure.approximate_polygon` with a tolerance of `1`. Lower values produce more detailed (higher vertex count) polygons; higher values produce coarser approximations. Adjust the `tolerance` parameter in `main.py` if needed.

### Image Size & Retina Masks

Both handlers run inference at `imgsz=640` with `retina_masks=True`. Retina masks produce higher-resolution segmentation masks — useful when fine-grained boundaries matter.

## Two-Step Annotation Workflow

YOLO segmentations alone can lack precision on difficult boundaries. A recommended two-step approach:

1. **Bulk annotate** all images with YOLOv8/v12 — fast, decent quality across the dataset.
2. **Refine** with SAM (Segment Anything Model) via CVAT's SAM integration to correct or tighten individual labels.

This yields highly accurate annotations with minimal manual effort.

## Flash Attention (YOLOv12)

YOLOv12 requires flash attention for efficient attention computation. The wheel URL in `function-gpu.yaml` must match your environment:

```yaml
- kind: RUN
  value: pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.3cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Components to match:
- **CUDA version** (`cu12` → CUDA 12.x)
- **PyTorch version** (`torch2.3` → PyTorch 2.3)
- **Python version** (`cp312` → CPython 3.12)
- **ABI** (`cxx11abiTRUE` vs `cxx11abiFALSE`)

Check the [Flash Attention releases page](https://github.com/Dao-AILab/flash-attention/releases) for the correct wheel for your setup.

## Troubleshooting

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| Function fails to build | Flash attention wheel mismatch | Update the wheel URL in `function-gpu.yaml` |
| No annotations returned | Confidence threshold too high | Lower the threshold in CVAT UI |
| Missing labels in UI | Placeholder labels in spec | Replace `<XYZ>` with actual class names |
| GPU not detected | `nvidia-container-toolkit` not installed | `sudo apt install nvidia-container-toolkit` and restart Docker |
| YOLOv12 inference errors | Python version mismatch | Ensure `function-gpu.yaml` runtime matches the base image's Python |
| Empty contour for detection | Small or degenerate mask | Both handlers skip empty contours |
