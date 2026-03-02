# 🦇 | GargoylEye | [Google Colab](https://colab.research.google.com/drive/1xDL6gSBJTmSdasms_cD03AahGRnNICmI?usp=sharing)

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![YOLOv8](https://img.shields.io/badge/YOLOv8x-Ultralytics-gold?style=flat-square)
![ControlNet](https://img.shields.io/badge/ControlNet-v1.1-darkred?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange?style=flat-square)

---

## Overview

**GargoylEye** is an end-to-end deep learning pipeline that automatically detects animals in any photograph, isolates them from humans and background noise, and reimagines them in entirely new environments through generative AI.

The system combines two state-of-the-art models: **YOLOv8x** (trained on the COCO dataset) handles real-time object detection, identifying animals while completely ignoring any humans in the frame. The detected animal is then cropped and passed to a **Stable Diffusion ControlNet** pipeline, which extracts the animal's structural edge map via Canny edge detection and uses it as a spatial guide during image generation. This ensures the animal's shape and pose are preserved while the surrounding scene is fully reimagined based on a user-defined text prompt.

You can also check out [GargoylEye's website](https://gargoyleye.lovable.app/) in order to experience it live.

---

## Features

- **Human-Suppressed Detection** — YOLOv8x runs at 1280px resolution with test-time augmentation, detecting all 10 COCO animal classes while silently filtering out humans
- **Advanced Edge Extraction** — CLAHE contrast normalization + bilateral filtering + Otsu auto-thresholding + Canny edge detection + morphological closing pipeline
- **Structure-Preserving Generation** — ControlNet v1.1 uses the extracted edge map as a shape guide so the animal's pose and body proportions survive the scene transformation
- **Photorealistic Output** — Realistic Vision V5.1 backbone with upgraded ft-mse VAE produces sharp, photo-quality results rather than illustrations
- **Scene Presets** — One-click prompt templates: Student, Astronaut, Cyberpunk, Ocean, Gotham
- **Smart Crop Quality Check** — Crops smaller than 80×80px are flagged before generation to prevent poor output
- **Auto-Save** — All generated renders saved automatically to `/content/output/`

---

## Models & Dataset

| Component | Source | Purpose |
|---|---|---|
| **YOLOv8x** | [Ultralytics](https://github.com/ultralytics/ultralytics) | Animal detection & localization |
| **ControlNet v1.1 Canny** | [lllyasviel/control_v11p_sd15_canny](https://huggingface.co/lllyasviel/control_v11p_sd15_canny) | Structure-guided image generation |
| **Realistic Vision V5.1** | [SG161222/Realistic_Vision_V5.1_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE) | Photorealistic generation backbone |
| **VAE ft-mse** | [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) | Sharp detail & accurate color decoding |
| **COCO Dataset** | [cocodataset.org](https://cocodataset.org) | YOLOv8x pretraining — 80 object classes |

### Detected Animal Classes (COCO IDs 14–23)
`bird` · `cat` · `dog` · `horse` · `sheep` · `cow` · `elephant` · `bear` · `zebra` · `giraffe`

---

## Pipeline Architecture
```
Upload Photo
     │
     ▼
YOLOv8x @ 1280px + Test-Time Augmentation
 ├── Detected: bird, cat, dog, horse... ✅
 └── Suppressed: person (class 0)       ❌
     │
     ▼
Adaptive Padding Crop (15% horizontal, 20% vertical)
     │
     ▼
Edge Extraction Pipeline
  CLAHE → Bilateral Filter → Otsu Threshold → Canny → Morphological Close
     │
     ▼
ControlNet v1.1 + Realistic Vision V5.1
  Edge map as shape guide + user text prompt
     │
     ▼
Generated Image @ 768×768  ✨
```

---

## Key Concepts

### 1. Object Detection
YOLO (You Only Look Once) processes the entire image in a single forward pass through the neural network, making it significantly faster than two-stage detectors. The `x` variant is the largest and most accurate in the YOLOv8 family. Running at `imgsz = 1280` doubles the default resolution, catching smaller or partially occluded animals. `augment = True` enables test-time augmentation — running detection on multiple flipped and scaled versions of the image and merging results to reduce missed detections.

### 2. Edge Extraction — Canny + Enhancements
Raw Canny edge detection struggles with uneven lighting and noisy fur textures. GargoylEye applies a four-stage preprocessing chain before Canny runs: CLAHE normalizes contrast across dark and bright regions; bilateral filtering removes noise while preserving structural edges (unlike Gaussian blur which softens everything); Otsu's method automatically finds the optimal threshold per image rather than using fixed values; morphological closing fills gaps in broken outlines so ControlNet sees complete body shapes.

### 3. Structure-Preserving Generation — ControlNet
Standard img2img Stable Diffusion degrades animal identity at high creativity strengths and fails to change the scene at low strengths. ControlNet solves this by extracting the animal's structural skeleton as a Canny edge map and conditioning the diffusion process on it. The model generates a completely new scene while the edge map acts as a rigid spatial constraint, preserving body pose and proportions throughout generation.

### 4. Prompt Engineering
Stable Diffusion's attention mechanism weighs earlier tokens more heavily. GargoylEye structures every prompt as: `animal name → user scene → photography style → technical quality terms`. A comprehensive negative prompt covering anatomy errors, unwanted people, style mismatches and quality issues is applied to every generation automatically.

---

## Results

| Metric | Value |
|---|---|
| Detection resolution | 1280px |
| Output resolution | 768×768 |
| Generation time (T4) | ~15–30s per image |
| Animal classes supported | 10 (COCO) |
| Human suppression | Automatic |
| Max images per run | 4 |

**Example prompt that work well:**
- `a noble cat portrait in the style of Renaissance oil painting, sitting in a regal pose, ornate gilded frame, dramatic chiaroscuro lighting, rich velvet robes, Italian Renaissance background with classical columns and drapes, masterpiece, museum quality`

---

## Requirements
```
ultralytics
diffusers
transformers
accelerate
xformers
safetensors
opencv-python
Pillow
ipywidgets
torch
```

---

## Limitations

- COCO's 10 animal classes do not include donkeys, foxes, wolves or exotic species — these get mapped to the nearest class (e.g. donkeys → `horse`)
- Very small animals in crowded scenes may be missed even at 1280px resolution
- ControlNet preserves shape but cannot recover detail from low-resolution crops smaller than 80×80px
- Generation quality depends heavily on prompt specificity — vague prompts produce generic results

---

## License

MIT License · Copyright (c) 2025 *1453nicat*
