
```markdown
<div align="center">

# 🔍 VisionLens
### Industry Standard vs. Research Frontier in Object Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Industry_Standard-00FF00.svg)](https://github.com/ultralytics/ultralytics)
[![Grounding DINO](https://img.shields.io/badge/Grounding_DINO-Research_Frontier-FFA500.svg)](https://arxiv.org/abs/2303.05499)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-blue)](https://huggingface.co/spaces/gujarkaleem07/visionlens)

**[Click here to try the live demo with webcam support](https://huggingface.co/spaces/gujarkaleem07/visionlens)**

</div>

---

### 🧠 Why This Project?
Traditional detectors (like YOLO) are locked to 80 fixed classes. If you want to detect a rare object, you must retrain the model. **VisionLens** compares the fast, production-ready **YOLOv8** against the cutting-edge **Grounding DINO**, which uses Vision-Language fusion to detect *any object* via text prompts (Zero-Shot Detection).

### ⚖️ The Comparison

| Feature | ⚡ YOLOv8 (Tab 1) | 💬 Grounding DINO (Tab 2) |
| :--- | :--- | :--- |
| **Paradigm** | Closed-Vocabulary | Open-Vocabulary |
| **Input** | Image Only | Image + Text Prompt |
| **Classes** | 80 Fixed (COCO) | Unlimited (Dynamic) |
| **Speed** | Very Fast (~100ms) | Slower (~2-5s) |
| **Best For** | Production, Real-time apps | Research, Novel objects |

### 🛠️ Tech Stack
- **Backend:** PyTorch, Ultralytics, Hugging Face Transformers
- **Frontend:** Gradio 4.0 (with native webcam streaming)
- **Models:** YOLOv8-Nano, Grounding DINO-Tiny
- **Optimization:** Dynamic image resizing to prevent CPU OOM errors on free-tier hosting.

### 🔮 Future Work (Research Directions)
- Integrate **SAM** (Segment Anything) for pixel-perfect instance segmentation.
- Implement **Few-Shot Fine-Tuning** pipeline for niche domains (e.g., medical imaging).
- Quantitative benchmarking (mAP scores) across custom datasets.

