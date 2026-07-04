<div align="center">

# 🔍 VisionLens
### Industry Standard vs. Research Frontier in Object Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FF00.svg)](https://github.com/ultralytics/ultralytics)
[![Grounding DINO](https://img.shields.io/badge/Grounding_DINO-Research-FFA500.svg)](https://arxiv.org/abs/2303.05499)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-blue)](https://huggingface.co/spaces/gujarkaleem07/visionlens)

**[Click here to try the live demo with webcam support](https://huggingface.co/spaces/gujarkaleem07/visionlens)**

</div>

---

### 🧠 The Core Concept
Traditional object detectors are locked to 80 fixed classes. If they haven't seen an object during training, they can't detect it. **VisionLens** breaks this barrier by letting you compare **YOLOv8** (fast, but fixed) against **Grounding DINO** (slower, but detects *anything* you type).

### ✨ Project Highlights
- **⚡ Blazing Fast YOLOv8:** Instantly detects people, cars, and animals with state-of-the-art accuracy.
- **💬 Zero-Shot Detection:** Type "coffee cup" or "rare bird" into Grounding DINO and it finds it without retraining.
- **📹 Live Webcam Support:** Real-time browser inference to test models on live video feeds.
- **🛡️ Crash-Proof Architecture:** Built-in dynamic image resizing to prevent CPU memory errors on free hosting.

### ⚖️ The Comparison
| Architecture | Speed | Flexibility |
| :--- | :--- | :--- |
| **YOLOv8** (Industry Standard) | ⚡ Lightning Fast | Locked to 80 classes |
| **Grounding DINO** (Research Frontier) | 🐢 Moderate | ✅ Unlimited (Text-prompted) |

### 🛠️ Tech Stack
- **Deep Learning:** PyTorch
- **Models:** YOLOv8-Nano, Grounding DINO-Tiny
- **Interface:** Gradio 4.0+



