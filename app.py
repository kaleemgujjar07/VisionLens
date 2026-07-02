import gradio as gr
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

print("!!!!!!!!!!! RUNNING NEW YOLOv8 CODE !!!!!!!!!!!")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading YOLOv8...")
yolo_model = YOLO("yolov8n.pt")
print("YOLOv8 Loaded!")

print("Loading Grounding DINO...")
gdino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
gdino_model.to(DEVICE)
print("Grounding DINO Loaded!")


def format_results(detections):
    if not detections:
        return "❌ No objects detected."
    class_counts = {}
    for d in detections:
        class_counts[d["label"]] = class_counts.get(d["label"], 0) + 1
    
    md = f"### ✅ Detected {len(detections)} object(s)\n\n| Class | Count | Avg Confidence |\n|-------|-------|----------------|\n"
    for label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        scores = [d["score"] for d in detections if d["label"] == label]
        md += f"| **{label}** | {count} | {sum(scores)/len(scores)*100:.1f}% |\n"
    md += f"\n*Device: {DEVICE}*"
    return md


def detect_yolo(image, threshold):
    if image is None:
        return None, "⚠️ Upload an image first."
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        results = yolo_model(image, conf=threshold, verbose=False)
        
        # YOLOv8 draws GREEN boxes by default
        annotated_image = Image.fromarray(results[0].plot())
        
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            detections.append({"label": yolo_model.names[cls_id], "score": float(box.conf[0])})
            
        return annotated_image, format_results(detections)
    except Exception as e:
        return image, f"Error: {str(e)[:100]}"


def detect_gdino(image, text_prompt, box_thresh, text_thresh):
    if image is None:
        return None, "⚠️ Upload an image first."
    if not text_prompt.strip():
        return image, "⚠️ Type what to detect."
    try:
        if max(image.size) > 800:
            ratio = 800 / max(image.size)
            image = image.resize((int(image.size[0]*ratio), int(image.size[1]*ratio)), Image.LANCZOS)
            
        inputs = gdino_processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = gdino_model(**inputs)
            
        results = gdino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_thresh, text_threshold=text_thresh, target_sizes=[image.size[::-1]]
        )[0]
        
        detections = [{"label": l, "score": float(s), "bbox": [int(c) for c in b.tolist()]} for b, s, l in zip(results["boxes"], results["scores"], results["labels"])]
        
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for i, d in enumerate(detections):
            draw.rectangle(d["bbox"], outline="purple", width=3)
            draw.text((d["bbox"][0], d["bbox"][1]-15), f"{d['label']}: {d['score']:.2f}", fill="purple", font=font)
            
        return image, format_results(detections)
    except Exception as e:
        return image, f"Error: {str(e)[:100]}"


with gr.Blocks(title="VisionLens") as demo:
    gr.HTML("<h1 style='text-align:center'>🔍 VisionLens</h1><p style='text-align:center;color:#666'>Industry Standard vs Research Frontier</p>")
    
    with gr.Tabs():
        with gr.Tab("⚡ YOLOv8 (Industry)"):
            with gr.Row():
                with gr.Column():
                    img1 = gr.Image(type="pil", sources=["upload", "webcam"], height=400)
                    thresh1 = gr.Slider(0.1, 0.9, value=0.4, label="Threshold")
                    btn1 = gr.Button("Detect", variant="primary")
                with gr.Column():
                    out1 = gr.Image(height=400)
                    txt1 = gr.Markdown()
            btn1.click(detect_yolo, [img1, thresh1], [out1, txt1])

        with gr.Tab("💬 Grounding DINO (Research)"):
            with gr.Row():
                with gr.Column():
                    img2 = gr.Image(type="pil", sources=["upload", "webcam"], height=400)
                    prompt = gr.Textbox(label="What to detect?", value="person, clothes")
                    with gr.Row():
                        bt = gr.Slider(0.05, 0.5, value=0.3, label="Box Thresh")
                        tt = gr.Slider(0.05, 0.5, value=0.25, label="Text Thresh")
                    btn2 = gr.Button("Detect", variant="primary")
                with gr.Column():
                    out2 = gr.Image(height=400)
                    txt2 = gr.Markdown()
            btn2.click(detect_gdino, [img2, prompt, bt, tt], [out2, txt2])

if __name__ == "__main__":
    demo.launch()