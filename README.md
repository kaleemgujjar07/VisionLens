VisionLens: Object Detection Playground
VisionLens is a web application that detects objects in images and live webcam feeds. It lets users compare two different AI approaches: a fast industry-standard model (YOLOv8) and a flexible research model (Grounding DINO) that can find anything you type.

🚀 Click Here to Try the Live Demo

Features
Automatic Detection: Instantly finds common objects like people, cars, dogs, and chairs in any image.
Custom Text Search: Type what you are looking for (e.g., "coffee cup, bird, phone") and the AI will locate it, even if it wasn't specifically trained on that exact item.
Live Webcam Support: Turn on your camera directly in the browser to test the AI on real-time video.
Side-by-Side Comparison: Easily switch between the two AI models to see the difference in speed and flexibility.
How to Use
Go to the live demo link above.
Under the "Automatic" tab, upload any photo or click the webcam icon to use your camera.
Click the "Detect Objects" button to see instant results.
Switch to the "Open-Vocab" tab to try typing your own custom objects to search for.
Built With
Python
PyTorch
YOLOv8 (by Ultralytics)
Grounding DINO (by IDEA-Research)
Gradio (for the web interface)
