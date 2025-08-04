# VisionFace

<div align="center">

  ![VisionFace](https://github.com/user-attachments/assets/52ac9123-304c-4098-a1e5-f413d03bfec9)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/visionface.svg)](https://badge.fury.io/py/visionface)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**Modern face detection, recognition & analysis in 3 lines of code**

VisionFace is a state-of-the-art, open-source framework for comprehensive face analysis, built with PyTorch. It provides a unified interface for face detection, recognition, landmark detection, and visualization with support for multiple cutting-edge models.

[Quick Start](#-quick-start) • [Examples](#-examples) • [Models](#-models) • [API Docs](https://visionface.readthedocs.io)

</div>

## ✨ What VisionFace Does

<div align="center">
<table>
  <tr>
    <td>
      <a href="#face-detection">
        <figure>
          <img src="banners/face_detection.jpg" alt="Face Detection" width="500"/>
          <figcaption style="text-align: center;">Face Detection</figcaption>
        </figure>
      </a>
    </td>
    <td>
      <a href="#face-recognition">
        <figure>
          <img src="banners/face_recognition.jpg" alt="Face Recognition" width="500"/>
          <figcaption style="text-align: center;">Face Recognition</figcaption>
        </figure>
      </a>
    </td>
    <td>
      <a href="#face-landmarks">
        <figure>
          <img src="banners/face_landmarks.jpg" alt="Face Landmarks" width="500"/>
          <figcaption style="text-align: center;">Face Landmarks</figcaption>
        </figure>
      </a>
    </td>
  </tr>
  <tr>
    <td>
      <a href="#-examples">
        <figure>
          <img src="banners/face_analysis.jpg" alt="Face Analysis" width="500"/>
          <figcaption style="text-align: center;">Face Analysis</figcaption>
        </figure>
      </a>
    </td>
    <td>
      <a href="#face-embeddings-1-minute">
        <figure>
          <img src="banners/face_verification.jpg" alt="Face Verification" width="500"/>
          <figcaption style="text-align: center;">Face Verification</figcaption>
        </figure>
      </a>
    </td>
    <td>
      <a href="#-examples">
        <figure>
          <img src="banners/face_visualization.jpg" alt="Face Visualization" width="500"/>
          <figcaption style="text-align: center;">Face Visualization</figcaption>
        </figure>
      </a>
    </td>
  </tr>
</table>
</div>


- **Detect faces** in images with 12+ models (YOLO, MediaPipe, MTCNN...)
- **Recognize faces** with vector search and embedding models
- **Extract landmarks** (68-point, 468-point face mesh)
- **Batch process** thousands of images efficiently
- **Production-ready** with Docker support and REST API

## 🚀 Quick Start

```bash
pip install visionface
```

### Face Detection 

```python
import cv2
from visionface import FaceDetection, FaceAnnotators

# 1. Initialize detector
detector = FaceDetection(detector_backbone="yolo-small")

# 2. Detect faces
image = cv2.imread("your_image.jpg")
faces = detector.detect_faces(image)

# 3. Visualize results
result = FaceAnnotators.box_annotator(image, faces)
cv2.imwrite("detected.jpg", result)
```

### Face Recognition

```python
from visionface import FaceRecognition

# 1. Setup recognition system
fr = FaceRecognition(detector_backbone="yolo-small", 
                     embedding_backbone="FaceNet-VGG")

# 2. Add known faces
fr.upsert_faces(
    images=["john.jpg", "jane.jpg", "bob.jpg"],
    labels=["John", "Jane", "Bob"],
    collection_name="employees"
)

# 3. Search for matches
matches = fr.search_faces("security_camera.jpg", 
                         collection_name="employees",
                         score_threshold=0.7)

for match in matches[0]:
    print(f"Found: {match['face_name']} (confidence: {match['score']:.2f})")
```

### Face Embeddings 

```python
from visionface import FaceEmbedder

# 1. Initialize embedder
embedder = FaceEmbedder(embedding_backbone="FaceNet-VGG")

# 2. Generate embeddings for face images
embeddings = embedder.embed_faces(
    face_imgs=["face1.jpg", "face2.jpg"],
    normalize_embeddings=True  # L2 normalization
)

# 3. Use embeddings
for i, embedding in enumerate(embeddings):
    print(f"Face {i+1} embedding shape: {embedding.shape}")  # (512,)
    # Use for: face verification, clustering, custom databases
```

### Face Landmarks
```python
from visionface import LandmarkDetection, FaceAnnotators

landmark_detector = LandmarkDetection(detector_backbone="mediapipe")
image = cv2.imread("your_image.jpg")

# Get 468 facial landmarks
landmarks = landmark_detector.detect_landmarks(image)

# Visualize with connections
result = FaceAnnotators.landmark_annotator(
    image, landmarks[0], connections=True
)
cv2.imwrite("landmarks.jpg", result)
```

## 💡 Examples

<details>
<summary><b>🎯 Real-time Face Detection</b></summary>

```python
import cv2
from visionface import FaceDetection, FaceAnnotators

detector = FaceDetection(detector_backbone="yolo-nano")  # Fastest model
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = detector.detect_faces(frame)
    annotated = FaceAnnotators.box_annotator(frame, faces)
    
    cv2.imshow('Face Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
</details>

<details>
<summary><b>📊 Batch Processing</b></summary>

```python
from visionface import FaceDetection
import glob

detector = FaceDetection(detector_backbone="yolo-medium")

# Process entire folder
image_paths = glob.glob("photos/*.jpg")
images = [cv2.imread(path) for path in image_paths]

# Detect all faces at once
all_detections = detector.detect_faces(images)

# Save cropped faces
for i, detections in enumerate(all_detections):
    for j, face in enumerate(detections):
        if face.cropped_face is not None:
            cv2.imwrite(f"faces/image_{i}_face_{j}.jpg", face.cropped_face)
```
</details>

<details>
<summary><b>🏢 Employee Recognition System</b></summary>

```python
from visionface import FaceRecognition
import os

# Initialize system
fr = FaceRecognition(db_backend="qdrant")

# Auto-enroll from employee photos folder
def enroll_employees(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png')):
            name = filename.split('.')[0]  # Use filename as name
            image_path = os.path.join(folder_path, filename)
            
            fr.upsert_faces(
                images=[image_path],
                labels=[name],
                collection_name="company_employees"
            )
            print(f"Enrolled: {name}")

# Enroll all employees
enroll_employees("employee_photos/")

# Check security camera feed
def identify_person(camera_image):
    results = fr.search_faces(
        camera_image,
        collection_name="company_employees",
        score_threshold=0.8,
        top_k=1
    )
    
    if results[0]:  # If match found
        return results[0][0]['face_name']
    return "Unknown person"
```
</details>

## 🎯 Models

**Choose the right model for your use case:**

| Use Case | Speed | Accuracy | Recommended Model |
|----------|-------|----------|------------------|
| 🚀 **Real-time apps** | ⚡⚡⚡ | ⭐⭐ | `yolo-nano`, `mediapipe` |
| 🎯 **General purpose** | ⚡⚡ | ⭐⭐⭐ | `yolo-small` (default) |
| 🔍 **High accuracy** | ⚡ | ⭐⭐⭐⭐ | `yolo-large`, `mtcnn` |
| 📱 **Mobile/Edge** | ⚡⚡⚡ | ⭐⭐ | `mediapipe`, `yolo-nano` |
| 🎭 **Landmarks needed** | ⚡⚡ | ⭐⭐⭐ | `mediapipe`, `dlib` |

<details>
<summary><b>📋 Complete Model List</b></summary>

**Detection Models:**
- `yolo-nano`, `yolo-small`, `yolo-medium`, `yolo-large`
- `yoloe-small`, `yoloe-medium`, `yoloe-large` (prompt-based)  
- `yolow-small`, `yolow-medium`, `yolow-large`, `yolow-xlarge` (open-vocabulary)
- `mediapipe`, `mtcnn`, `opencv`

**Embedding Models:**
- `FaceNet-VGG` (512D) - Balanced accuracy/speed
- `FaceNet-CASIA` (512D) - High precision
- `Dlib` (128D) - Lightweight

**Landmark Models:**
- `mediapipe` - 468 points + 3D mesh
- `dlib` - 68 points, robust
</details>


## 📚 Documentation

- 📖 [Full Documentation](https://visionface.readthedocs.io)
- 🎓 [Tutorials & Guides](https://visionface.readthedocs.io/tutorials)
- 🔌 [REST API Reference](https://visionface.readthedocs.io/api)
- 💡 [Use Case Examples](https://github.com/username/visionface/tree/main/examples)

## 🤝 Contributing
<a href="https://github.com/miladfa7/visionface/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=miladfa7/visionface" />
</a>

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md).

**Quick ways to help:**
- ⭐ Star the repo
- 🐛 Report bugs
- 💡 Request features  
- 📝 Improve docs
- 🔧 Submit PRs

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Citation

```bibtex
@software{VisionFace2025,
  title = {VisionFace: Modern Face Detection & Recognition Framework},
  author = {VisionFace Team},
  year = {2025},
  url = {https://github.com/username/visionface}
}
```

---

<div align="center">

**[⬆ Back to Top](#visionface)** • **Made with ❤️ by the VisionFace team**

</div>
