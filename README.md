# DeepTracersV0 - Advanced Deepfake Detection System with Social Media Integration

## Overview

**DeepTracersV0** is a social media platform prototype built with **React.js** where users can upload posts (images, audio, and video). The platform analyzes uploaded content for deepfakes using a sophisticated model. If a deepfake is detected, the platform prevents the post from being uploaded, flags it, and sends a report to cybersecurity professionals. Real posts are watermarked with a pixel-based identifier for traceability, providing verifiable evidence.

## Key Features

1. **Deepfake Detection**: 
   - Upon uploading, the platform runs a deepfake detection algorithm using a combination of **ResNet**, **Inception**, and **Vision Transformer** architectures trained on industry-leading datasets.
   - If a deepfake is detected, the user is notified, and the post is not uploaded.
   - A report is generated and sent to cybersecurity professionals for investigation.

2. **Pixel Watermarking**:
   - For genuine posts, a pixel watermark is applied to provide verifiable evidence for mitigating future deepfake concerns.

3. **Cybersecurity Dashboard**:
   - The platform provides an interface for professionals to upload and investigate suspicious content.
   - Offers **Explainable AI (XAI)** to display the key features that led to identifying content as deepfake.
   - A comprehensive dashboard displays the number of blocked deepfakes, mediums of origin (Instagram, Facebook, Reddit, X), and the number of reported and resolved deepfake cases.

4. **Reverse Image Search**:
   - Users can perform reverse image searches to find the origin of an uploaded image and generate reports for platforms to remove fake content.

## Datasets Used for Model Training

The model is trained using the following datasets:
- **The World Leader’s Dataset (WLDR)**
- **The FaceForensics++ dataset (FF)**
- **The DeepFake Detection dataset (DFD)**
- **The Deep Fake Detection Challenge Preview dataset (DFDC-P)**
- **The Celeb-DF dataset**

## Technology Stack

- **Frontend**: HTML, Tailwind CSS, React + Vite, Streamlit
- **Backend**: Flask
- **Database**: MongoDB, MySQL
- **Libraries & Frameworks**: PyTorch, TensorFlow, OpenCV, FaceNet_PyTorch, NumPy, Matplotlib, GradCAM

## Model Details

- **Federated Learning**: Due to computational complexities, we trained a separate model for each dataset and combined them using federated learning to form a unified final model.
- **Accuracy**: Achieved an accuracy of **97.64%**.
- **Inference Speed**: Implemented **Post-Training Quantization** for faster inference and reduced memory usage. Average inference time is between **120ms-150ms** with GPU utilization.

## Key Use Cases

1. **Content Moderation**:
   - Platforms like Facebook and TikTok can integrate this system to identify and flag harmful deepfake content before it spreads.
   
2. **Cybersecurity**:
   - Law enforcement and cybersecurity experts can utilize this system to investigate malicious or manipulated media, helping prevent fraud and deception.

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/jvishwa06/DeepTracers.git
   cd DeepTracers
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Start flask server:
   ```bash
   python Backend/Models/Deepfake_backend.py
   ```


## License

This project is licensed under the MIT License.

---

Feel free to adjust any sections to fit your exact project setup!
