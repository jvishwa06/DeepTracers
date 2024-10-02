import os
import random
import logging
import numpy as np
import librosa
import torch
import cv2
import matplotlib
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from collections import Counter
from facenet_pytorch import MTCNN, InceptionResnetV1
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Use a non-interactive backend for plotting

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "http://localhost:5173"}})

# Configure device
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load models
mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=1, device=DEVICE)
checkpoint = torch.load('D:\\DeepTracers\\Backend\\resnetinceptionv1_epoch_32.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# Set up upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def format_frames(frame, output_size):
    return cv2.resize(frame, output_size)

def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
    result = []
    src = cv2.VideoCapture(video_path)
    video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    need_length = 1 + (n_frames - 1) * frame_step

    start = 0 if need_length > video_length else random.randint(0, video_length - need_length)
    src.set(cv2.CAP_PROP_POS_FRAMES, start)

    ret, frame = src.read()
    if not ret:
        return np.zeros((n_frames, *output_size, 3), dtype=np.uint8)

    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        result.append(format_frames(frame, output_size) if ret else np.zeros_like(result[0]))

    src.release()
    return np.array(result)

def pred(image_path: str):
    img = Image.open(image_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        return "error"  # No face detected

    face = face.unsqueeze(0).to(DEVICE).float() / 255.0
    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        return "real" if output.item() < 0.5 else "fake"

def predictFake(path):
    m, _ = librosa.load(path, sr=16000)
    mfccs = librosa.feature.mfcc(y=m, sr=16000, n_mfcc=40)
    max_length = 500

    if mfccs.shape[1] < max_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]

    model = load_model(r'C:\Users\iQube_VR\DeepfakeDetection\Backend\Models\audio_classifier.h5')
    output = model.predict(mfccs.reshape(-1, 40, 500))
    return "fake" if output[0][0] > 0.5 else "real"

def save_images(path):
    paths = []
    for i in range(3):
        image_3d = frames_from_video_file(path, 3)[i]
        if image_3d.shape[2] == 4:
            image_3d = image_3d[:, :, :3]
        plt.figure(figsize=(1, 1))
        plt.imshow(image_3d)
        plt.axis('off')
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"image{i}.jpg")
        paths.append(save_path)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    return paths

def find_mode(arr):
    counts = Counter(arr)
    max_count = max(counts.values())
    return next(key for key, value in counts.items() if value == max_count)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' in request.files:
        file = request.files['image']
        if not file.filename:
            return jsonify({'error': 'No selected file'})
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            prediction = pred(file_path)
            logging.info(f'Prediction: {prediction}')
            return jsonify({'prediction': prediction})
        except Exception as e:
            logging.error(f'Error in image prediction: {str(e)}')
            return jsonify({'error': str(e)})
        finally:
            os.remove(file_path)

    if 'audio' in request.files:
        file = request.files['audio']
        if not file.filename:
            return jsonify({'error': 'No selected file'})
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            prediction = predictFake(file_path)
            logging.info(f'Audio Prediction: {prediction}')
            return jsonify({'prediction': prediction})
        except Exception as e:
            logging.error(f'Error in audio prediction: {str(e)}')
            return jsonify({'error': str(e)})
        finally:
            os.remove(file_path)

    if 'video' in request.files:
        file = request.files['video']
        if not file.filename:
            return jsonify({'error': 'No selected file'})
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            paths = save_images(file_path)
            predictions = []

            for i in paths:
                try:
                    prediction = pred(i)
                    predictions.append(prediction)
                    logging.info(f'Frame Prediction: {prediction} for image {i}')
                except Exception as e:
                    predictions.append("error")
                    logging.error(f'Error in frame prediction for image {i}: {str(e)}')

            final_prediction = find_mode(predictions)
            logging.info(f'Mode Prediction: {final_prediction}')
            return jsonify({'prediction': final_prediction})
        except Exception as e:
            logging.error(f'Error in video processing: {str(e)}')
            return jsonify({'error': str(e)})
        finally:
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
