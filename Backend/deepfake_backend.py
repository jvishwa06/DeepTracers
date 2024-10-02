# import random
# import cv2
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import pickle
# import librosa
# import torch
# import torch.nn.functional as F
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# from typing import List, Tuple
# from collections import Counter
# import logging

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# device = torch.device('cpu')

# # Define the device
# DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# # Load MTCNN and InceptionResnetV1
# mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()
# model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=1, device=DEVICE)
# checkpoint = torch.load('F:\\DeepfakeDetective\\Backend\\Models\\resnetinceptionv1_epoch_32.pth', map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint['model_state_dict'])
# model.to(DEVICE).eval()

# def format_frames(frame, output_size):
#     frame = cv2.resize(frame, output_size)
#     return frame

# def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
#     result = []
#     src = cv2.VideoCapture(str(video_path))
#     video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))  # Ensure this is an integer
#     need_length = 1 + (n_frames - 1) * frame_step

#     if need_length > video_length:
#         start = 0
#     else:
#         max_start = int(video_length - need_length)  # Ensure this is an integer
#         start = random.randint(0, max_start + 1)

#     src.set(cv2.CAP_PROP_POS_FRAMES, start)
#     ret, frame = src.read()
#     result.append(format_frames(frame, output_size))

#     for _ in range(n_frames - 1):
#         for _ in range(frame_step):
#             ret, frame = src.read()
#         if ret:
#             frame = format_frames(frame, output_size)
#             result.append(frame)
#         else:
#             result.append(np.zeros_like(result[0]))
#     src.release()
#     result = np.array(result)
#     return result


# def pred(image_path: str):
#     img = Image.open(image_path).convert('RGB')
#     face = mtcnn(img)
#     if face is None:
#         return "error"  # No face detected
#     face = face.unsqueeze(0).to(DEVICE).float() / 255.0

#     with torch.no_grad():
#         output = torch.sigmoid(model(face).squeeze(0))
#         prediction = "real" if output.item() < 0.5 else "fake"
    
#     return prediction

# def predictFake(path):
#     m, _ = librosa.load(path, sr=16000)
#     max_length = 500
#     mfccs = librosa.feature.mfcc(y=m, sr=16000, n_mfcc=40)

#     if mfccs.shape[1] < max_length:
#         mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
#     else:
#         mfccs = mfccs[:, :max_length]
    
#     model = load_model(r'C:\Users\iQube_VR\DeepfakeDetection\Backend\Models\audio_classifier.h5')
#     output = model.predict(mfccs.reshape(-1, 40, 500))
#     return "fake" if output[0][0] > 0.5 else "real"

# def save_images(path):
#     paths = []
#     for i in range(3):
#         image_3d = frames_from_video_file(path, 3)[i]
#         if image_3d.shape[2] == 4:
#             image_3d = image_3d[:, :, :3]
#         plt.figure(figsize=(1, 1))
#         plt.imshow(image_3d)
#         plt.axis('off')
#         save_path = f"uploads/image{i}.jpg"
#         paths.append(save_path)
#         plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#         plt.close()
#     return paths

# app = Flask(__name__)
# CORS(app, resources={r"/upload": {"origins": "http://localhost:5173"}})

# UPLOAD_FOLDER = 'uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# def find_mode(arr):
#     counts = Counter(arr)
#     max_count = max(counts.values())
#     mode = next(key for key, value in counts.items() if value == max_count)
#     return mode

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'image' in request.files:
#         file = request.files['image']
#         filename = file.filename
#         if filename == '':
#             return jsonify({'error': 'No selected file'})
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         try:
#             prediction = pred(file_path)
#             logging.info(f'Prediction: {prediction}')
#             os.remove(file_path)
#             return jsonify({'prediction': prediction})
#         except Exception as e:
#             logging.error(f'Error in image prediction: {str(e)}')
#             os.remove(file_path)
#             return jsonify({'error': str(e)})

#     if 'audio' in request.files:
#         file = request.files['audio']
#         filename = file.filename
#         if filename == '':
#             return jsonify({'error': 'No selected file'})
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
#         try:
#             prediction = predictFake(file_path)
#             logging.info(f'Audio Prediction: {prediction}')
#             os.remove(file_path)
#             return jsonify({'prediction': prediction})
#         except Exception as e:
#             logging.error(f'Error in audio prediction: {str(e)}')
#             os.remove(file_path)
#             return jsonify({'error': str(e)})

#     if 'video' in request.files:
#         file = request.files['video']
#         filename = file.filename
#         if filename == '':
#             return jsonify({'error': 'No selected file'})
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         try:
#             paths = save_images(file_path)
#             predictions = []
#             for i in paths:
#                 try:
#                     prediction = pred(i)
#                     predictions.append(prediction)
#                     logging.info(f'Frame Prediction: {prediction} for image {i}')
#                 except Exception as e:
#                     predictions.append("error")
#                     logging.error(f'Error in frame prediction for image {i}: {str(e)}')

#             final_prediction = find_mode(predictions)
#             logging.info(f'Mode Prediction: {final_prediction}')
#             os.remove(file_path)
#             return jsonify({'prediction': final_prediction})
#         except Exception as e:
#             logging.error(f'Error in video processing: {str(e)}')
#             os.remove(file_path)
#             return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)




import random
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import librosa
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from typing import List, Tuple
from collections import Counter
import logging
import subprocess
import onnx
from onnx2pytorch import ConvertModel
from models.TMC import ETMC
from models import image

# Configure logging
logging.basicConfig(level=logging.DEBUG)

device = torch.device('cpu')

# Define the device
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load MTCNN and InceptionResnetV1
mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=1, device=DEVICE)
checkpoint = torch.load('F:\\DeepfakeDetective\\Backend\\Models\\resnetinceptionv1_epoch_32.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# Load ONNX model for audio
onnx_model = onnx.load('D:\\Datathon-24\\audio_deepfake_detector\\checkpoints\\efficientnet.onnx')
pytorch_model = ConvertModel(onnx_model)

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the audio_args dictionary
audio_args = {
    'nb_samp': 64600,
    'first_conv': 1024,
    'in_channels': 1,
    'filts': [20, [20, 20], [20, 128], [128, 128]],
    'blocks': [2, 4],
    'nb_fc_node': 1024,
    'gru_node': 1024,
    'nb_gru_layer': 3,
    'nb_classes': 2
}

def preprocess_audio(audio_file):
    audio_pt = torch.unsqueeze(torch.Tensor(audio_file), dim=0)
    return audio_pt

def deepfakes_spec_predict(input_audio):
    try:
        x, _ = librosa.load(input_audio, sr=16000)
    except ValueError:
        norm_input_audio = str(input_audio[:-5] + '_norm' + input_audio[-5:])
        subprocess.call(['ffmpeg-normalize', input_audio, '-nt=rms', '-o=' + norm_input_audio, '-ar=16000', '-f', '-c:a=flac'])
        x, _ = librosa.load(norm_input_audio, sr=16000)
    audio = preprocess_audio(x)
    spec_grads = spec_model.forward(audio)
    spec_grads_inv = np.exp(spec_grads.cpu().detach().numpy().squeeze())

    max_value = np.argmax(spec_grads_inv)

    if max_value > 0.5:
        preds = round(100 - (max_value * 100), 3)
        text2 = f"The audio is REAL."
    else:
        preds = round(max_value * 100, 3)
        text2 = f"The audio is FAKE."

    return text2

# The rest of your existing functions remain unchanged...

def pred(image_path: str):
    img = Image.open(image_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        return "error"  # No face detected
    face = face.unsqueeze(0).to(DEVICE).float() / 255.0

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"
    
    return prediction

def save_images(path):
    paths = []
    for i in range(3):
        image_3d = frames_from_video_file(path, 3)[i]
        if image_3d.shape[2] == 4:
            image_3d = image_3d[:, :, :3]
        plt.figure(figsize=(1, 1))
        plt.imshow(image_3d)
        plt.axis('off')
        save_path = f"uploads/image{i}.jpg"
        paths.append(save_path)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    return paths

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "http://localhost:5173"}})

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def find_mode(arr):
    counts = Counter(arr)
    max_count = max(counts.values())
    mode = next(key for key, value in counts.items() if value == max_count)
    return mode

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' in request.files:
        file = request.files['image']
        filename = file.filename
        if filename == '':
            return jsonify({'error': 'No selected file'})
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            prediction = pred(file_path)
            logging.info(f'Prediction: {prediction}')
            os.remove(file_path)
            return jsonify({'prediction': prediction})
        except Exception as e:
            logging.error(f'Error in image prediction: {str(e)}')
            os.remove(file_path)
            return jsonify({'error': str(e)})

    if 'audio' in request.files:
        file = request.files['audio']
        filename = file.filename
        if filename == '':
            return jsonify({'error': 'No selected file'})
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            prediction = deepfakes_spec_predict(file_path)  # Using the new audio model
            logging.info(f'Audio Prediction: {prediction}')
            os.remove(file_path)
            return jsonify({'prediction': prediction})
        except Exception as e:
            logging.error(f'Error in audio prediction: {str(e)}')
            os.remove(file_path)
            return jsonify({'error': str(e)})

    if 'video' in request.files:
        file = request.files['video']
        filename = file.filename
        if filename == '':
            return jsonify({'error': 'No selected file'})
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
            os.remove(file_path)
            return jsonify({'prediction': final_prediction})
        except Exception as e:
            logging.error(f'Error in video processing: {str(e)}')
            os.remove(file_path)
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
