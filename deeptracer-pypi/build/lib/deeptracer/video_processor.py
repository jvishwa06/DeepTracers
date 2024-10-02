# deeptracer/video_processor.py

import cv2
import random
import numpy as np
from .model import DeepFakeDetector

class VideoProcessor:
    def __init__(self):
        self.detector = DeepFakeDetector()

    def frames_from_video_file(self, video_path, n_frames=3, output_size=(224, 224), frame_step=15):
        result = []
        src = cv2.VideoCapture(video_path)
        video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))

        need_length = 1 + (n_frames - 1) * frame_step
        max_start = max(0, video_length - need_length)
        start = random.randint(0, max_start)

        src.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(n_frames):
            for _ in range(frame_step):
                ret, frame = src.read()
            if ret:
                frame = cv2.resize(frame, output_size)
                result.append(frame)
            else:
                result.append(np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8))
        
        src.release()
        return result

    def predict_video(self, video_path):
        frames = self.frames_from_video_file(video_path)
        predictions = [self.detector.predict_image(frame) for frame in frames]
        
        # Aggregate predictions
        final_prediction = max(predictions, key=lambda x: x['confidence'])
        return final_prediction
