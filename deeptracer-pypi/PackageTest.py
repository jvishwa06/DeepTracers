from deeptracer import DeepFakeDetector

detector = DeepFakeDetector()

#for image prediction
image = detector.predict_image('D:\\DeepTracers\\DeepfakeTestingData\\fake4.png')
print(image)

#for video prediction
video = detector.predict_video('D:\\DeepTracers\\DeepfakeTestingData\\real1.mp4')
print(video)
