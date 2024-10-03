from deeptracer import DeepFakeDetector

detector = DeepFakeDetector()

# #for image prediction
# image = detector.predict_image('D:\\DeepTracers\\Datathon-Datasets\\IMAGE\\FAKE\\683.jpeg')
# print(image)

#for video prediction
video = detector.predict_video('D:\\DeepTracers\\Datathon-Datasets\\VIDEO\\FAKE\\id20_id9_0007.mp4')
print(video)
