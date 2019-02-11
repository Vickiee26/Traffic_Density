from files import VideoObjectDetection

detector=VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("files/yolo.h5")
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path="traffic-mini.mp4",
                                output_file_path="test1"
                                , frames_per_second=29, log_progress=True)
print(video_path)