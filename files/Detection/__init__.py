import numpy as np
import tensorflow as tf
import os
import cv2 
from keras import backend as K
from keras.layers import Input
from PIL import Image
import colorsys

from files.Detection.YOLOv3.models import yolo_main, tiny_yolo_main
from files.Detection.YOLOv3.utils import letterbox_image, yolo_eval

def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 3)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)


def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, detections, color=None, generator=None):
    """ Draws detections in an image.

    # Arguments
        image      : The image to draw on.
        detections : A [N, 4 + num_classes] matrix (x1, y1, x2, y2, cls_1, cls_2, ...).
        color      : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        generator  : (optional) Generator which can map label to class name.
    """
    for d in detections:
        label   = np.argmax(d[4:])
        c       = color if color is not None else label_color(label)
        score   = d[4 + label]
        caption = (generator.label_to_name(label) if generator else str(label)) + ': {0:.2f}'.format(score)
        draw_caption(image, d, caption)

        draw_box(image, d, color=c)


def draw_annotations(image, annotations, color=(0, 255, 0), generator=None):
    """ Draws annotations in an image.

    # Arguments
        image       : The image to draw on.
        annotations : A [N, 5] matrix (x1, y1, x2, y2, label).
        color       : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        generator   : (optional) Generator which can map label to class name.
    """
    for a in annotations:
        label   = a[4]
        c       = color if color is not None else label_color(label)
        caption = '{}'.format(generator.label_to_name(label) if generator else label)
        draw_caption(image, a, caption)

        draw_box(image, a, color=c)

class VideoObjectDetection:
    """
                    This is the object detection class for videos and camera live stream inputs in the ImageAI library. It provides support for RetinaNet,
                     YOLOv3 and TinyYOLOv3 object detection networks. After instantiating this class, you can set it's properties and
                     make object detections using it's pre-defined functions.

                     The following functions are required to be called before object detection can be made
                     * setModelPath()
                     * At least of of the following and it must correspond to the model set in the setModelPath()
                      [setModelTypeAsRetinaNet(), setModelTypeAsYOLOv3(), setModelTinyYOLOv3()]
                     * loadModel() [This must be called once only before performing object detection]

                     Once the above functions have been called, you can call the detectObjectsFromVideo() function
                     or the detectCustomObjectsFromVideo() of  the object detection instance object at anytime to
                     obtain observable objects in any video or camera live stream.
    """

    def __init__(self):
        self.__modelType = ""
        self.modelPath = ""
        self.__modelPathAdded = False
        self.__modelLoaded = False
        self.__model_collection = []
        self.__input_image_min = 1333
        self.__input_image_max = 800
        self.__detection_storage = None

        self.numbers_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                   7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
                   13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                   21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
                   28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                   34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                   39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                   46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
                   53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                   60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
                   67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                   73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                   79: 'toothbrush'}

        # Unique instance variables for YOLOv3 model
        self.__yolo_iou = 0.45
        self.__yolo_score = 0.1
        self.__yolo_anchors = np.array(
            [[10., 13.], [16., 30.], [33., 23.], [30., 61.], [62., 45.], [59., 119.], [116., 90.], [156., 198.],
             [373., 326.]])
        self.__yolo_model_image_size = (416, 416)
        self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes = "", "", ""
        self.sess = K.get_session()

        # Unique instance variables for TinyYOLOv3.
        self.__tiny_yolo_anchors = np.array(
            [[10., 14.], [23., 27.], [37., 58.], [81., 82.], [135., 169.], [344., 319.]])

 

    def setModelTypeAsYOLOv3(self):
        """
                'setModelTypeAsYOLOv3()' is used to set the model type to the YOLOv3 model
                for the video object detection instance instance object .
                :return:
                """
        self.__modelType = "yolov3"


    def setModelPath(self, model_path):
        """
         'setModelPath()' function is required and is used to set the file path to a RetinaNet,
         YOLOv3 or TinyYOLOv3 object detection model trained on the COCO dataset.
          :param model_path:
          :return:
        """

        if(self.__modelPathAdded == False):
            self.modelPath = model_path
            self.__modelPathAdded = True



    def loadModel(self, detection_speed="normal"):
        """
                'loadModel()' function is required and is used to load the model structure into the program from the file path defined
                in the setModelPath() function. This function receives an optional value which is "detection_speed".
                The value is used to reduce the time it takes to detect objects in an image, down to about a 10% of the normal time, with
                 with just slight reduction in the number of objects detected.


                * prediction_speed (optional); Acceptable values are "normal", "fast", "faster", "fastest" and "flash"

                :param detection_speed:
                :return:
        """

        if (self.__modelType == "yolov3"):
            if (detection_speed == "normal"):
                self.__yolo_model_image_size = (416, 416)
            elif (detection_speed == "fast"):
                self.__yolo_model_image_size = (320, 320)
            elif (detection_speed == "faster"):
                self.__yolo_model_image_size = (208, 208)
            elif (detection_speed == "fastest"):
                self.__yolo_model_image_size = (128, 128)
            elif (detection_speed == "flash"):
                self.__yolo_model_image_size = (96, 96)


        if (self.__modelLoaded == False):
            if (self.__modelType == ""):
                raise ValueError("You must set a valid model type before loading the model.")
            
            elif (self.__modelType == "yolov3"):
                model = yolo_main(Input(shape=(None, None, 3)), len(self.__yolo_anchors) // 3,
                                  len(self.numbers_to_names))
                model.load_weights(self.modelPath)

                hsv_tuples = [(x / len(self.numbers_to_names), 1., 1.)
                              for x in range(len(self.numbers_to_names))]
                self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                self.colors = list(
                    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                        self.colors))
                np.random.seed(10101)
                np.random.shuffle(self.colors)
                np.random.seed(None)

                self.__yolo_input_image_shape = K.placeholder(shape=(2,))
                self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes = yolo_eval(model.output,
                                                                                       self.__yolo_anchors,
                                                                                       len(self.numbers_to_names),
                                                                                       self.__yolo_input_image_shape,
                                                                                       score_threshold=self.__yolo_score,
                                                                                       iou_threshold=self.__yolo_iou)

                self.__model_collection.append(model)
                self.__modelLoaded = True


    def detectObjectsFromVideo(self, input_file_path="", camera_input = None, output_file_path="", frames_per_second=20, frame_detection_interval=1, minimum_percentage_probability=50, log_progress=False, display_percentage_probability=True, display_object_name = True, save_detected_video = True, per_frame_function = None, per_second_function = None, per_minute_function = None, video_complete_function = None, return_detected_frame = False ):

        """
                    'detectObjectsFromVideo()' function is used to detect objects observable in the given video path or a camera input:
                            * input_file_path , which is the file path to the input video. It is required only if 'camera_input' is not set
                            * camera_input , allows you to parse in camera input for live video detections
                            * output_file_path , which is the path to the output video. It is required only if 'save_detected_video' is not set to False
                            * frames_per_second , which is the number of frames to be used in the output video
                            * frame_detection_interval (optional, 1 by default)  , which is the intervals of frames that will be detected.
                            * minimum_percentage_probability (optional, 50 by default) , option to set the minimum percentage probability for nominating a detected object for output.
                            * log_progress (optional) , which states if the progress of the frame processed is to be logged to console
                            * display_percentage_probability (optional), can be used to hide or show probability scores on the detected video frames
                            * display_object_name (optional), can be used to show or hide object names on the detected video frames
                            * save_save_detected_video (optional, True by default), can be set to or not to save the detected video
                            * per_frame_function (optional), this parameter allows you to parse in a function you will want to execute after
                                                                each frame of the video is detected. If this parameter is set to a function, after every video
                                                                frame is detected, the function will be executed with the following values parsed into it:
                                                                -- position number of the frame
                                                                -- an array of dictinaries, with each dictinary corresponding to each object detected.
                                                                    Each dictionary contains 'name', 'percentage_probability' and 'box_points'
                                                                -- a dictionary with with keys being the name of each unique objects and value
                                                                    are the number of instances of the object present
                                                                -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed
                                                                    as the fourth value into the function

                            * per_second_function (optional), this parameter allows you to parse in a function you will want to execute after
                                                                each second of the video is detected. If this parameter is set to a function, after every second of a video
                                                                 is detected, the function will be executed with the following values parsed into it:
                                                                -- position number of the second
                                                                -- an array of dictionaries whose keys are position number of each frame present in the last second , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame

                                                                -- an array of dictionaries, with each dictionary corresponding to each frame in the past second, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame

                                                                -- a dictionary with its keys being the name of each unique object detected throughout the past second, and the key values are the average number of instances of the object found in all the frames contained in the past second

                                                                -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed
                                                                    as the fifth value into the function

                            * per_minute_function (optional), this parameter allows you to parse in a function you will want to execute after
                                                                each minute of the video is detected. If this parameter is set to a function, after every minute of a video
                                                                 is detected, the function will be executed with the following values parsed into it:
                                                                -- position number of the minute
                                                                -- an array of dictionaries whose keys are position number of each frame present in the last minute , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame

                                                                -- an array of dictionaries, with each dictionary corresponding to each frame in the past minute, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame

                                                                -- a dictionary with its keys being the name of each unique object detected throughout the past minute, and the key values are the average number of instances of the object found in all the frames contained in the past minute

                                                                -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed
                                                                    as the fifth value into the function

                            * video_complete_function (optional), this parameter allows you to parse in a function you will want to execute after
                                                                all of the video frames have been detected. If this parameter is set to a function, after all of frames of a video
                                                                 is detected, the function will be executed with the following values parsed into it:
                                                                -- an array of dictionaries whose keys are position number of each frame present in the entire video , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame

                                                                -- an array of dictionaries, with each dictionary corresponding to each frame in the entire video, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame

                                                                -- a dictionary with its keys being the name of each unique object detected throughout the entire video, and the key values are the average number of instances of the object found in all the frames contained in the entire video

                            * return_detected_frame (optionally, False by default), option to obtain the return the last detected video frame into the per_per_frame_function,
                                                                                    per_per_second_function or per_per_minute_function






                    :param input_file_path:
                    :param camera_input
                    :param output_file_path:
                    :param save_detected_video:
                    :param frames_per_second:
                    :param frame_detection_interval:
                    :param minimum_percentage_probability:
                    :param log_progress:
                    :param display_percentage_probability:
                    :param display_object_name:
                    :param per_frame_function:
                    :param per_second_function:
                    :param per_minute_function:
                    :param video_complete_function:
                    :param return_detected_frame:
                    :return output_video_filepath:
                    :return counting:
                    :return output_objects_array:
                    :return output_objects_count:
                    :return detected_copy:
                    :return this_second_output_object_array:
                    :return this_second_counting_array:
                    :return this_second_counting:
                    :return this_minute_output_object_array:
                    :return this_minute_counting_array:
                    :return this_minute_counting:
                    :return this_video_output_object_array:
                    :return this_video_counting_array:
                    :return this_video_counting:
                """

        if(input_file_path == ""  and camera_input == None):
            raise ValueError("You must set 'input_file_path' to a valid video file, or set 'camera_input' to a valid camera")
        elif (save_detected_video == True and output_file_path == ""):
            raise ValueError(
                "You must set 'output_video_filepath' to a valid video file name, in which the detected video will be saved. If you don't intend to save the detected video, set 'save_detected_video=False'")

        else:
            try:
                if(self.__modelType == "retinanet"):

                    output_frames_dict = {}
                    output_frames_count_dict = {}


                    input_video = cv2.VideoCapture(input_file_path)
                    if (camera_input != None):
                        input_video = camera_input

                    output_video_filepath = output_file_path + '.avi'

                    frame_width = int(input_video.get(3))
                    frame_height = int(input_video.get(4))
                    output_video = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                                   frames_per_second,
                                                   (frame_width, frame_height))

                    counting = 0
                    predicted_numbers = None
                    scores = None
                    detections = None

                    model = self.__model_collection[0]

                    while (input_video.isOpened()):
                        ret, frame = input_video.read()

                        if (ret == True):

                            output_objects_array = []

                            counting += 1

                            if (log_progress == True):
                                print("Processing Frame : ", str(counting))

                            detected_copy = frame.copy()
                            detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

                            frame = preprocess_image(frame)
                            frame, scale = resize_image(frame, min_side=self.__input_image_min,
                                                        max_side=self.__input_image_max)

                            check_frame_interval = counting % frame_detection_interval

                            if (counting == 1 or check_frame_interval == 0):
                                _, _, detections = model.predict_on_batch(np.expand_dims(frame, axis=0))
                                predicted_numbers = np.argmax(detections[0, :, 4:], axis=1)
                                scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_numbers]

                                detections[0, :, :4] /= scale

                            min_probability = minimum_percentage_probability / 100

                            for index, (label, score), in enumerate(zip(predicted_numbers, scores)):
                                if score < min_probability:
                                    continue

                                color = label_color(label)

                                detection_details = detections[0, index, :4].astype(int)
                                draw_box(detected_copy, detection_details, color=color)

                                if (display_object_name == True and display_percentage_probability == True):
                                    caption = "{} {:.3f}".format(self.numbers_to_names[label], (score * 100))
                                    draw_caption(detected_copy, detection_details, caption)
                                elif (display_object_name == True):
                                    caption = "{} ".format(self.numbers_to_names[label])
                                    draw_caption(detected_copy, detection_details, caption)
                                elif (display_percentage_probability == True):
                                    caption = " {:.3f}".format((score * 100))
                                    draw_caption(detected_copy, detection_details, caption)

                                each_object_details = {}
                                each_object_details["name"] = self.numbers_to_names[label]
                                each_object_details["percentage_probability"] = score * 100
                                each_object_details["box_points"] = detection_details
                                output_objects_array.append(each_object_details)

                            output_frames_dict[counting] = output_objects_array

                            output_objects_count = {}
                            for eachItem in output_objects_array:
                                eachItemName = eachItem["name"]
                                try:
                                    output_objects_count[eachItemName] = output_objects_count[eachItemName] + 1
                                except:
                                    output_objects_count[eachItemName] = 1

                            output_frames_count_dict[counting] = output_objects_count

                            detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

                            if (save_detected_video == True):
                                output_video.write(detected_copy)

                            if (per_frame_function != None):
                                if(return_detected_frame == True):
                                    per_frame_function(counting, output_objects_array, output_objects_count, detected_copy)
                                elif(return_detected_frame == False):
                                    per_frame_function(counting, output_objects_array, output_objects_count)

                            if (per_second_function != None):
                                if (counting != 1 and (counting % frames_per_second) == 0):

                                    this_second_output_object_array = []
                                    this_second_counting_array = []
                                    this_second_counting = {}

                                    for aa in range(counting):
                                        if (aa >= (counting - frames_per_second)):
                                            this_second_output_object_array.append(output_frames_dict[aa + 1])
                                            this_second_counting_array.append(output_frames_count_dict[aa + 1])

                                    for eachCountingDict in this_second_counting_array:
                                        for eachItem in eachCountingDict:
                                            try:
                                                this_second_counting[eachItem] = this_second_counting[eachItem] + \
                                                                                 eachCountingDict[eachItem]
                                            except:
                                                this_second_counting[eachItem] = eachCountingDict[eachItem]

                                    for eachCountingItem in this_second_counting:
                                        this_second_counting[eachCountingItem] = this_second_counting[
                                                                                     eachCountingItem] / frames_per_second


                                    if (return_detected_frame == True):
                                        per_second_function(int(counting / frames_per_second),
                                                            this_second_output_object_array, this_second_counting_array,
                                                            this_second_counting, detected_copy)

                                    elif (return_detected_frame == False):
                                        per_second_function(int(counting / frames_per_second),
                                                            this_second_output_object_array, this_second_counting_array,
                                                            this_second_counting)

                            if (per_minute_function != None):

                                if (counting != 1 and (counting % (frames_per_second * 60)) == 0):


                                    this_minute_output_object_array = []
                                    this_minute_counting_array = []
                                    this_minute_counting = {}

                                    for aa in range(counting):
                                        if (aa >= (counting - (frames_per_second * 60))):
                                            this_minute_output_object_array.append(output_frames_dict[aa + 1])
                                            this_minute_counting_array.append(output_frames_count_dict[aa + 1])

                                    for eachCountingDict in this_minute_counting_array:
                                        for eachItem in eachCountingDict:
                                            try:
                                                this_minute_counting[eachItem] = this_minute_counting[eachItem] + \
                                                                                 eachCountingDict[eachItem]
                                            except:
                                                this_minute_counting[eachItem] = eachCountingDict[eachItem]

                                    for eachCountingItem in this_minute_counting:
                                        this_minute_counting[eachCountingItem] = this_minute_counting[
                                                                                     eachCountingItem] / (frames_per_second * 60)


                                    if (return_detected_frame == True):
                                        per_minute_function(int(counting / (frames_per_second * 60)),
                                                            this_minute_output_object_array, this_minute_counting_array,
                                                            this_minute_counting, detected_copy)

                                    elif (return_detected_frame == False):
                                        per_minute_function(int(counting / (frames_per_second * 60)),
                                                            this_minute_output_object_array, this_minute_counting_array,
                                                            this_minute_counting)


                        else:
                            break

                    if (video_complete_function != None):


                        this_video_output_object_array = []
                        this_video_counting_array = []
                        this_video_counting = {}

                        for aa in range(counting):
                            this_video_output_object_array.append(output_frames_dict[aa + 1])
                            this_video_counting_array.append(output_frames_count_dict[aa + 1])


                        for eachCountingDict in this_video_counting_array:
                            for eachItem in eachCountingDict:
                                try:
                                    this_video_counting[eachItem] = this_video_counting[eachItem] + \
                                                                     eachCountingDict[eachItem]
                                except:
                                    this_video_counting[eachItem] = eachCountingDict[eachItem]

                        for eachCountingItem in this_video_counting:
                            this_video_counting[eachCountingItem] = this_video_counting[
                                                                         eachCountingItem] / counting

                        video_complete_function(this_video_output_object_array, this_video_counting_array,
                                            this_video_counting)


                    input_video.release()
                    output_video.release()

                    if (save_detected_video == True):
                        return output_video_filepath

                elif(self.__modelType == "yolov3" or self.__modelType == "tinyyolov3"):

                    output_frames_dict = {}
                    output_frames_count_dict = {}




                    input_video = cv2.VideoCapture(input_file_path)

                    if(camera_input != None):
                        input_video = camera_input

                    output_video_filepath = output_file_path + '.avi'

                    frame_width = int(input_video.get(3))
                    frame_height = int(input_video.get(4))
                    output_video = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                                   frames_per_second,
                                                   (frame_width, frame_height))

                    counting = 0
                    out_boxes = None
                    out_scores = None
                    out_classes = None

                    model = self.__model_collection[0]

                    while (input_video.isOpened()):
                        ret, frame = input_video.read()

                        if (ret == True):

                            output_objects_array = []

                            counting += 1

                            if (log_progress == True):
                                print("This isFrame : ", str(counting))

                            detected_copy = frame.copy()
                            detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

                            frame = Image.fromarray(np.uint8(frame))

                            new_image_size = (self.__yolo_model_image_size[0] - (self.__yolo_model_image_size[0] % 32),
                                              self.__yolo_model_image_size[1] - (self.__yolo_model_image_size[1] % 32))
                            boxed_image = letterbox_image(frame, new_image_size)
                            image_data = np.array(boxed_image, dtype="float32")

                            image_data /= 255.
                            image_data = np.expand_dims(image_data, 0)

                            check_frame_interval = counting % frame_detection_interval

                            if (counting == 1 or check_frame_interval == 0):
                                out_boxes, out_scores, out_classes = self.sess.run(
                                    [self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes],
                                    feed_dict={
                                        model.input: image_data,
                                        self.__yolo_input_image_shape: [frame.size[1], frame.size[0]],
                                        K.learning_phase(): 0
                                    })


                            min_probability = minimum_percentage_probability / 100

                            for a, b in reversed(list(enumerate(out_classes))):
                                predicted_class = self.numbers_to_names[b]
                                box = out_boxes[a]
                                score = out_scores[a]


                                if score < min_probability:
                                    continue

                                label = "{} {:.2f}".format(predicted_class, score)

                                top, left, bottom, right = box
                                top = max(0, np.floor(top + 0.5).astype('int32'))
                                left = max(0, np.floor(left + 0.5).astype('int32'))
                                bottom = min(frame.size[1], np.floor(bottom + 0.5).astype('int32'))
                                right = min(frame.size[0], np.floor(right + 0.5).astype('int32'))

                                try:
                                    color = label_color(b)
                                except:
                                    color = (255, 0, 0)

                                detection_details = (left, top, right, bottom)
                                draw_box(detected_copy, detection_details, color=color)

                                if (display_object_name == True and display_percentage_probability == True):
                                    draw_caption(detected_copy, detection_details, label)
                                elif (display_object_name == True):
                                    draw_caption(detected_copy, detection_details, predicted_class)
                                elif (display_percentage_probability == True):
                                    draw_caption(detected_copy, detection_details, str(score * 100))

                                each_object_details = {}
                                each_object_details["name"] = predicted_class
                                each_object_details["percentage_probability"] = score * 100
                                each_object_details["box_points"] = detection_details
                                output_objects_array.append(each_object_details)


                            output_frames_dict[counting] = output_objects_array


                            output_objects_count = {}
                            for eachItem in output_objects_array:
                                eachItemName = eachItem["name"]
                                try:
                                    output_objects_count[eachItemName] = output_objects_count[eachItemName] + 1
                                except:
                                    output_objects_count[eachItemName] = 1

                            output_frames_count_dict[counting] = output_objects_count
                            detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

                            if(save_detected_video == True):
                                output_video.write(detected_copy)

                            if(per_frame_function != None):
                                if (per_frame_function != None):
                                    if (return_detected_frame == True):
                                        per_frame_function(counting, output_objects_array, output_objects_count,
                                                           detected_copy)
                                    elif (return_detected_frame == False):
                                        per_frame_function(counting, output_objects_array, output_objects_count)

                            if(per_second_function != None):
                                if(counting != 1 and (counting % frames_per_second) == 0):

                                    this_second_output_object_array = []
                                    this_second_counting_array = []
                                    this_second_counting = {}

                                    for aa in range(counting):
                                        if(aa >= (counting - frames_per_second)):
                                            this_second_output_object_array.append(output_frames_dict[aa + 1])
                                            this_second_counting_array.append(output_frames_count_dict[aa + 1])




                                    for eachCountingDict in this_second_counting_array:
                                        for eachItem in eachCountingDict:
                                            try:
                                                this_second_counting[eachItem] = this_second_counting[eachItem] + eachCountingDict[eachItem]
                                            except:
                                                this_second_counting[eachItem] = eachCountingDict[eachItem]

                                    for eachCountingItem in this_second_counting:
                                        this_second_counting[eachCountingItem] = this_second_counting[eachCountingItem] / frames_per_second



                                    if (return_detected_frame == True):
                                        per_second_function(int(counting / frames_per_second),
                                                            this_second_output_object_array, this_second_counting_array,
                                                            this_second_counting, detected_copy)

                                    elif (return_detected_frame == False):
                                        per_second_function(int(counting / frames_per_second),
                                                            this_second_output_object_array, this_second_counting_array,
                                                            this_second_counting)

                            if (per_minute_function != None):

                                if (counting != 1 and (counting % (frames_per_second * 60)) == 0):


                                    this_minute_output_object_array = []
                                    this_minute_counting_array = []
                                    this_minute_counting = {}

                                    for aa in range(counting):
                                        if (aa >= (counting - (frames_per_second * 60))):
                                            this_minute_output_object_array.append(output_frames_dict[aa + 1])
                                            this_minute_counting_array.append(output_frames_count_dict[aa + 1])

                                    for eachCountingDict in this_minute_counting_array:
                                        for eachItem in eachCountingDict:
                                            try:
                                                this_minute_counting[eachItem] = this_minute_counting[eachItem] + \
                                                                                 eachCountingDict[eachItem]
                                            except:
                                                this_minute_counting[eachItem] = eachCountingDict[eachItem]

                                    for eachCountingItem in this_minute_counting:
                                        this_minute_counting[eachCountingItem] = this_minute_counting[
                                                                                     eachCountingItem] / (frames_per_second * 60)

                                    if (return_detected_frame == True):
                                        per_minute_function(int(counting / (frames_per_second * 60)),
                                                            this_minute_output_object_array, this_minute_counting_array,
                                                            this_minute_counting, detected_copy)

                                    elif (return_detected_frame == False):
                                        per_minute_function(int(counting / (frames_per_second * 60)),
                                                            this_minute_output_object_array, this_minute_counting_array,
                                                            this_minute_counting)




                        else:
                            break

                    if (video_complete_function != None):


                        this_video_output_object_array = []
                        this_video_counting_array = []
                        this_video_counting = {}

                        for aa in range(counting):
                            this_video_output_object_array.append(output_frames_dict[aa + 1])
                            this_video_counting_array.append(output_frames_count_dict[aa + 1])


                        for eachCountingDict in this_video_counting_array:
                            for eachItem in eachCountingDict:
                                try:
                                    this_video_counting[eachItem] = this_video_counting[eachItem] + \
                                                                     eachCountingDict[eachItem]
                                except:
                                    this_video_counting[eachItem] = eachCountingDict[eachItem]

                        for eachCountingItem in this_video_counting:
                            this_video_counting[eachCountingItem] = this_video_counting[
                                                                         eachCountingItem] / counting

                        video_complete_function(this_video_output_object_array, this_video_counting_array,
                                            this_video_counting)


                    input_video.release()
                    output_video.release()

                    if (save_detected_video == True):
                        return output_video_filepath


            except:
                raise ValueError("An error occured. It may be that your input video is invalid. Ensure you specified a proper string value for 'output_file_path' is 'save_detected_video' is not False. "
                                 "Also ensure your per_frame, per_second, per_minute or video_complete_analysis function is properly configured to receive the right parameters. ")
