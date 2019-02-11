import numpy as np
import tensorflow as tf
import os
import cv2 
from keras import backend as K
from keras.layers import Input
from PIL import Image
import colorsys
import files.vehicles2
import time
from files.YOLOv3.models import yolo_main
from files.YOLOv3.utils import letterbox_image, yolo_eval

def draw_box(image, box, color, thickness=2):
   
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 3)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)


def draw_boxes(image, boxes, color, thickness=2):
    
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


class VideoObjectDetection:

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
       
             
        self.__modelType = "yolov3"


    def setModelPath(self, model_path):

        if(self.__modelPathAdded == False):
            self.modelPath = model_path
            self.__modelPathAdded = True



    def loadModel(self, detection_speed="normal"):
        

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


    def detectObjectsFromVideo(self, input_file_path="", camera_input = None, output_file_path="", frames_per_second=20, frame_detection_interval=1, minimum_percentage_probability=50, log_progress=False, display_percentage_probability=False, display_object_name = True, save_detected_video = True, per_frame_function = None, per_second_function = None, per_minute_function = None, video_complete_function = None, return_detected_frame = False ):

      
            try:
                

                if(self.__modelType == "yolov3"):

                    output_frames_dict = {}
                    output_frames_count_dict = {}

                    input_video = cv2.VideoCapture(input_file_path)

                    output_video_filepath = output_file_path + '.avi'

                    frame_width = int(input_video.get(3))
                    frame_height = int(input_video.get(4))
                    output_video = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                                   frames_per_second,
                                                   (frame_width, frame_height))

                    cnt_up=0
                    cnt_down=0

                    w=input_video.get(3)
                    h=input_video.get(4)
                    frameArea=h*w
                    areaTH=frameArea/400

                    #Lines
                    line_up=int(2*(h/5))
                    line_down=int(3*(h/5))

                    midline=int(2.5*(h/5))

                    up_limit=int(1*(h/5))
                    down_limit=int(4*(h/5))

                    print("Red line y:",str(line_down))
                    print("Blue line y:",str(line_up))
                    line_down_color=(255,0,0)
                    line_up_color=(255,0,255)



                    pt1 =  [0, line_down]
                    pt2 =  [w, line_down]


                    pts_L1 = np.array([pt1,pt2], np.int32)
                    pts_L1 = pts_L1.reshape((-1,1,2))


                    pt3 =  [0, line_up]
                    pt4 =  [w, line_up]


                    pts_L2 = np.array([pt3,pt4], np.int32)
                    pts_L2 = pts_L2.reshape((-1,1,2))


                    pt5 =  [0, up_limit]
                    pt6 =  [w, up_limit]

                    pts_L3 = np.array([pt5,pt6], np.int32)
                    pts_L3 = pts_L3.reshape((-1,1,2))


                    pt7 =  [0, down_limit]
                    pt8 =  [w, down_limit]


                    pts_L4 = np.array([pt7,pt8], np.int32)
                    pts_L4 = pts_L4.reshape((-1,1,2))


                    pt9 =  [0, midline]
                    pt10 =  [w, midline]


                    pts_L6 = np.array([pt9,pt10], np.int32)
                    pts_L6 = pts_L6.reshape((-1,1,2))

                    #Background Subtractor
                    fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)

                    #Kernals
                    kernalOp = np.ones((3,3),np.uint8)
                    kernalOp2 = np.ones((5,5),np.uint8)
                    kernalCl = np.ones((11,11),np.uint)


                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cars = []
                    max_p_age = 5
                    pid = 1



                    counting = 0
                    out_boxes = None
                    out_scores = None
                    out_classes = None

                    model = self.__model_collection[0]

                    while (input_video.isOpened()):
                        ret, frame = input_video.read()
                        for i in cars:
                            i.age_one()
                        fgmask=fgbg.apply(frame)
                        fgmask2=fgbg.apply(frame)

                        if ret==True:

                            #Binarization
                            ret,imBin=cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
                            ret,imBin2=cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)

                            #OPening i.e First Erode the dilate
                            mask=cv2.morphologyEx(imBin,cv2.MORPH_OPEN,kernalOp)
                            mask2=cv2.morphologyEx(imBin2,cv2.MORPH_CLOSE,kernalOp)

                            #Closing i.e First Dilate then Erode
                            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernalCl)
                            mask2=cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,kernalCl)


                            #Find Contours
                            countours0,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                            for cnt in countours0:
                                area=cv2.contourArea(cnt)
                                print(area)
                                if area>areaTH:
                                    ####Tracking######
                                    m=cv2.moments(cnt)
                                    cx=int(m['m10']/m['m00'])
                                    cy=int(m['m01']/m['m00'])
                                    x,y,w,h=cv2.boundingRect(cnt)

                                    new=True
                                    if cy in range(up_limit,down_limit):
                                        for i in cars:
                                            if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                                                new = False
                                                i.updateCoords(cx, cy)

                                                if i.going_UP(line_down,line_up)==True:
                                                    cnt_up+=1
                                                    print("ID:",i.getId(),'crossed going up at', time.strftime("%c"))
                                                elif i.going_DOWN(line_down,line_up)==True:
                                                    cnt_down+=1
                                                    print("ID:", i.getId(), 'crossed going up at', time.strftime("%c"))
                                                break
                                            if i.getState()=='1':
                                                if i.getDir()=='down'and i.getY()>down_limit:
                                                    i.setDone()
                                                elif i.getDir()=='up'and i.getY()<up_limit:
                                                    i.setDone()
                                            if i.timedOut():
                                                index=cars.index(i)
                                                cars.pop(index)
                                                del i

                                        if new==True: #If nothing is detected,create new
                                            p=vehicles2.Car(pid,cx,cy,max_p_age)
                                            cars.append(p)
                                            pid+=1

                                    cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
                                    #img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                            for i in cars:
                                cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 0.3, i.getRGB(), 1, cv2.LINE_AA)




                            str_up='Crossed: '+str(cnt_up)
                            str_down='DOWN: '+str(cnt_down)
                           # frame=cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
                           # frame=cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
                            frame=cv2.polylines(frame,[pts_L6],False,line_up_color,thickness=2)
                           # frame=cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
                           # frame=cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
                            cv2.putText(frame, str_up, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            
                            #cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                            #cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                            
                            #cv2.imshow('Frame',frame)

                            if cv2.waitKey(1)&0xff==ord('q'):
                                break


                            output_objects_array = []

                            counting += 1

                            if (log_progress == True):
                                print("Processing Frame : ", str(counting))

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

                                #print (predicted_class)
                                s1 = 'person'
                               
                                '''if predicted_class == s1:
                                    continue'''                                    
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
                raise ValueError("An error occured. May be input video is invalid.  ")
