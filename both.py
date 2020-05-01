import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os, sys
# Base64 encoding packages.
from PIL import Image
import base64
from io import StringIO
import io
# socket.io
#import socketio
import json

# classes
classes_path = "/scratch/buchanam/cv/all/data/classes/coco.names"
classes_file = open(classes_path, 'r')
list_classes = classes_file.readlines()
list_classes = [x.replace('\n', '') for x in list_classes]

#print(list_classes)
# serlizer
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

#sio = socketio.Client()
#sio.connect('http://localhost:8081')
#sio.emit("messages", "python")

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "/scratch/buchanam/cv/all/models/yolov3_coco.pb"
video_path = "/scratch/buchanam/dataset/" + sys.argv[1]

num_classes     = 80
# colors for classes
import random
import colorsys
hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
random.seed(0)
random.shuffle(colors)
random.seed(None)

input_size      = 1280
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

pred_bbox_mot = []
END_FRAME = 300

with tf.Session(graph=graph) as sess:
    starttime=time.time()
    frame_n = 1
    # read from video
    if(sys.argv[1] != "cam"):
        vid = cv2.VideoCapture(video_path)
        while (frame_n <= END_FRAME):
            return_value, frame = vid.read()
            #if(frame_n % 1 == 0):
            if(frame_n <= END_FRAME):
                if return_value:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                    #frame_n += 1
                    print("Frame: % 4d\n" %(frame_n))
                    #print(pred_bbox)
                else:
                    usedtime=time.time()-starttime
                    #print("total processtime:",usedtime)
                    break;
                    raise ValueError("No image!")
                frame_size = frame.shape[:2]
                image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
                image_data = image_data[np.newaxis, ...]

                #prev_time = time.time()

                pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                    [return_tensors[1], return_tensors[2], return_tensors[3]],
                            feed_dict={ return_tensors[0]: image_data})

                pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

                bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
                bboxes = utils.nms(bboxes, 0.45, method='nms')

                #curr_time = time.time()
                #exec_time = curr_time - prev_time
                #info = "time: %f" %(exec_time)
                #print(info)

                new_bboxes = []
                for x in bboxes:
                    if(round(x[4]*100.0,1) > 45 and x[5] < 9):
                        new_bboxes.append(x)
                bboxes = new_bboxes
                # image = utils.draw_bbox(frame, bboxes)

                bboxes_list = []
                if(len(bboxes) != 0):
                    #print("Objects Detected:% 3d\n" % (len(bboxes)))
                    x = 0
                    while(x < len(bboxes)):
                        #print([bboxes[x][5],list_classes[bboxes[x][5].astype(np.int64)], round(bboxes[x][4]*100.0,1), colors[bboxes[x][5].astype(np.int64)]])
                        #print([list_classes[bboxes[x][5].astype(np.int64)], json.dumps(round(bboxes[x][4]*100.0,1), cls=MyEncoder), json.dumps(colors[bboxes[x][5].astype(np.int64)], cls=MyEncoder)])
                        #bboxes_list.append([list_classes[bboxes[x][5].astype(np.int64)], json.dumps(round(bboxes[x][4]*100.0,1), cls=MyEncoder), json.dumps(colors[bboxes[x][5].astype(np.int64)], cls=MyEncoder)])
                        #print(bboxes[x])
                        pred_bbox_mot.append([int(frame_n), -1, bboxes[x][0], bboxes[x][1], abs(bboxes[x][2]-bboxes[x][0]), abs(bboxes[x][3] - bboxes[x][1]), 1, -1, -1, -1])
                        #pred_bbox_mot.append([int(frame_n), -1, bboxes[x][0], bboxes[x][1], bboxes[x][2], bboxes[x][3], bboxes[x][4]])
                        x += 1
                result = np.asarray(image)
                #else:
                #    pred_bbox_mot.append([int(frame_n), 0, 0, 0, 0, 0, 0])
                #new_bboxes_list = json.dumps(bboxes_list)
                #print(new_bboxes_list)


                # display on a window
                #result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                #cv2.imshow("frame", result)
                #if cv2.waitKey(1) & 0xFF == ord('q'): break

                # send to node.js
                #im = Image.fromarray(result.astype("uint8"))
                #rawBytes = io.BytesIO()
                #im.save(rawBytes, "JPEG", quality=100)
                #rawBytes.seek(0)  # return to the start of the file
                #base64_encoding = base64.b64encode(rawBytes.read())
                #sio.emit("frame-to-server", str(base64_encoding))
                #sio.emit("objects-to-server", new_bboxes_list)
                #print(str(base64_encoding))

                # store into a folder
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if (frame_n < 10):
                    im_name = "00000" + str(frame_n) + ".jpg"
                elif (frame_n < 100):
                    im_name = "0000" + str(frame_n) + ".jpg"
                else:
                    im_name = "000" + str(frame_n) + ".jpg"
                path = '../output/img1/' + im_name
                cv2.imwrite(path, frame)

            frame_n += 1

    # read from camera
    else:
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), 'Cannot capture source'
        while cap.isOpened():
            return_value, frame = cap.read()
            if(frame_n % 1 == 0):
            #if(frame_n >= 0):
                if return_value:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                    #frame_n += 1
                    #print("Frame: % 4d\n" %(frame_n))
                    #print(pred_bbox)
                else:
                    usedtime=time.time()-starttime
                    #print("total processtime:",usedtime)
                    break;
                    raise ValueError("No image!")
                frame_size = frame.shape[:2]
                image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
                image_data = image_data[np.newaxis, ...]

                #prev_time = time.time()

                pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                    [return_tensors[1], return_tensors[2], return_tensors[3]],
                            feed_dict={ return_tensors[0]: image_data})

                pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

                bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
                bboxes = utils.nms(bboxes, 0.45, method='nms')

                #curr_time = time.time()
                #exec_time = curr_time - prev_time
                #info = "time: %f" %(exec_time)
                #print(info)

                new_bboxes = []
                for x in bboxes:
                    if(round(x[4]*100.0,1) > 70 and x[5] < 10):
                        new_bboxes.append(x)
                bboxes = new_bboxes
                image = utils.draw_bbox(frame, bboxes)

                # MAZEN: Display Objects Detected when there are objected to be detected
                bboxes_list = []
                if(len(bboxes) != 0):
                    #print("Objects Detected:% 3d\n" % (len(bboxes)))
                    x = 0
                    while(x < len(bboxes)):
                        #print([bboxes[x][5],list_classes[bboxes[x][5].astype(np.int64)], round(bboxes[x][4]*100.0,1), colors[bboxes[x][5].astype(np.int64)]])
                        #print([list_classes[bboxes[x][5].astype(np.int64)], json.dumps(round(bboxes[x][4]*100.0,1), cls=MyEncoder), json.dumps(colors[bboxes[x][5].astype(np.int64)], cls=MyEncoder)])
                        bboxes_list.append([list_classes[bboxes[x][5].astype(np.int64)], json.dumps(round(bboxes[x][4]*100.0,1), cls=MyEncoder), json.dumps(colors[bboxes[x][5].astype(np.int64)], cls=MyEncoder)])
                        #print(bboxes[x])
                        #pred_bbox_mot.append([int(frame_n), -1, bboxes[x][0], bboxes[x][1], abs(bboxes[x][2]-bboxes[x][0]), abs(bboxes[x][4] - bboxes[x][1]), bboxes[x][5]])
                        #pred_bbox_mot.append([int(frame_n), -1, bboxes[x][0], bboxes[x][1], bboxes[x][2], bboxes[x][3], bboxes[x][4]])
                        x += 1
                new_bboxes_list = json.dumps(bboxes_list)
                #print(new_bboxes_list)

                result = np.asarray(image)

                # display on a window
                #result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                #cv2.imshow("frame", result)
                #if cv2.waitKey(1) & 0xFF == ord('q'): break

                # send to node.js
                im = Image.fromarray(result.astype("uint8"))
                rawBytes = io.BytesIO()
                im.save(rawBytes, "JPEG", quality=100)
                rawBytes.seek(0)  # return to the start of the file
                base64_encoding = base64.b64encode(rawBytes.read())
                sio.emit("frame-to-server", str(base64_encoding))
                sio.emit("objects-to-server", new_bboxes_list)
                #print(str(base64_encoding))

                # store into a folder
                #if (frame_n < 10):
                #    im_name = "00000" + str(frame_n) + ".png"
                #elif (frame_n < 100):
                #    im_name = "0000" + str(frame_n) + ".png"
                #else:
                #    im_name = "000" + str(frame_n) + ".png"
                #path = '../output/img1/' + im_name
                #cv2.imwrite(path, result)

            frame_n += 1
#np.savetxt('../output/det/det.txt', pred_bbox_mot, fmt="%i,%i,%.1f,%.1f,%.1f,%.1f,%i,%i,%i,%i")




