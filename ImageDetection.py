import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from process_methods import *


labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


image_path = "images/image1.jpeg"
model_path = "yolo_weights.h5"

image_pil = Image.open(image_path)
image_w, image_h = image_pil.size
#plt.imshow(image_pil)
#plt.show()

#Set image size to Darknet input image size
scaled_image_height = 416
scaled_image_width = 416

new_image = preprocess_input(image_pil,scaled_image_height,scaled_image_width)
#plt.imshow(new_image[0])
#plt.show()

#Loading Darknet model and making predictions
darknet = tf.keras.models.load_model(model_path)
yolo_outputs = darknet.predict(new_image)
print(len(yolo_outputs), yolo_outputs[0].shape, yolo_outputs[1].shape, yolo_outputs[2].shape)




anchors = [[[116,90], [156,198], [373,326]], [[30,61], [62,45], [59,119]], [[10,13], [16,30], [33,23]]]


def detect_image(image_pil, obj_thresh = 0.4, nms_thresh = 0.45, darknet=darknet, net_h=416, net_w=416, anchors=anchors, labels=labels):
  new_image = preprocess_input(image_pil, net_h, net_w)


  yolo_outputs = darknet.predict(new_image)


  boxes = decode_netout(yolo_outputs, obj_thresh, anchors, image_pil.size[1], image_pil.size[0], net_h, net_w) #width and height are swapped here
  boxes = do_nms(boxes, nms_thresh, obj_thresh)
  return draw_boxes(image_pil, boxes, labels)


#Values to tune
obj_threshold = 0.4
nms_threshold = 0.45

#Drawing bounding boxes
annotated_image = detect_image(image_pil,obj_thresh=obj_threshold,nms_thresh=nms_threshold)
annotated_image.save("resultant_image.jpg")
plt.figure(figsize=(12,12))
plt.imshow(annotated_image)
plt.show()