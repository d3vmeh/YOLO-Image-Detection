# YOLO-Image-Detection
This repository contains code for detecting objects in single images as well as video footage.

Darknet is used to draw bounding boxes around the objects and classify them. These boxes are then drawn on the original image with the labels included. For the video, each frame is analyzed and edited by Darknet before being pieced together to recreate the video with the annotated frames.

# Object Detection using Driver Footage
https://youtu.be/eYuPTQJ8zdk

# Distance Estimation

Using the camera's focal length, the camera's sensor dimensions, the object's actual dimensions, and the height of the bounding box around the object, a mathematical formula can be used to estimate the distance from the object to the camera. 

However, this requires information about the object's actual dimensions, which could vary. 
