from process_methods import *
from ImageDetection import detect_image, darknet, anchors, labels
import cv2


def detect_video(video_path, output_path, obj_thresh = 0.5, nms_thresh = 0.3, darknet=darknet, net_h=416, net_w=416, anchors=anchors, labels=labels):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    num_frame = 0
    while vid.isOpened():
      ret, frame = vid.read()
      num_frame += 1
      #print("Frame {} ".format(num_frame))
      if (num_frame == 10000 or num_frame == 30000):
          print("hi")
      if ret:
          ### YOUR CODE HERE
          new_frame = frame
          image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
          image_pil = detect_image(image_pil)
          new_frame = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)
          ### END CODE
          out.write(new_frame)
      else:
          break
    vid.release()
    out.release()


video_path = 'videos/NYCvideo.mp4'
output_path = 'resultant_videos/resultant_video3.mp4'
detect_video(video_path, output_path)