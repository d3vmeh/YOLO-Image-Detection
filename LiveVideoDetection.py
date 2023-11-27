from process_methods import *
from ImageDetection import detect_image 
import cv2

vid = cv2.VideoCapture(0)

macbook_camera = Camera(50, 0.111*1080, 0.111*1920)
logitech_webcam = Camera(3.67, 3.6, 4.8)
phone_camera = Camera(4, 3024 * 1.4, 4042 * 1.4)

while True:

    ret, frame = vid.read()
    
    new_frame = frame
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_pil = detect_image(image_pil,camera=logitech_webcam)
    new_frame = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow('Video',new_frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

vid.release()
cv2.destroyAllWindows()
