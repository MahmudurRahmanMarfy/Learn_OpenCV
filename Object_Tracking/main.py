import cv2
import numpy as np
import time
from object_detection import ObjectDetection


pre_time = 0
crnt_time = 0
fps = 0

# Initialize the object detection model

OD_model = ObjectDetection()

source = cv2.VideoCapture("traffic.mp4")

window_name = 'Camera Preview'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    has_frame, frame = source.read()

    # Object detection on frame
    (class_ids, confidences, boxes) = OD_model.detect(frame)
    
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break;

    #crnt_time = time.time()
    #fps = 1 / (crnt_time - pre_time)
    #pre_time = crnt_time
    #print(f"FPS: {int(fps)}")
    
    #cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 50, 200), 2)
    
    cv2.imshow(window_name , frame)



source.release()
cv2.destroyWindow(window_name)
