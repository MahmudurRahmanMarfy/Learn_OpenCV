##############################################################################################################

import cv2
import sys
import time

previous_time = 0

s = 0

if len(sys.argv) > 1:
    s = int(sys.argv[1])

source = cv2.VideoCapture(s)
window_name = 'Camera Preview'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


while cv2.waitKey(1) != 27:
    has_frame , frame = source.read()
    if not has_frame:
         break
    
    # Displaying FPS on the frame
    
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    print(f"FPS: {int(fps)}")

    cv2.putText(frame, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 0, 200), 2)

    cv2.imshow(window_name, frame)

source.release()
cv2.destroyWindow(window_name)

##############################################################################################################