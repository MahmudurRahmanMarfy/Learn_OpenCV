import cv2
import sys

s = 0
if len(sys.argv) > 1:
     s = sys.argv[1]

source = cv2.VideoCapture(s, cv2.CAP_DSHOW)
source.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
source.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)

win_name = 'Camera Preview'
cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 1080, 720)

while cv2.waitKey(1) != 27: #(window close bt ESC button)
    has_frame , frame = source.read()
    if not has_frame:
         break
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)