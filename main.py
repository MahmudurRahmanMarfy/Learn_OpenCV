## Camera Preview

#import cv2
#import sys
#
#s = 0
#if len(sys.argv) > 1:
#     s = sys.argv[1]
#
#source = cv2.VideoCapture(s, cv2.CAP_DSHOW)
#source.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
#source.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
#
#win_name = 'Camera Preview'
#cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
#cv2.resizeWindow(win_name, 1080, 720)
#
#while cv2.waitKey(1) != 27: #(window close bt ESC button)
#    has_frame , frame = source.read()
#    if not has_frame:
#         break
#    cv2.imshow(win_name, frame)
#
#source.release()
#cv2.destroyWindow(win_name)

# Camera filtering

import cv2
import sys
import numpy 

PREVIEW = 0
BLUR = 1
FEATURE = 2
CANNY = 3

featuer_preview = dict( maxCorners = 500,
                       qaultiyLevel = 0.2,
                       minDistance = 15,
                       blockSize = 9)

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

image_filter = PREVIEW
alive = True

window_name = 'Camera filter'
cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
result = None

source = cv2.VideoCapture(s)

while alive:
     has_frame, frame = source.read()
     if not has_frame:
        break
    
     frame = cv2.flip(frame, 1)
    
     if image_filter == PREVIEW:
          result = frame
     elif image_filter == CANNY:
         result = cv2.Canny(frame, 145,150)
     elif image_filter == BLUR:
         result = cv2.Blur(frame, (13,13))
     elif image_filter == FEATURE:
         result = frame
         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         corners = cv2.goodFeaturesToTrack(frame_gray, **featuer_preview)
         if corners is not None:
              for x,y in numpy.float32(corners).reshape(-1,2):
                   cv2.circle(result, (x,y), 10, (0,255,0), 2)

     cv2.imshow(window_name, result)

     key = cv2.waitkey(1)
     if key == ord('Q') or key == ord('q') or key == 27:
         alive = False
     elif key == ord('C') or key == ord('c'):
         image_filter = CANNY
     elif key == ord('B') or key == ord('b'):
         image_filter = BLUR
     elif key == ord('F') or key == ord('f'):
         image_filter = FEATURE
     elif key == ord('P') or key == ord('p'):
         image_filter = PREVIEW

source.release()
cv2.destroyWindow(window_name)