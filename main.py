### Camera Preview

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






#### Camera filtering

#import cv2
#import sys
#import numpy 
#
#PREVIEW = 0
#BLUR = 1
#FEATURE = 2
#CANNY = 3
#
#featuer_preview = dict( maxCorners=500,
#                        qualityLevel=0.01,  # ensure correct spelling here
#                        minDistance=15,
#                        blockSize=9 )
#
#s = 0
#if len(sys.argv) > 1:
#    s = sys.argv[1]
#
#image_filter = PREVIEW
#alive = True
#
#window_name = 'Camera filter'
#cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
#result = None
#
#source = cv2.VideoCapture(s)
#
#while alive:
#     has_frame, frame = source.read()
#     if not has_frame:
#        break
#    
#     frame = cv2.flip(frame, 1)
#    
#     if image_filter == PREVIEW:
#          result = frame
#     elif image_filter == CANNY:
#         result = cv2.Canny(frame,100,150)
#     elif image_filter == BLUR:
#         result = cv2.blur(frame, (15,15))
#     elif image_filter == FEATURE:
#         result = frame
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         corners = cv2.goodFeaturesToTrack(frame_gray, **featuer_preview)
#         if corners is not None:
#              for x,y in numpy.float32(corners).reshape(-1,2):
#                   cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)
#
#     cv2.imshow(window_name, result)
#
#     key = cv2.waitKey(1)
#     if key == ord('Q') or key == ord('q') or key == 27:
#         alive = False
#     elif key == ord('C') or key == ord('c'):
#         image_filter = CANNY
#     elif key == ord('B') or key == ord('b'):
#         image_filter = BLUR
#     elif key == ord('F') or key == ord('f'):
#         image_filter = FEATURE
#     elif key == ord('P') or key == ord('p'):
#         image_filter = PREVIEW
#
#source.release()
#cv2.destroyWindow(window_name)




### Face Detection

import cv2
import sys

s = 0
if len(sys.argv)>1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = ' Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

net = cv2.dnn.readNetFromCaffe("data/deploy.prototxt", "data/res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Model Parameters

in_width = 300
in_height = 300
mean = [140, 117, 123]
conf_threshold = 0.7

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from a frame

    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB = False, crop = False)

    # Run a model

    net.setInput(blob)
    detections = net.forward()

    for i in range (detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence> conf_threshold:
            x_left_bottom = int(detections[0,0,i,3] * frame_width)
            y_left_bottom = int(detections[0,0,i,4] * frame_height)
            x_right_top = int(detections[0,0,i,5] * frame_width)
            y_right_top = int(detections[0,0,i,6] * frame_height)

            cv2.rectangle(frame, (x_left_bottom,y_left_bottom),(x_right_top, y_right_top), (0,255,0))
            label = "Confidence : %.4f" % confidence
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)

            cv2.rectangle(frame,(x_left_bottom, y_left_bottom- label_size[1]), (x_left_bottom + label_size[0], y_left_bottom + baseline), (255,255,255), cv2.FILLED)
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

    t, _ = net.getPerfProfile()
    label = ' Inference time : %.2f ms' % (t + 1000.0/ cv2.getTickFrequency())
    cv2.putText(frame, label, (0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))

    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)