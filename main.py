import cv2
import numpy as np
import matplotlib.pyplot as plt

musicnotes = cv2.imread("musicnotes.jpg", cv2.IMREAD_GRAYSCALE) 


# Threshold

retval, Thres1 = cv2.threshold(musicnotes, 50, 255, cv2.THRESH_BINARY)
retval, Thres2 = cv2.threshold(musicnotes, 100, 255, cv2.THRESH_BINARY)
retval, Thres3 = cv2.threshold(musicnotes, 200, 255, cv2.THRESH_BINARY)

# Adaptive threshold

adt_threshold1 = cv2.adaptiveThreshold(musicnotes,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,21,15)
adt_threshold2 = cv2.adaptiveThreshold(musicnotes,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,11)

# Plot

plt.figure(figsize=[30,30])

plt.subplot(3,2,1); plt.imshow(musicnotes, cmap="gray"); plt.title("Original", fontsize=12);
plt.subplot(3,2,2); plt.imshow(Thres1, cmap="gray"); plt.title("Threshold 50", fontsize=12);
plt.subplot(3,2,3); plt.imshow(Thres2, cmap="gray"); plt.title("Threshold 100", fontsize=12);
plt.subplot(3,2,4); plt.imshow(Thres3, cmap="gray"); plt.title("Threshold 200", fontsize=12);

plt.subplot(3,2,5); plt.imshow(adt_threshold1, cmap="gray"); plt.title("Adaptive Threshold 1", fontsize=12);
plt.subplot(3,2,6); plt.imshow(adt_threshold2, cmap="gray"); plt.title("Adaptive Threshold 2", fontsize=12);

plt.show()