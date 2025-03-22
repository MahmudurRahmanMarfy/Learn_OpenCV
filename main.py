import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

plt.rcParams['figure.figsize'] = [8,8]
plt.rcParams.update({'font.size': 18})

# define domain
dx = 0.001
L = np.pi
x = L * np.arange(-1+dx, 1+dx, dx)
n = len(x)
nquart = int (np.floor(n/4))

# define hat function
f = np.zeros_like(x)
f[nquart:2*nquart] =(4/n)*np.arange(1, nquart+1)
f[2*nquart : 3*nquart] = np.ones(nquart) - (4/n)*np.arange(0,nquart)
fig, ax = plt.subplots()
ax.plot(x,f,'-', color = 'k' , linewidth = 2)

# Compute Fourier Series

name = "Accent"
cmap = plt.get_cmap('tab10')
colors = cmap.colors
ax.set_prop_cycle( color= colors)

A0 = np.sum(f * np.ones_like(x)) * dx
FFS = A0/2

A = np.zeros(20)
B = np.zeros(20)

for k in range (20):
    A[k] = np.sum(f * np.cos((k+1) * np.pi * x / L)) * dx
    B[k] = np.sum(f * np.sin((k+1) * np.pi * x / L)) * dx
    FFS = FFS + A[k] * np.cos((k+1) * np.pi * x / L) + B[k] * np.sin((k+1) * np.pi * x / L)
    ax.plot(x, FFS, '-')




 musicnotes = cv2.imread("musicnotes.jpg", cv2.IMREAD_GRAYSCALE) 


# Threshold

retval, Thres1 = cv2.threshold(musicnotes, 50, 255, cv2.THRESH_BINARY)
retval, Thres2 = cv2.threshold(musicnotes, 100, 255, cv2.THRESH_BINARY)
retval, Thres3 = cv2.threshold(musicnotes, 200, 255, cv2.THRESH_BINARY)

# Adaptive threshold

adt_threshold1 = cv2.adaptiveThreshold(musicnotes,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,21,15)
adt_threshold2 = cv2.adaptiveThreshold(musicnotes,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,11)

# Plot

plt.figure(figsize=[20,20])

plt.subplot(3,2,1); plt.imshow(musicnotes, cmap="gray"); plt.title("Original", fontsize=12);
plt.subplot(3,2,2); plt.imshow(Thres1, cmap="gray"); plt.title("Threshold 50", fontsize=12);
plt.subplot(3,2,3); plt.imshow(Thres2, cmap="gray"); plt.title("Threshold 100", fontsize=12);
plt.subplot(3,2,4); plt.imshow(Thres3, cmap="gray"); plt.title("Threshold 200", fontsize=12);

plt.subplot(3,2,5); plt.imshow(adt_threshold1, cmap="gray"); plt.title("Adaptive Threshold 1", fontsize=12);
plt.subplot(3,2,6); plt.imshow(adt_threshold2, cmap="gray"); plt.title("Adaptive Threshold 2", fontsize=12);

plt.show()