import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()

img1 = cv2.imread("Data/1_wan.jpg", 0)
img2 = cv2.imread("Data/agari2.png", 0)

sift = cv2.SIFT_create()
kp1, desc1 = sift.detectAndCompute(img1, None)
kp2, desc2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

good = []
for match1,match2 in matches:
    if match1.distance < 0.8*match2.distance:
        good.append([match1])

sift_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
display(sift_matches)