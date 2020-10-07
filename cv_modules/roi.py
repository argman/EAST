import cv2
import numpy as np

img = cv2.imread("zoomedSign.jpg")
cv2.imshow("Original", img)
# cv2.waitKey(0)

# Find out the aspect ratio of original image
dims = img.shape
height = dims[0]
width = dims[1]
ar = width / height
print(width, height)

# Dummy ROI coordinates
minX = 50
minY = 70
maxX = 100
maxY = 200

# Crop out the desired region
if (maxX - minX) / (maxY - minY) < ar:  # Desired Y > desired X
    correctedWidth = int((maxY - minY) * ar)
    offset = correctedWidth - (maxX - minX)

    # Check that the corrected AR region doesn't go out of frame
    if minX - offset / 2 < 0:
        croppedImg = img[minY:maxY, 0:correctedWidth]

    elif maxX + offset / 2 > width:
        croppedImg = img[minY:maxY, width - correctedWidth : width]

    else:
        croppedImg = img[minY:maxY, int(minX - offset / 2) : int(maxX + offset / 2)]

else:  # Desired Y > desired X
    correctedHeight = int((maxX - minX) * ar)
    offset = correctedHeight - (maxX - minX)

    # Check that the corrected AR region doesn't go out of frame
    if minY - offset / 2 < 0:
        croppedImg = img[0:correctedWidth, minX:maxX]

    elif maxY + offset / 2 > height:
        croppedImg = img[height - correctedHeight : height, minX:maxX]

    else:
        croppedImg = img[int(minY - offset / 2) : int(maxY + offset / 2), minX:maxX]

cv2.imshow("Cropped", croppedImg)
# cv2.waitKey(0)

resized = cv2.resize(croppedImg, (dims[1], dims[0]), interpolation=cv2.INTER_AREA)
print(resized.shape)
cv2.imshow("Resized Cropped", resized)
cv2.waitKey(0)
