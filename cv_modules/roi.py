import cv2
import numpy as np
import argparse

img = cv2.imread("3.png")
original_img = img
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 800, 570)
#cv2.imshow("Original", img)
cv2.imshow('img', img)
# cv2.waitKey(0)

clickCoord = []
cropping = False


def click_and_crop_cb(event, x, y, flags, params):
    print("IN")
    global clickCoord, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        clickCoord = [[x,y],]
        cropping = False

    elif event == cv2.EVENT_LBUTTONUP:
        clickCoord.append( [x,y] )
        cropping = True


def crop(img, clickCoord):
    # Find out the aspect ratio of original image
    dims = img.shape
    height = dims[0]
    width = dims[1]
    ar = width / height
    print(width, height)

    # Dummy ROI coordinates
    minX = clickCoord[0][0]
    minY = clickCoord[0][1]
    maxX = clickCoord[1][0]
    maxY = clickCoord[1][1]

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

    #cv2.imshow("Cropped", croppedImg)
    # cv2.waitKey(0)

    resized = cv2.resize(croppedImg, (dims[1], dims[0]), interpolation=cv2.INTER_AREA)
    print(resized.shape)

    return resized, croppedImg


def checkROI(img, clickCoord):
    validROI = 0;    # 0 for non-valid, 1 for tiny/click, 2 for valid zoom
    if len(clickCoord) == 2:
            #cv2.rectangle(img, clickCoord[0], clickCoord[1], (0, 255, 0), 2)
            if (clickCoord[0][0]>clickCoord[1][0]) or (clickCoord[0][1]>clickCoord[1][1]):
                temp = clickCoord[0][0]
                clickCoord[0][0] = clickCoord[1][0]
                clickCoord[1][0] = temp
                temp = clickCoord[0][1]
                clickCoord[0][1] = clickCoord[1][1]
                clickCoord[1][1] = temp
            print(clickCoord[1][0]-clickCoord[0][0])
            print(clickCoord[1][1]-clickCoord[0][1])
            if (clickCoord[1][0]-clickCoord[0][0] < 20) or (clickCoord[1][1]-clickCoord[0][1] < 20):
                print("Tiny ROI selected")
                validROI = 1;
            else:
                validROI = 2;
            #img, cropped =  crop(img, clickCoord)
            #print(  len(e)  )

    return clickCoord, validROI


cv2.setMouseCallback("img", click_and_crop_cb)



while True:
    
    cv2.imshow('img', img)
    cv2.waitKey(3)
    #cv2.destroyAllWindows() 
    if cropping is True:
        clickCoord, validROI = checkROI(img, clickCoord)
        if validROI == 1:
            img = original_img
        elif validROI == 2:
            print("INTO VALID ROI")
            img, img_cropped = crop(img, clickCoord)
        cropping = False
        clickCoord = []
        

