import cv2
import argparse

img = cv2.imread('3.png')
cv2.imshow('img', img)
#cv2.waitKey(0)

clickCoord = []
cropping = False




def click_and_crop_cb(event, x, y, flags, params):
	#print("IN")
	global clickCoord, cropping

	if event == cv2.EVENT_LBUTTONDOWN:
		clickCoord = [(x,y),]
		cropping = True

	elif event == cv2.EVENT_LBUTTONUP:
		clickCoord.append( (x,y) )
		cropping = False
		
		cv2.rectangle(img, clickCoord[0], clickCoord[1], (0, 255, 0), 2)
		cv2.imshow("image", img)
		cv2.waitKey(0)


cv2.setMouseCallback("img", click_and_crop_cb)

while True:
	cv2.imshow('img', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows() 
	a = 1
	#print(clickCoord)
