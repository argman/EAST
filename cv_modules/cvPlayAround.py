import cv2

img = cv2.imread('1.jpg')
cv2.imshow('original', img)
cv2.waitKey(0)

#img_filtered = cv2.GaussianBlur(img, (5, 5), 0)
#cv2.imshow('filtered', img_filtered)
#cv2.waitKey(0)

#alpha = 3
#beta = 1-alpha 

#weighted = cv2.addWeighted(img, alpha, img_filtered, beta, 0.0)
#cv2.imshow('better?', weighted)
#cv2.waitKey(0)

img_gray = cv2.bitwise_not( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) )

rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, rectKern)
 
temp, img_otsu_org = cv2.threshold(img_gray,0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
temp, img_otsu = cv2.threshold(blackhat,0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blackhat_org = cv2.morphologyEx(  ( img_otsu_org )  , cv2.MORPH_BLACKHAT, rectKern)

cv2.imshow('better?', blackhat)
cv2.waitKey(0)

cv2.imshow('OTSU of original gray', blackhat_org )
cv2.waitKey(0)

cv2.imshow('OTSU', img_otsu)
cv2.waitKey(0)
