img_name = 'custom1'

img_path = 'test_images/'+ img_name + '.jpg'

img = cv2.imread("big_vertical_text.jpg")
# points for test.jpg
cnt = np.array([
        [[64, 49]],
        [[122, 11]],
        [[391, 326]],
        [[308, 373]]
    ])
print("shape of cnt: {}".format(cnt.shape))
rect = cv2.minAreaRect(cnt)
print("rect: {}".format(rect))

# the order of the box points: bottom left, top left, top right,
# bottom right
box = cv2.boxPoints(rect)
box = np.int0(box)

print("bounding box: {}".format(box))
cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

# get width and height of the detected rectangle
width = int(rect[1][0])
height = int(rect[1][1])

src_pts = box.astype("float32")
# coordinate of the points in box points after the rectangle has been
# straightened
dst_pts = np.array([[0, height-1],
                    [0, 0],
                    [width-1, 0],
                    [width-1, height-1]], dtype="float32")

# the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# directly warp the rotated rectangle to get the straightened rectangle
warped = cv2.warpPerspective(img, M, (width, height))
