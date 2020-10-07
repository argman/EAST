# TODO: Consider case of vertical text (maybe try rotatating both ways
# TODO: image proprocessing on warped output to get better OCR performance

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract


# img contains the image data
# coords_str is a string of 8 comma separated coordinates, 4 * (x,y) coordinates)
# returns coords, the original coordinates and the flattened region
def crop_rotated_rect(img, coords_str):
    coords = []
    split_coords_str =  coords_str.strip().split(',')
    for i in range(int(len(split_coords_str)/2)):
        coord = [[int(split_coords_str[2*i]), int(split_coords_str[2*i + 1])]]
        coords.append(coord)
    cnt = np.array(coords)

    # print("shape of cnt: {}".format(cnt.shape))
    rect = cv2.minAreaRect(cnt)
    # print("rect: {}".format(rect))

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
    plt.imshow(warped)
    plt.show()
    return coords, warped

if __name__ == "__main__":
    img_name = '1'

    img_path = 'test_images/'+ img_name + '.jpg'
    info_path = 'tmp/' + img_name + '.txt'

    img = cv2.imread(img_path)

    #configuration setting to convert image to string.  
    configuration = ("-l eng --oem 1 --psm 8")

    # points for test.jpg
    f = open(info_path, 'r')

    all_coords = []
    all_texts = []
    for line in f.readlines():
        coords_str = line.strip()
        coords, flat_rect = crop_rotated_rect(img, coords_str)

        # This will recognize the text from flattened bounding box
        
        text = pytesseract.image_to_string(flat_rect, config=configuration)
        print("{}\n".format(text))

        # only add text if the character is english
        text_to_add = "".join([x if ord(x) < 128 else "" for x in text]).strip()
        print(coords)
        cv2.putText(img, text_to_add, (coords[0][0][0], coords[0][0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)
    
    plt.imshow(img)
    plt.title('Output')
    plt.show()
