def nms(box_list, IOU_THRESH=0.65):
    sorted_list = sorted(box_list, key=lambda box: (box[0], box[1]))
    result_frame = sorted_list[:1]

    for bbox in sorted_list[1:]:
        if iou(result_frame[-1], bbox) > IOU_THRESH:
            result_frame[-1] = get_new_corners(bbox, result_frame[-1])
        else:
            result_frame.append(bbox)

    return result_frame


def get_w_h(box):
    w = box[2] - box[0]
    h = box[3] - box[1]
    return w, h


def iou(box1, box2):
    b1_width, b1_height = get_w_h(box1)
    b2_width, b2_height = get_w_h(box2)

    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersection = intersecting_area(box1, box2)
    if intersection > min(b1_area, b2_area) * 0.80:
        return 1.0

    union = b1_area + b2_area - intersection
    return intersection / union


def intersecting_area(box1, box2):
    [l1, t1, r1, b1] = box1
    [l2, t2, r2, b2] = box2

    left = max(l1, l2)
    right = min(r1, r2)
    top = max(t1, t2)
    bottom = min(b1, b2)

    if right > left and bottom > top:
        return (right - left) * (bottom - top)
    return 0


def get_new_corners(box1, box2):
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box1[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    return x1, y1, x2, y2
