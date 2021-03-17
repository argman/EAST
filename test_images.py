from run_demo_server import get_predictor
import cv2 as cv
import os
import numpy as np

import flags
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def rst2np(rst):
    boxes = []
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype=np.int32)
        boxes.append(d.reshape((4, 2)))
    return boxes

print('path to images:', FLAGS.test_data_path)
imgnames = [f for f in os.listdir(FLAGS.test_data_path)
            if os.path.isfile(os.path.join(FLAGS.test_data_path, f))
            and os.path.splitext(f)[1] != '.txt']
assert len(imgnames)

os.makedirs(FLAGS.output_dir, exist_ok=True)

model_loaded = False

winname = 'EAST'
cv.namedWindow(winname, cv.WINDOW_KEEPRATIO)
wait_ms = 0

for imgname in imgnames:
    # read test image
    imgpath = os.path.join(FLAGS.test_data_path, imgname)
    img = cv.imread(imgpath, cv.IMREAD_COLOR)
    if img is None:
        print('%s is not an image! Skipping...' % imgpath)
        continue

    # detect text boxes if not previously detected
    outpath = os.path.join(FLAGS.output_dir, 'res_' + os.path.splitext(imgname)[0] + '.txt')
    if not os.path.isfile(outpath):
        print('Detecting text boxes for %s' % imgname)
        if not model_loaded:
            print('Loading model from %s' % FLAGS.checkpoint_path)
            predict = get_predictor(FLAGS.checkpoint_path)
            model_loaded = True
        rst = predict(img)
        print('Process took %.2f seconds' % (rst['timing']['overall']))
        boxes = rst2np(rst)
        # write out detection result
        with open(outpath, 'w') as f:
            for d in boxes:
                for p in d.reshape(-1, 1):
                    f.write('%d,' % p)
                f.write('\n')

    # read corresponding detection result
    boxes = []
    with open(outpath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line[0:-1]   # strip newline
        output = line.split(',')
        if len(output) != 9:
            print('Invalid output in %s: %s' % (outpath, line))
            continue
        box = np.array([int(b) for b in output[0:8]], dtype=np.int32).reshape((4, 2))
        boxes.append(box)

    # display result
    illu = cv.polylines(img, boxes, isClosed=True, color=(255, 255, 0),
                        thickness=3, lineType=cv.LINE_AA)
    cv.imshow(winname, illu)
    ch = cv.waitKey(wait_ms) & 0xFF

    if chr(ch).lower() == 's':
        outname = os.path.join(output_dir, imgname)
        if cv.imwrite(outname, illu):
            print('Wrote out %s' % outname)
        else:
            print('Could not write %s' % outname)

    if ch == 27:    # ESC
        break
