from run_demo_server import get_predictor, draw_illu
import cv2 as cv

imgpath = 'training_samples/img_%1d.jpg'
print(imgpath)
cap = cv.VideoCapture(imgpath)
assert cap.isOpened()

predict = get_predictor('models/east_icdar2015_resnet_v1_50_rbox')

ch = 0
i = 0
while ch != 27: # ESC is pressed
    _, img = cap.read()
    if img is None:
        break
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    rst = predict(img)
    illu = draw_illu(img.copy(), rst)

    print('Process took %.2f seconds' % (rst['timing']['overall']))
    cv.imshow('img', illu)
    ch = cv.waitKey() & 0xFF

    if chr(ch) == 's':
        outname = 'res_' + str(i) + '.png'
        i += 1
        if cv.imwrite(outname, illu):
            print('Wrote out %s' % outname)
        else:
            print('Could not write %s' % outname)
