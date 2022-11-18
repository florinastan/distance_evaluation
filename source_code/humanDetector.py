import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

#function to find the area of the rectangle
def rect_area(rect):
    return abs(rect[0] - rect[2]) * abs(rect[1] - rect[3])

def Detector(frame):
    winSize = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    # Histogram of Oriented Gradients Detector
    # Create a HOGDescriptor object
    HOGCV = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)


    # Initialize the People Detector
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = HOGCV.detectMultiScale(frame, winStride=(8,8), padding=(4, 4), scale=1.04)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

    areas = [rect_area(rect) for rect in rects]
    good_rects = []

    for rect, weight in zip(rects, weights):
        if rect_area(rect) > (max(areas) / 5) and weight > 0.5:
            good_rects.append(rect)

    good_rects = np.array(good_rects)

    pick = non_max_suppression(good_rects, probs=None, overlapThresh=0.7)
    for x, y, w, h in pick:
        cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 5, 2)

    cv2.imshow('output', frame)
    return pick
