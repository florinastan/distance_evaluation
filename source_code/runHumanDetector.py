import cv2
from humanDetector import Detector
from select_points import select_points
from make_ref_person import set_refObj, midpoint
from tkinter import filedialog as fd
import itertools
from scipy.spatial import distance as dist
import numpy as np
# parameters for saving the new video
frame_height, frame_width = (1280, 720)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/florina/evaluareDistanta/videouri_licenta_30fps/video_3_1.mp4', fourcc, 15, (frame_height, frame_width))

#Choose the video from folder
filename = fd.askopenfilename(title="Open a video")

# Load a video
cap = cv2.VideoCapture(filename)
height = int(input("Enter left-most person's height"))

#parameters for LK tracker
lk_params = dict(winSize = (15, 15),
                maxLevel = 4,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#create an old frame for LK
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#creating the referance object
refObj = None

total_frames = 0
divider = 0
first_time = True
while (cap.isOpened()):
    ret, frame = cap.read()

    if frame is None:
        break
        #create a new frame for LK
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if (total_frames % 30 == 0):
        new_rectangles = Detector(frame)

        people_count = len(new_rectangles)
        print(f"people_count = {people_count}")

        # checking if we have enough people in the frame
        if people_count == 0 or people_count == 1:
            continue
        # setting the left-most person in the first frame as our reference person
        if total_frames == 0:
            ref_coords = new_rectangles[0]

        #selecting the points we want to track
        old_points = select_points(frame,new_rectangles)


    #run the LK tracker to obtain the new points
    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
    #checking if all the optical flow objects go out of the frame
    if new_points is None:
        old_points = select_points(frame,new_rectangles)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)

    # Select good points
    good_new = new_points[status == 1]
    good_old = old_points[status == 1]

    #if the reference object is None, we'll set it
    #copying the new points in a new list
    person_point = []
    for i, new in enumerate(good_new):
        a, b = new.ravel()
        if refObj is None:
            refObj = set_refObj(ref_coords, height, a, b)
        person_point.append((int(a), int(b)))

    #for each selected point that indicates a person, we'll draw a circle
    for person in person_point:
        cv2.circle(frame, person, 5, (0, 255, 0), -1)

    #making pairs of 2 persons
    #eg. for 3 people we'll get the next pairs:(p1,p2), (p2,p3), (p1,p3)
    combinations_all_people = list(itertools.combinations(person_point, 2))

    distances = []
    for (person1, person2) in combinations_all_people:

        # compute the Euclidean distance between the coordinates,
        # and then convert the distance in pixels to distance in
        # units
        if first_time is True:
            first_time = False
            divider = refObj[1]

        D = dist.euclidean(person1, person2) / divider
        distances.append((D, person1, person2))

    distances.sort()

    n_people = len(person_point)

    closest_distances = []
    for distance in distances[:n_people - 1]:
        closest_distances.append(distance)

    #drawing the lines between persons and writing the distance in cm
    for (distanta, person1, person2) in closest_distances:
        cv2.line(frame, person1, person2,
                 (0, 255, 0), 2)
        (mX, mY) = midpoint(person1, person2)
        cv2.putText(frame, "{:.1f}cm".format(distanta), (int(mX), int(mY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # write a frame in the loop
    out.write(frame)
    frame = cv2.resize(frame, (frame_height, frame_width))
    cv2.imshow('output', frame)
    total_frames += 1

    key = cv2.waitKey(1)
    if key == 27:
        break

    # Updating Previous frame and points
    old_gray = gray_frame.copy()
    old_points = good_new.reshape(-1, 1, 2)
cap.release()
out.release()
cv2.destroyAllWindows()