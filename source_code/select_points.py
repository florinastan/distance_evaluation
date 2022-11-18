import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_mcs_upperbody.xml')

def select_points(frame,coords):
    old_points = []
    old_frame = frame
    if coords is not None:
        initial_points = []
        for i in coords:
            a = i[0]
            b = i[1]
            c = i[2]
            d = i[3]
            midX = a + int((c-a)/2)
            midY = b + int((d-b)/2)
            partY = b+int((d-b)/7)
            cv2.imwrite('/home/florina/evaluareDistanta/poza.jpg', frame)
            #choosing the box where we found the person to search for the head region
            person = old_frame[b:midY, a:c, ...]
            gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(gray, 1.05, 5)
            res = isinstance(face, tuple)
            if not res:
                face = np.array([[a, b, a + c, b + d] for (a, b, c, d) in face])
                for m, n, p, q in face:
                    midXface = m + int((p - m) / 2)
                    midYface = n + int((q - n) / 2)
                    cv2.circle(person, (midXface, midYface), 5, (0, 255, 0), -1)
                    p0 = np.array([[midXface+a, midYface+b]], np.float32)
                    initial_points.append(p0)
            else:
                #we'll choose a point in the rectangle area to track it, somewhere in the region of the head, most exactly 1/5 down from the rectangle
                p0 = np.array([[midX, partY]], np.float32)
                initial_points.append(p0)
        #covert from list of points to a vector of 2D points
        old_points = np.array(initial_points)
        for geta in initial_points:
            center = (int(geta[0][0]), int(geta[0][1]))
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            cv2.imwrite('/home/florina/evaluareDistanta/cercuri.jpg', frame)

    return old_points

