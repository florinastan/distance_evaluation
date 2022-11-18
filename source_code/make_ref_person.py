
from scipy.spatial import distance as dist

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def set_refObj(rectangle_coord, height, coord_X, coord_Y):
    refObj = None
    x = rectangle_coord[0]
    y = rectangle_coord[1]
    w = rectangle_coord[2]
    h = rectangle_coord[3]

    box = [(x, y), (w, y), (w, h), (x, h)]

    (tl, tr, br, bl) = box
    (tlblX, tlblY) = midpoint(tl, tr)
    (trbrX, trbrY) = midpoint(bl, br)

    # compute the Euclidean distance between the midpoints,
    # then construct the reference object
    D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    refObj = ((int(coord_X), int(coord_Y)), D / height)

    return refObj