# Author:        From openCV
# Upadeted by:   Junesh Gautam
# Date:          2023/1/29
# This script is used to calibrate a camera using a set of images of a chessboard pattern.

import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria      = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
'''The termination criteria are set for the calibration process. It specifies the criteria for termination of the iterative optimization algorithm.'''

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp          = np.zeros((6*5,3), np.float32)
objp[:,:2]    = np.mgrid[0:6,0:5].T.reshape(-1,2)
'''The object points are prepared in the real-world space, representing the expected coordinates of corners on a chessboard.'''

# Arrays to store object points and image points from all the images.
objpoints     = [] # 3d point in real world space
imgpoints     = [] # 2d points in image plane.
'''These arrays will store the object points and image points for each image.'''

images = glob.glob('*.jpg')

for fname in images:

    print( "reading filename ", fname )
    img          = cv.imread(fname)
    gray         = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6,5), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (6,5), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
'''The calibrateCamera function is used to calibrate the camera, and it returns the camera matrix (mtx), distortion coefficients (dist), and rotation and translation vectors (rvecs and tvecs)'''

print("camera matrix : ", mtx)
print("distortion matrix: ", dist)
'''Finally, the script prints the camera matrix and distortion matrix obtained after calibration.'''