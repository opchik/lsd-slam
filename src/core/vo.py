import os
import cv2
import numpy as np

from src.models import App, Image

# calibration matrix
w = 12.80
h = 10.24
fx = 0.4
fy = 0.53
cx = 0.5
cy = 0.5
K = np.array([[w*fx, 0, w*cx-0.5], [0, h*fy, h*cy-0.5], [0, 0, 1]])


def find_keypoints_descriptors(app: App, img_name: str):
    """
    function takes a picture with certain path
    and finds all keypoints and descriptors
    """
    img_path = cv2.imread(img_name)
    img = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)
    orb_detector = cv2.ORB_create()
    kp, desc = orb_detector.detectAndCompute(img, None)
    image = Image(imp_path=img_name, img_gray=img_name, kp=kp, desc=desc)
    app.images_data.append(image)


def find_all_matrices(app: App):
    """
    function takes two pictures from the app
    and finds all matrices
    """
    if len(app.images_data) > 2:
        app.images_data.pop(0)
    img1, img2 = app.images_data

    # matches
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf_matcher.match(img1.desc, img2.desc)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:20]
    points1 = np.float32([img1.kp[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    points2 = np.float32([img2.kp[m.trainIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )

    # F_matrix
    F_matrix, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 0.1, 0.99)

    # essential matrix
    E = np.dot(np.dot(K.T, F_matrix), K)

    # R, T matrices
    ret, R, t, _ = cv2.recoverPose(E, points1, points2)
    # R1, R2, T = cv2.decomposeEssentialMat(E)
    T = np.hstack((R, t))
    T = np.vstack((T, np.array((0, 0, 0, 1), dtype=np.float32)))
    if len(app.T) == 0:
        app.T = np.dot(np.eye(4), T)
    else:
        app.T = np.dot(app.T, T)
    return (points1, points2)

def find_points(app: App, points1, points2, P_new):
    # prev_matched_points = cv2.undistortPoints( 
    #         prev_matched_keypoints, camera_matrix, distortion_coeffs, None, None) 
    # matched_points = cv2.undistortPoints( 
    #         matched_keypoints, camera_matrix, distortion_coeffs, None, None) 
    R, T = app.T[0:-1, 0:-1], app.T[0:-1, 3]
    P_prev = P_new
    P_new = np.hstack((R.T, np.dot(-R.T, T).reshape((3, 1))))
    P_new = np.dot(K, P_new)
    points_4d_homogeneous = cv2.triangulatePoints(P_prev, P_new, points1, points2)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d_homogeneous.T)
    if not app.points:
        app.points["x"] = np.hstack(points_3d[:,:,0])
        app.points["y"] = np.hstack(points_3d[:,:,1])
        app.points["z"] = np.hstack(points_3d[:,:,2])
    else:
        app.points["x"] = np.hstack((app.points["x"], np.hstack(points_3d[:,:,0])))
        app.points["y"] = np.hstack((app.points["y"], np.hstack(points_3d[:,:,1])))
        app.points["z"] = np.hstack((app.points["z"], np.hstack(points_3d[:,:,2])))
    return P_new

def vo_cicle(app: App):
    """main cicle of visual odometry"""
    extension = ".jpg"
    all_img_names = [
        os.path.join(app.images_path, f)
        for f in os.listdir(app.images_path)
        if f.endswith(extension)
    ]
    P = np.hstack((np.eye(3), np.zeros((3, 1))))
    for img_name in sorted(all_img_names)[:4000]:
        find_keypoints_descriptors(app, img_name)
        if len(app.images_data) > 1:
            points1, points2 = find_all_matrices(app)
            P = find_points(app, points1, points2, P)