import cv2
import numpy as np

class VOextraction():
    def __init__(self, img_path):
        self.img = img_path
        self.kp, self.desc =
        self.img_with_keypoints = self.extract_keypoints_descriptors(self.img_path)

    def extract_keypoints_descriptors(self):
        img_path = cv2.imread(img_path)

        # turn into gray scale
        img = cv2.cvtColor(self.img_path, cv2.COLOR_BGR2GRAY)

        # create a orb detector object
        orb_detector = cv2.ORB_create()

        # get both the keypoints and descriptors
        kp1, desc1 = orb_detector.detectAndCompute(image1, None)

        # draw keypoints on both images
        img_with_keypoints = cv2.drawKeypoints(img, kp, None, (0, 255, 0))

#
#
# class IMGsCompare():
#
#

class Matrix:
    pass

    
def match_2_images(img_path):

    # read images
    img1 = cv2.imread(img_path[0]) 
    img2 = cv2.imread(img_path[1]) 
    cv2.imshow('Original image', img1) 

    # turn into gray scale
    image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # create a orb detector object 
    orb_detector = cv2.ORB_create()

    # get both the keypoints and descriptors 
    kp1, desc1 = orb_detector.detectAndCompute(image1, None) 
    kp2, desc2 = orb_detector.detectAndCompute(image2, None)
    print("type kps: ", type(kp1[0]))
    print("type desc: ", type(desc1[0]))


    # draw keypoints on both images 
    image1_with_keypoints = cv2.drawKeypoints(image1, kp1, None, (0, 255, 0)) 
    image2_with_keypoints = cv2.drawKeypoints(image2, kp2, None)
    print("type img_with_kp: ", type(image1_with_keypoints))






    # use bfmatcher to match the descriptors 
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck= True) 

    matches = bf_matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)

    # extract good matches 
    good_matches = matches[:10]  
    print("type matches:  ", type(good_matches[0]))

    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    h, status = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    print("\n\nh:\n", h, "\ntype:\n", type(h))

    matches_image = cv2.drawMatches(
            image1, 
            kp1,  
            image2, 
            kp2, 
            good_matches,  
            None,  
            matchColor=(0, 255, 0),
            singlePointColor=(0, 255, 0),
            flags=2
        ) 
    cv2.imshow("Matches", matches_image)

