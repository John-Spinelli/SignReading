import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

'''
################ FUNCTIONS ################
'''

def getMask(img):
    ''' returns mask of the template image '''
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    max_a = None
    max_cont = None

    # Find the outermost contour
    for cont in contours:
        # First contour
        if max_a == None:        
            max_a = cv2.contourArea(cont)
            max_cont = cont
            
        elif cv2.contourArea(cont) > max_a:
            # Longer contour than before
            max_a = cv2.contourArea(cont)
            max_cont = cont

    # Create mask from contour
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask,[max_cont],0,255,-1,)

    cv2.imshow('mask',mask)
    return mask        

# Preprocessing of given image
def preproc(img):
    ''' Return smoothed image, canny edges and grayscale '''
    blur = cv2.blur(img, (5,5))
    cv2.imshow('blurred',blur)

    edges = cv2.Canny(blur, 140,200)
    cv2.imshow('canny',edges)

    return blur, edges

def getMatches(img1, img2):
    ''' This function just returns the number of matches for a pair
    of images. Use it to find best pair to combine '''
    # Create the SIFT detector
    sift = cv2.SIFT_create()
    keys1, desc1 = sift.detectAndCompute(img1,None)
    keys2, desc2 = sift.detectAndCompute(img2,None)

    # Brute Force Matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Ratio Test
    good_match = []

    # Compare distances between two nearest points
    for m,n in matches:
        if m.distance < 0.4*n.distance:
            good_match.append([m])  

    ''' Just for display purposes, in case you want to visualize the matches '''
    matched_img = cv2.drawMatchesKnn(img1, keys1, img2, keys2, good_match, None, flags = 2)
    cv2.imshow('matched_img',matched_img)
    ''' End Display '''
    return 


def matchImages(img1, img2):
    ''' This function computes the homography to transform img1
        to match with img2. Returns the transformed img1 '''

    # Get grayscale images
    img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    keys1, desc1 = sift.detectAndCompute(img1_g,None)
    keys2, desc2 = sift.detectAndCompute(img2_g,None)

    keys1 = np.float32([kp.pt for kp in keys1])
    keys2 = np.float32([kp.pt for kp in keys2])

    # Brute Force Matching
    bf = cv2.DescriptorMatcher_create("BruteForce")
    matches = bf.knnMatch(desc1, desc2, 2)

    # Ratio Test
    good_match = []
    # Compare distances between two nearest points
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_match.append((m.trainIdx,m.queryIdx))

    pts1 = np.float32([keys1[i] for (_, i) in good_match])
    pts2 = np.float32([keys2[i] for (i, _) in good_match])
    (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)

    # Warp img1 to its matching perspective
    result = cv2.warpPerspective(img1, H,
    	(img1.shape[1] + img2.shape[1], img1.shape[0]+img2.shape[0]))

    result = np.array(result)
    result = result[0:img2.shape[0],0:img2.shape[1]]
    res = imutils.resize(result, width = 400)
    
    return result

def isolateSign(image, mask):
    ''' returns just the sign from transformed image '''
    extracted = cv2.bitwise_and(image,image, mask = mask)
    return extracted


image = cv2.imread('sign_2.png')
cv2.imshow('sign_2',image)
b_im, e_im = preproc(image)
mask = getMask(e_im)

test = cv2.imread('test_3.jpg')
b_test, e_test = preproc(test)
#getMatches(b_test, b_im)
#result = matchImages(b_test, b_im)
result = matchImages(test, image)
cv2.imshow("result",result)

result = isolateSign(result, mask)
cv2.imshow("Isolated",result)

'''
Structure:

take sign image, preprocess and use as template
> LP filter
> get edges for edge comparison 

take input image, preprocess
> LP filter
> get edges for edge comparison

Run SIFT to get the matches with the template sign

Transform image to be aligned with camera

get mask from edges and erase everything but the mask content

run tesseract on the resulting image


'''

cv2.waitKey(0)
cv2.destroyAllWindows()
