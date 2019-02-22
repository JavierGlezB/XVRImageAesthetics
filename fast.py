import numpy as np
import cv2
from matplotlib import pyplot as plt

# def fast():
# 	img = cv2.imread('./CapptuPhotos/107854_3.jpg',0)
# 	# Initiate FAST object with default values
# 	fast = cv2.FastFeatureDetector_create()
# 	# find and draw the keypoints
# 	kp = fast.detect(img,None)[:500]
# 	print "len kp", len(kp)
# 	img2 = cv2.drawKeypoints(img, kp, color=(255,0,0), outImage=True)
# 	# Print all default params
# 	print "Threshold: ", fast.getThreshold()
# 	print "nonmaxSuppression: ", fast.getNonmaxSuppression()
# 	print "neighborhood: ", fast.getType()
# 	print "Total Keypoints with nonmaxSuppression: ", len(kp)
# 	cv2.imwrite('fast_true.png',img2)
# 	# Disable nonmaxSuppression
# 	fast.setNonmaxSuppression(0)
# 	kp2 = fast.detect(img,None)[:500]
# 	print "Total Keypoints without nonmaxSuppression: ", len(kp)
# 	img3 = cv2.drawKeypoints(img, kp2, color=(255,0,0), outImage=True)
# 	cv2.imwrite('fast_false.png',img3)
# 	return kp, img, img2, img3
