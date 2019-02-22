import os 
import sys
import numpy as np
import cv2
def get_nss(path):

	im = cv2.imread(path, 0) # read as gray scale
	blurred = cv2.GaussianBlur(im, (7, 7), 1.166) # apply gaussian blur to the image
	blurred_sq = blurred * blurred 
	sigma = cv2.GaussianBlur(im * im, (7, 7), 1.166) 
	sigma = (sigma - blurred_sq) ** 0.5
	sigma = sigma + 1.0/255 # to make sure the denominator doesn't give DivideByZero Exception
	structdis = (im - blurred)/sigma # final
	return {
			'image':im,
			'blurred':blurred,
			'blurred_sq':blurred_sq,
			'sigma':sigma,
			'structdis':structdis
		}

def pairwiseproducts(structdis):
	# indices to calculate pair-wise products (H, V, D1, D2)
	shifts = [[0,1], [1,0], [1,1], [-1,1]]
	
	# calculate pairwise components in each orientation
	products = []
	for itr_shift in range(1, len(shifts) + 1):

		ShiftArr = np.zeros(np.shape(structdis))
		OrigArr = structdis
		reqshift = shifts[itr_shift-1] # shifting index

		for i in range(structdis.shape[0]):
			for j in range(structdis.shape[1]):
				if(i + reqshift[0] >= 0 and i + reqshift[0] < structdis.shape[0] and j + reqshift[1] >= 0 and j  + reqshift[1] < structdis.shape[1]):
					ShiftArr[i, j] = OrigArr[i + reqshift[0], j + reqshift[1]]
				else:
					ShiftArr[i, j] = 0
		products.append(ShiftArr)
	return 	products