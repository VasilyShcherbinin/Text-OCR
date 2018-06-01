from __future__ import print_function
import numpy as np 
import warnings
import os
from skimage.feature import hog
from scipy.misc import imread,imresize
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pickle


if __name__ == '__main__':
    #load the detector
    clf = pickle.load( open("letter.detector","rb"))

    char_small = ''
    char_cap = ''
    char_num = '12'
    char = char_small + char_cap + char_num

    directory = "testingChar"
    sortedlist = sorted(os.listdir(directory))

    for filename in sortedlist:
	warnings.filterwarnings("ignore", category=DeprecationWarning) 
        test_image = imread(directory+'/'+filename, 1)
	hog_features = hog(test_image, orientations=12, pixels_per_cell=(5, 5), cells_per_block=(1, 1))
   	result_type = clf.predict(hog_features)
	for i in range(len(char)):	
	    if result_type == i:
		if char[i] == '1':
		    print ("shcherbinin1")
		elif char[i] == '2':
		    print ("shcherbinin2")	
    	
#	if result_type == 0:
#	   print (char[0], end='')
#	else:
#	   print (result_type, end='')

#    #now load a test image and get the hog features. 
#    test_image = imread('testingChar/3296_snippet.png',1)
#    test_image = imresize(test_image, (20,20))

#    hog_features = hog(test_image, orientations=8, pixels_per_cell=(2, 2),
#                    cells_per_block=(1, 1))

#    result_type = clf.predict(hog_features)
#    print result_type
       

            
