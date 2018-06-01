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

    counter1 = 0
    counter2 = 0	
    
    for filename in sortedlist:
	warnings.filterwarnings("ignore", category=DeprecationWarning) 
        test_image = imread(directory+'/'+filename, 1)
	hog_features = hog(test_image, orientations=6, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
   	result_type = clf.predict(hog_features)
	
	for i in range(len(char)):	
	    if result_type == i:
		if char[i] == '1':
		    print("shcherbinin")
		    counter1 = counter1 + 1
		elif char[i] == '2':
		    print("fivos")	
		    counter2 = counter2 + 1

    if counter1 > counter2:
	print("Document written by Mr.Shcherbinin")
    else:
	print("Document written by Mr.Ntelemis")	
    	
	
       

            
