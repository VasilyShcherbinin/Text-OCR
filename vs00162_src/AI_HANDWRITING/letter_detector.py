from __future__ import print_function
from __future__ import division
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

    char_small = 'abcdefghijklmnopqrstuvwxyz'
    char_cap = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    char_num = '123456789*#'
    char = char_small + char_cap + char_num

    directory = "testingChar"
    sortedlist = sorted(os.listdir(directory))
    templist = list()

    for filename in sortedlist:
	warnings.filterwarnings("ignore", category=DeprecationWarning) 
        test_image = imread(directory+'/'+filename, 1)
	hog_features = hog(test_image, orientations=6, pixels_per_cell=(5, 5), cells_per_block=(1, 1))
   	result_type = clf.predict(hog_features)
	for i in range(len(char)):	
	    if result_type == i:
		if char[i] == '#':
		    print ("")
		elif char[i] == '*':
		    print (" ", end='')
		else:
		    print(char[i], end='')
		    templist.append(char[i])

    resultlist = (''.join(templist))
    correct = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"

    counter = 0

    for x, y in zip (resultlist, correct):
	if x == y:
	   counter = counter + 1
    
    accuracy = counter / len(templist)
    errors = len(templist)-counter

    print()
    print("{0:.1%}".format(accuracy))
    print("Errors: "+str(errors))    

    print(resultlist)
    print(correct)


            
