# -*- coding: utf-8 -*-
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

    test_directory = "testingChar"
    sortedlist = sorted(os.listdir(test_directory))
    templist = list()

    for filename in sortedlist:
	warnings.filterwarnings("ignore", category=DeprecationWarning) 
        test_image = imread(test_directory+'/'+filename, 1)
	hog_features = hog(test_image, orientations=8, pixels_per_cell=(6, 6), cells_per_block=(1, 1))
   	result_type = clf.predict(hog_features)
	for i in range(len(char)):	
	    if result_type == i:
		if char[i] == '#':
		    print ("")
		elif char[i] == '*':
		    print (" ", end='')
#		elif char[i] == 'C':
#		    print ("c", end='')
#		    templist.append('c')
#		elif char[i] == 'I':
#		    print ("l", end='')
		else:
		    print(char[i], end='')
#		    templist.append(char[i].lower())
		    templist.append(char[i])
		    
    path_testing_files = './ocr/testing/'
    testing_filenames = sorted([filename for filename in os.listdir(path_testing_files) if (filename.endswith('.txt')) ])
    testing_filenames = [path_testing_files+filename for filename in testing_filenames]   

    with open(testing_filenames[1], 'r') as myfile:
	data = myfile.read().replace('\n', '')
#	data = data.replace('0','O')
	data = data.replace(' ', '')
	data = data.replace('.', '')
	data = data.replace('‘', '')
	data = data.replace(',', '')
	data = data.replace('!', '')
	data = data.replace('?', '')
	data = data.replace('’', '')
	data = data.replace('“', '')
	data = data.replace('”', '')
	data = data.replace('-', '')
	data = data.replace('(', '')
	data = data.replace(')', '')
#	data = data.lower()
	
    resultlist = (''.join(templist))
    counter = 0

    for x, y in zip (resultlist, data):
	if x == y:
	   counter = counter + 1
    
    accuracy = counter / len(templist)
    errors = len(templist)-counter
    print()
    print("{0:.1%}".format(accuracy))
    print("Errors: "+str(errors))    

    print(resultlist)
    print(data)
