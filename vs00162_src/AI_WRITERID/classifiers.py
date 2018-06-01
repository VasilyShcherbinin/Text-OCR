import numpy as np 
import time
import os
import itertools
import operator
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.feature import hog
from skimage import color, exposure
from scipy.misc import imread,imsave,imresize
import numpy.random as nprnd
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.svm import LinearSVC
import matplotlib
import pickle

if __name__ == '__main__':
    #paths for the training samples 
    
    folder_string_small = ''
    folder_string_cap = ''
    folder_string_num = '12'
    folder_string = folder_string_small + folder_string_cap + folder_string_num

    count = -1
    data = []
    labels = []

    for char in folder_string:
	tpath = './trainingChar/'+char+'/'
	locals()['path_%s' % char]= tpath
	locals()['filenames_%s' % char] = sorted([filename for filename in os.listdir(tpath) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp')))])
	locals()['filenames_%s' % char] = sorted([locals()["path_"+char]+filename for filename in locals()["filenames_"+char]])
	print 'Num of training images -> ' + char + ': ' + str(len(locals()["filenames_"+char]))
	count = count + 1
	for filename in vars()["filenames_"+char]:
	    image = imread(filename,1)
	    hog_features = hog(image, orientations=6, pixels_per_cell=(4,4), cells_per_block=(1, 1))
            data.append(hog_features)
            labels.append(count)
	
    start_time1 = time.time()
    print 'Training the SVM'
    #create the SVC
    clf = LinearSVC(dual=False,verbose=1)
    #train the svm
    clf.fit(data, labels)

    #pickle it - save it to a file
    pickle.dump( clf, open( "letter.detector", "wb" ) )
    print time.time() - start_time1, "seconds" 
