import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.misc import imread,imresize,imsave
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import os
from array import array
from PIL import Image

class Extract_Letters:
    def extractFile(self, filename):

        image = imread(filename,1)
    
        #apply threshold in order to make the image binary
        bw = image < threshold_otsu(image)

        cleared = bw.copy()

        # label image regions
        label_image = label(cleared,neighbors=8)
        borders = np.logical_xor(bw, cleared)
        label_image[borders] = -1

        letters = list()
        order = list()
    
        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            # skip small images
            if maxr - minr > len(image)/250: # better to use height rather than area.
                rect = mpatches.Rectangle((minr, minc), maxr - minc, maxc - minc,
                                      fill=False, edgecolor='red', linewidth=2)
                order.append(region.bbox)


        #sort the detected characters left->right, top->bottom
        lines = list()
        first_in_line = ''
        counter = 0

        #worst case scenario there can be 1 character per line
        for x in range(len(order)):
            lines.append([])
    
        for character in order:
            if first_in_line == '':
                first_in_line = character
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) < (first_in_line[2] - first_in_line[0]):
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) > (first_in_line[2] - first_in_line[0]):
                first_in_line = character
                counter += 1
                lines[counter].append(character)


        for x in range(len(lines)):       
            lines[x].sort(key=lambda tup: tup[1])

        final = list()
        prev_maxc = 0
        prev_line_maxr = 0
	prev_line_minr = 0
        for i in range(len(lines)):
            for j in range(len(lines[i])):
		minr = lines[i][j][0]
		maxr = lines[i][j][2]
                minc = lines[i][j][1]
		maxc = lines[i][j][3]

		
                if minc > prev_maxc and minr > prev_line_maxr:
                    letter_raw = bw[minr-10:maxr+5,minc-2:maxc+2]
                    letter_norm = imresize(letter_raw ,(20 ,20))
                    final.append(letter_norm)
                    prev_maxc = lines[i][j][3]
	

                if j == (len(lines[i])-1):
                    prev_line_maxr = lines[i][j][2]
		    prev_line_minr = lines[i][j][0]
            prev_maxc = 0
            minc = 0
        return final

    def extractTest(self, filename):

        image = imread(filename,1)
    
        #apply threshold in order to make the image binary
        bw = image < threshold_otsu(image)
    
        cleared = bw.copy()

        # label image regions
        label_image = label(cleared,neighbors=8)
        borders = np.logical_xor(bw, cleared)
        label_image[borders] = -1

        letters = list()
        order = list()
    
        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            # skip small images
            if maxr - minr > len(image)/150: # better to use height rather than area.
                rect = mpatches.Rectangle((minr, minc), maxr - minc, maxc - minc,
                                      fill=False, edgecolor='red', linewidth=2)
                order.append(region.bbox)


        #sort the detected characters left->right, top->bottom
        lines = list()
        first_in_line = ''
        counter = 0

        #worst case scenario there can be 1 character per line
        for x in range(len(order)):
            lines.append([])
    
        for character in order:
            if first_in_line == '':
                first_in_line = character
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) < (first_in_line[2] - first_in_line[0]):
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) > (first_in_line[2] - first_in_line[0]):
                first_in_line = character
                counter += 1
                lines[counter].append(character)


        for x in range(len(lines)):       
            lines[x].sort(key=lambda tup: tup[1])

        final = list()
        prev_maxc = 0
        prev_line_maxr = 0
	prev_line_minr = 0
        print lines[0]
        for i in range(len(lines)):
            for j in range(len(lines[i])):
		minr = lines[i][j][0]
		maxr = lines[i][j][2]
                minc = lines[i][j][1]
		maxc = lines[i][j][3]
		difference = abs(minc - prev_maxc)
		print difference
                if minc > prev_maxc and minr > prev_line_maxr:
                    letter_raw = bw[minr-10:maxr+5,minc-2:maxc+2]
                    letter_norm = imresize(letter_raw ,(20 ,20))
                    final.append(letter_norm)
                    prev_maxc = lines[i][j][3]
		
#		if prev_maxr > minr and minc < prev_line_br:
			
                if j == (len(lines[i])-1):
                    prev_line_maxr = lines[i][j][2]
		    prev_line_minr = lines[i][j][0]
            prev_maxc = 0
            minc = 0
        return final


    def __init__(self):
        print "Extracting characters..."

start_time1 = time.time()
start_time2 = time.time()
extract = Extract_Letters()

path_training_files_1 = './ocr/training/1/'
training_filenames_1 = sorted([filename for filename in os.listdir(path_training_files_1) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
training_filenames_1 = [path_training_files_1+filename for filename in training_filenames_1]

path_training_files_2 = './ocr/training/2/'
training_filenames_2 = sorted([filename for filename in os.listdir(path_training_files_2) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
training_filenames_2 = [path_training_files_2+filename for filename in training_filenames_2]

path_testing_files = './ocr/testing/'
testing_files = sorted([filename for filename in os.listdir(path_testing_files) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
testing_files = [path_testing_files+filename for filename in testing_files]

folder_string_small = ''
folder_string_cap = ''
folder_string_num = '12'
folder_string = folder_string_small + folder_string_cap + folder_string_num

for char in folder_string:
	newpath1 = ((r'trainingChar/%s') % (char)) 
    	if not os.path.exists(newpath1): os.makedirs(newpath1)

name_counter = 1

for files in training_filenames_1:
	letters = extract.extractFile(files)
	string_counter = 0
	
	for i in letters:
		if string_counter > 60:
			string_counter = 0
		imsave('./trainingChar/1/' + str(name_counter) + '_snippet.png', i)
		
		string_counter += 1
		name_counter += 1

for files in training_filenames_2:
	letters = extract.extractFile(files)
	string_counter = 0
	
	for i in letters:
		if string_counter > 60:
			string_counter = 0
		imsave('./trainingChar/2/' + str(name_counter) + '_snippet.png', i)
		
		string_counter += 1
		name_counter += 1

print time.time() - start_time1, "seconds" 

for files in testing_files:
	letters = extract.extractTest(testing_files[0])
	name_counter = 1000
	print testing_files[0]	

	for i in letters:
		path = './testingChar/' + str(name_counter) + '_snippet.png'
		imsave(path, i)

		name_counter += 1
		
