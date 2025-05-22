# -----------------------------------------------------------------------------
# License-Identifier: GPL-3.0-only
# This file is part of the TT-sandbox project.
# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# Contributor: Teng Xue <teng.xue@idiap.ch>
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os,sys
from sklearn.model_selection import train_test_split

import logging
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from time import time
from sklearn.pipeline import Pipeline

from PIL import Image
import numpy as np
import glob

data_load = 'SVD'

"""
TT-SVD data preparation
"""
# Min's Link 
path = sys.path[0]+'/CroppedYalePNG'
factor = 1
def read_images(path, sz=(168//factor,192//factor),occlusion = False): 
    t0 = time() # time reading images
    c=0 # count of read images
    
    # y to store subject number of images read
    y = [] 
    # Images are resized to 640x480
    # x to store all image data
    x = np.empty([1, sz[0], sz[1]]) 

    mat_y = [] 
    # Images are resized to 640x480
    # x to store all image data
    mat_x = np.empty([1, sz[0]* sz[1]]) 
    num_categories = 15
    for dirname , dirnames , filenames in os.walk(path):
        # loop all directories, sub-dir, files to read images
        # in data folder of this project, only 1st dirnames is valid, others are empty list (no subfolder)
        # for subdirname in dirnames:
        subdirname = dirname
        subject_path = os.path.join(dirname , subdirname) 
        for filename in sorted(os.listdir(subject_path)):
            # read files start with 'yale'
            # eclude windows hidden files of 'desktop.ini'
            if filename[:4] != 'yale':
                continue
            
            # for files start with 'yale'
            # 1. read image
            # 2. convert images from RGB to L(color to black/white)
            # In this project, all images are greyscale images, run this step in case of exceptions
            # 3. resize image to 168 X 192
            # In this project, all images are 168 X 192, run this step in case of exceptions
            # 4. read images to matrix (ndarray)
            # 5. normalize image matrix
            # 6. flatten image ndarray (2-d to 1-d)
            # 7. store image ndarray to data (x)
            try:
                im = Image.open(os.path.join(subject_path , filename)) 
                # im = im.convert("L")
                # resize to given size (if given)
                if (sz is not None):
                    im = im.resize(sz, Image.ANTIALIAS)
                im  = np.asarray(im, dtype=np.uint8)
                im = im.transpose(1,0)
                # im.flags.writeable = True # set ndarray as writeable. not read-only
                
                # Normalization
                im = (im - im.mean())/im.std()
                im = im / np.amax(np.abs(im))

                # Set occlusion area, not useful in this project. 
                # Just for fun and possible applications in extending projects
                if occlusion:
                    im[30:60,10:150]=im.min()
                    im[70:100,60:100]=im.min()
                
                # # Convert 2-d ndarray images to 1-d array by rows through ndarray.ravel()
                # # append image array to x
                # # first element is empty array
                # x = np.vstack((x,im.ravel()))

                x = np.concatenate((x,im[None, :, :]), axis=0)

                # append image subject number to label data: y
                y.append(filename[5:7])

                mat_x = np.vstack((mat_x, im.ravel()))

                # y.append(c)
                # append image subject number to label data: y
                mat_y.append(filename[5:7])

            except IOError:
                #print "I/O error({0}): {1}".format("errno", "strerror") #there is some text files in the path
                pass
            except:
                print ("Unexpected error:", sys.exc_info()[0] )
                raise
            c = c+1

            if c >=65*num_categories:
                break
        
    y = np.asarray(y)
    x = x[1:] # Skip the first line which is void
    mat_y = np.asarray(mat_y)
    mat_x = mat_x[1:] # Skip the first line which is void

    print("Image read in %0.3fs" % (time() - t0))
    return [x, y, mat_x, mat_y]
    ### <center> train test split<center>

x,y, mat_x, mat_y = read_images(path=path,occlusion=False) #x: image, y:label

import pickle

# with open('facex_TT.pickle', 'wb') as handle:
#     pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('facey_TT.pickle', 'wb') as handle:
#     pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('facex.pickle', 'wb') as handle:
    pickle.dump(mat_x, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('facey.pickle', 'wb') as handle:
    pickle.dump(mat_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("data loaded")


