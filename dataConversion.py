from scipy.misc import imread
import scipy
import numpy as np
import pandas as pd
import os

root = './gestures'

#go through each directory in root folder given above
for directory, subdirectory, files in os.walk(root):
    #read the .jpg image files and extract its pixel
    for file in files:
        im = imread(os.path.join(directory, file))
        value = im.flatten()
        value = np.hstack((directory[8:], value))
        df = pd.DataFrame(value).T
        df = df.sample(frac = 1)
        with open('train_set.csv', 'a') as dataset:
            df.to_csv(dataset,header=False,index = False)


