import numpy as np
import matplotlib.pyplot as plt
import dicom
import os
import pandas as pd
import cv2
import math

data_dir = 'train_data_dir'
patients = os.listdir(data_dir)

IMG_SIZE_PX = 150
SLICE_COUNT = 9


def chunks( l,n ):
    #n-sized chunks from list l
    count=0
    for i in range(0, len(l), n):
        if(count < SLICE_COUNT):
            yield l[i:i + n]
            count=count+1


def mean(a):
    return sum(a) / len(a)



for patient in patients[:1]:
    path = os.path.join( data_dir, patient )
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []
    slices = [cv2.resize( np.array(each_slice.pixel_array), (IMG_SIZE_PX,IMG_SIZE_PX))   for each_slice in slices]
    
    chunk_sizes = math.floor(len(slices) / SLICE_COUNT)

    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    fig = plt.figure()
    for num,each_slice in enumerate(new_slices):
        y = fig.add_subplot(3,3,num+1)
        y.imshow(each_slice)
    plt.show()
