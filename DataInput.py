import numpy as np
import pandas as pd
import dicom
import os
import matplotlib.pyplot as plt
import cv2
import math

IMG_SIZE_PX = 50
SLICE_COUNT = 20


def chunks( l,n ):
    count=0
    for i in range(0, len(l), n):
        if(count < SLICE_COUNT):
            yield l[i:i + n]
            count=count+1

def mean(a):
    return sum(a) / len(a)


def process_data( patient, labels_df ):
    
    label = labels_df.get_value(patient, 'cancer')
    path = os.path.join( data_dir, patient )
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []
    slices = [cv2.resize( np.array(each_slice.pixel_array), (IMG_SIZE_PX,IMG_SIZE_PX))   for each_slice in slices]
    
    chunk_sizes = math.floor(len(slices) / SLICE_COUNT)

    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if label == 1: label=np.array([0,1])
    elif label == 0: label=np.array([1,0])
        
    return np.array(new_slices),label






data_dir = 'train_data_dir'

patients = os.listdir(data_dir)
labels_df = pd.read_csv('stage1_labels.csv', index_col=0)


lung_data = []
for num,patient in enumerate(patients):
    try:
        img_data, label = process_data( patient, labels_df )
        lung_data.append( [img_data, label] )
    except KeyError as e:
        print('This is unlabeled data!')

np.save('lungdata-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), lung_data)
