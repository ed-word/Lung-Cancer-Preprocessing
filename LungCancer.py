import dicom
import os
import pandas as pd

data_dir = 'C:/Eddy/Workspace/Machine Learning/Lung Cancer/sample_images'

patients = os.listdir(data_dir)
labels_df = pd.read_csv('C:/Eddy/Workspace/Machine Learning/Lung Cancer/stage1_labels.csv', index_col=0)
#print( labels_df.head() )

for patient in patients[:1]:
    label = labels_df.get_value( patient, 'cancer' )
    path = os.path.join( data_dir, patient )
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    #print(len(slices),label)
    #print(slices[0])

#print(len(patients))




import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

IMG_PX_SIZE = 150
HM_SLICES = 20                      #How Many slices



def chunks( l,n ):
    #n-sized chunks from list l
    count=0
    for i in range(0, len(l), n):
        if(count < HM_SLICES):
            yield l[i:i + n]
            count=count+1

def mean(l):
    return sum(l)/len(l)


for patient in patients[:1]:
    label = labels_df.get_value( patient, 'cancer' )
    path = os.path.join( data_dir, patient )
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    fig = plt.figure()
    for num,each_slice in enumerate( slices[:9] ):
        y = fig.add_subplot( 3,3,num+1 )
        new_img = cv2.resize( np.array(each_slice.pixel_array), (IMG_PX_SIZE,IMG_PX_SIZE) )
        y.imshow(new_img)

    plt.show()



def process_data( patient, labels_df):
    for patient in patients[:1]:
        try:
            label = labels_df.get_value( patient, 'cancer' )
            path = os.path.join( data_dir, patient )
            slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
            slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

            new_slices = []
            slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

            chunk_sizes = math.floor(len(slices) / HM_SLICES)

            for slice_chunk in chunks(slices, chunk_sizes):
                slice_chunk = list( map( mean, zip(*slice_chunk ) ))
                new_slices.append(slice_chunk)

            #print(len(slices), len(new_slices))

            fig = plt.figure()
            for num,each_slice in enumerate( new_slices[:9] ):
                y = fig.add_subplot( 3,3,num+1 )
                y.imshow(each_slice)

            plt.show()
        except:
            # some patients don't have labels, so we'll just pass on this for now
            pass

















