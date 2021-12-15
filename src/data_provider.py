from numpy.core.shape_base import block
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from random import randint

from tensorflow.python.data.ops.dataset_ops import BatchDataset
from enum import Enum

from src.const import *


class DataType(Enum):
    ALL = 1
    PUG = 2
    NOISE = 3
    
class DataProvider:
    data_type:DataType
    dataset:tf.data.Dataset
    show:bool
    
    def __init__(self, data:DataType, show = True, cut = 1):
        self.data_type = data
        self.show = show
        self.cut = cut
        
    def load_data(self):
        data:np.array
        
        if(self.data_type is DataType.ALL):
            data = self._load_all_data()
        elif(self.data_type is DataType.PUG):
            data = self._load_pug_data()
        else:
            data = self._load_noise_data()
        
        self._show_example_data(data)
        
        train_images = (data - 127.5)/127.5
        self.dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE//self.cut).batch(BATCH_SIZE//self.cut)
        
        return self.dataset
    
    def _show_example_data(self,data):
        SIZE = 10
        fig = plt.figure(figsize=(SIZE, SIZE))

        for i in range(SIZE*SIZE):
            plt.subplot(SIZE, SIZE, i+1)
            plt.imshow(data[randint(0,data.shape[0]-1), :, :, :]/255)
            plt.axis('off')

        plt.savefig('./examples.png')
        plt.show(block=WAIT)
            
    def _load_all_data(self):
        all_directory = os.listdir(ALL_DIR)
        directory_files  = ['']*len(all_directory)
        directory_folder = ['']*len(all_directory)
        size = 0
        
        for iterator, dog_folder in enumerate(all_directory):
            if(len(dog_folder.split("."))>1): continue
            directory_files[iterator] = os.listdir(ALL_DIR+dog_folder)
            directory_folder[iterator] = dog_folder
            size += len(directory_files[iterator])
            
        size = size//self.cut
        
        output = np.zeros((size,IMAGE_SIZE,IMAGE_SIZE,3),dtype=np.float32)

        print("Loading started...")
        folders = len(directory_files)
        iterator = 0
        for iterator_f,files in enumerate(directory_files):
            print("\rLoading "+str(iterator_f+1)+"/"+str(folders)+"...",end="")
            for iterator_p, pug in enumerate(files):
                try:
                    img = PIL.Image.open(ALL_DIR+directory_folder[iterator_f]+"/"+pug)
                    scaled_img = img.resize((IMAGE_SIZE,IMAGE_SIZE),PIL.Image.ANTIALIAS)
                    output[iterator,:,:,:] = np.array(scaled_img,dtype=np.float32)
                    iterator += 1
                    if iterator == size: break
                except:
                    print("File error within the following directory: "+directory_folder[iterator_f])
                    print("File causing problems: "+pug)
                    print("Please remove this file before training next time")
            if iterator == size: break
            
            
        print("\n"+str(folders)+" folders containing "+str(size)+" files loaded!")
        return output
            
    def _load_noise_data(self):
        print("Loading started...")
        output = tf.random.normal([BUFFER_SIZE, 128, 128, 3])
        print(str(BUFFER_SIZE)+" noise images loaded!")
        return output
        
        
    def _load_pug_data(self):
        directory = os.listdir(PUG_DIR)
        size = len(directory)
        output = np.zeros((size,IMAGE_SIZE,IMAGE_SIZE,3),dtype=np.float32)

        print("Loading started...")
        for iterator, pug in enumerate(directory):
            img = PIL.Image.open(PUG_DIR+pug)
            scaled_img = img.resize((IMAGE_SIZE,IMAGE_SIZE),PIL.Image.ANTIALIAS)
            output[iterator,:,:,:] = np.array(scaled_img,dtype=np.float32)
        print(str(size)+" pug images loaded!")
        
        return output