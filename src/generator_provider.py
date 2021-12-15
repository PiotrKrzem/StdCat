from re import M
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from enum import Enum

from src.const import *

class GeneratorType(Enum):
    MOCK = 1
    SMALL = 2
    MEDIUM = 3
    BIG = 4
    HUGE = 5

class GeneratorProvider:
    
    generator:tf.keras.Sequential
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    def __init__(self,type:GeneratorType):
        self._initialize_generator(type)
        self._test_generator()
        
    def loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def get(self):
        return self.generator
    
    def apply_gradients(self,gradients_of_generator):
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
    def _initialize_generator(self, type:GeneratorType):
        if(type is GeneratorType.MOCK):
            self.generator = self._make_generator_model_mock()
        elif (type is GeneratorType.SMALL):
            self.generator = self._make_generator_model_small()
        elif (type is GeneratorType.MEDIUM):
            self.generator = self._make_generator_model_medium()
        elif (type is GeneratorType.BIG):
            self.generator = self._make_generator_model_big()
        else:
            self.generator = self._make_generator_model_huge()
        self.generator.summary()
        
    def _test_generator(self):
        noise = tf.random.normal([1, IMAGE_SIZE, IMAGE_SIZE, 3])
        generated_image = self.generator(noise, training=False)
        plt.imshow(generated_image[0, :, :, :])
        
    def _make_generator_model_mock(self):
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(128,128,3)))
        model.add(layers.Activation(tf.nn.tanh))
        model.add(layers.Reshape((128,128,3)))

        assert model.output_shape == (None, 128, 128, 3)

        return model

    def _make_generator_model_small(self):
        model = tf.keras.Sequential()

        model.add(layers.Reshape((IMAGE_SIZE,IMAGE_SIZE,3,1), input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
        
        model.add(layers.Conv3D(8, (7, 7, 1), strides=(2, 2, 1), padding='valid', use_bias=False))
        model.add(layers.Conv3D(8, (5, 5, 1), strides=(2, 2, 1), padding='valid', use_bias=False))
        model.add(layers.Conv3D(8, (3, 3, 1), strides=(1, 1, 1), padding='valid', use_bias=False))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(1024*3,activation='tanh'))
        model.add(layers.Dense(1024*3,activation='tanh'))
        model.add(layers.Dense(1024*3,activation='tanh'))
        
        model.add(layers.Reshape((32,32,3,1)))
        model.add(layers.Conv3D(32, (5, 5, 1), strides=(2, 2, 1), padding='valid', use_bias=False))
        model.add(layers.Reshape((14,14,3*32,1)))
        model.add(layers.Conv3D(32, (3, 3, 1), strides=(2, 2, 1), padding='valid', use_bias=False))
        model.add(layers.Flatten())
        
        model.add(layers.Dense(1024*3,activation='tanh'))   
        model.add(layers.Dense(1024*3,activation='tanh'))        
        model.add(layers.Dense(1024*3*4,activation='tanh'))             
        model.add(layers.Dense(1024*3*4,activation='tanh'))
                
        #model.add(layers.Dense(49152,activation='tanh'))

        #model.add(layers.Reshape((128,128,3)))
        #model.add(layers.AveragePooling2D())
        
        model.add(layers.Reshape((IMAGE_SIZE,IMAGE_SIZE,3)))
        
        assert model.output_shape == (None, IMAGE_SIZE, IMAGE_SIZE, 3)
        
        return model
    
    def _make_generator_model_medium(self):
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(128,128,3)))
        model.add(layers.Reshape((128,128,3,1)))

        model.add(layers.Conv3D(64, (17, 17, 1), strides=(4, 4, 1), padding='valid', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        #28,28,3,64
        model.add(layers.Reshape((28,28,64*3,1)))
        
        model.add(layers.Conv3D(64, (5, 5, 4), strides=(6, 6, 4), padding='valid', use_bias=False))
        #4,4,48,64
        model.add(layers.Activation(tf.nn.tanh))
        model.add(layers.Reshape((128,128,3)))
        
        assert model.output_shape == (None, 128, 128, 3)
        
        return model

    def _make_generator_model_big(self):
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(128,128,3)))
        model.add(layers.Reshape((128,128,3,1)))

        model.add(layers.Conv3D(128, (15, 15, 1), padding='valid', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        #114,114,3,128
        model.add(layers.Reshape((114,114,3*128,1)))
        model.add(layers.MaxPool3D(pool_size=(2,2,4),strides=(2,2,4), padding='valid'))
        
        
        model.add(layers.Conv3D(64, (11, 11, 1), padding='valid', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        #47,47,96,64
        model.add(layers.Reshape((47,47,96*64,1)))
        model.add(layers.MaxPool3D(pool_size=(2,2,4),strides=(2,2,4), padding='valid'))

        model.add(layers.Conv3D(32,  (7, 7, 1), padding='valid', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        #17,17,1536,32
        model.add(layers.Reshape((17,17,1536*32,1)))
        model.add(layers.MaxPool3D(pool_size=(2,2,4),strides=(2,2,4), padding='valid'))
        
        
        model.add(layers.Conv3D(16, (5, 5, 1), padding='valid', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        #4,4,12288,16
        model.add(layers.Reshape((4,4,12288*16,1)))
        model.add(layers.MaxPool3D(pool_size=(4,4,4),strides=(4,4,4), padding='valid'))
        
        model.add(layers.Activation(tf.nn.tanh))
        model.add(layers.Reshape((128,128,3)))
        
        
        assert model.output_shape == (None, 128, 128, 3)
        
        return model
    
    def _make_generator_model_huge(self):
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(128,128,3)))
        model.add(layers.Reshape((128,128,3,1)))

        model.add(layers.Conv3D(64, (9, 9, 3), strides=(2, 2, 1), padding='valid', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        #60,60,1,64
        model.add(layers.Reshape((60,60,64,1)))

        model.add(layers.Conv3D(64, (9, 9, 4), strides=(2, 2, 4), padding='valid', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        #26,26,16,64
        model.add(layers.Reshape((26,26,16*64,1)))

        model.add(layers.Conv3D(32, (7, 7, 16), strides=(1, 1, 16), padding='valid', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        #20,20,64,32
        model.add(layers.Reshape((20,20,64*32,1)))

        model.add(layers.Conv3D(16, (3, 3, 32), strides=(1, 1, 32), padding='valid', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        #18,18,64,16
        model.add(layers.Reshape((18,18,64*16,1)))

        model.add(layers.Conv3D(3, (3, 3, 4), strides=(2, 2, 4), padding='valid', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(tf.nn.tanh))
        #8,8,256,3
        model.add(layers.Reshape((128,128,3)))

        assert model.output_shape == (None, 128, 128, 3)

        return model