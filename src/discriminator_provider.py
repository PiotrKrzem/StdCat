from enum import Enum

import tensorflow as tf
from src.const import *
from tensorflow.keras import layers

class DiscriminatorType(Enum):
    MOCK = 1
    SMALL = 2
    MEDIUM = 3
    BIG = 4
    HUGE = 5

class DiscriminatorProvider:
    
    discriminator:tf.keras.Sequential
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    def __init__(self,type:DiscriminatorType):
        self._initialize_discriminator(type)
        self._test_discriminator()
    
    def loss(self,real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def get(self):
        return self.discriminator
        
    def apply_gradients(self,gradients_of_discriminator):
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
    def _initialize_discriminator(self,type:DiscriminatorType):
        if(type is DiscriminatorType.MOCK):
            self.discriminator = self._make_discriminator_model_mock()
        elif(type is DiscriminatorType.SMALL):
            self.discriminator = self._make_discriminator_model_small()
        elif(type is DiscriminatorType.MEDIUM):
            self.discriminator = self._make_discriminator_model_medium()
        elif(type is DiscriminatorType.BIG):
            self.discriminator = self._make_discriminator_model_big()
        else:
            self.discriminator = self._make_discriminator_model_huge()
            
        self.discriminator.summary()
        
    def _test_discriminator(self):
        noise = tf.random.normal([1, IMAGE_SIZE, IMAGE_SIZE, 3])
        decision = self.discriminator(noise, training=False)
        print(decision)
    
    def _make_discriminator_model_mock(self):
        model = tf.keras.Sequential()

        model.add(layers.Reshape((IMAGE_SIZE,IMAGE_SIZE,3,1), input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
        model.add(layers.MaxPool3D((128,128,3),strides=(1,1,1), padding='valid'))
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        
        return model
    
    def _make_discriminator_model_small(self):
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
        model.add(layers.Dense(1))
        return model

    def _make_discriminator_model_medium(self):
        model = tf.keras.Sequential()
        
        model.add(layers.Reshape((IMAGE_SIZE,IMAGE_SIZE,3,1), input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
        model.add(layers.Conv3D(64, kernel_size=(3,3,1), strides=(2,2,1), use_bias=False))        
        model.add(layers.Conv3D(64, kernel_size=(3,3,1), strides=(2,2,1), use_bias=False))        
        model.add(layers.Conv3D(64, kernel_size=(3,3,1), strides=(2,2,1), use_bias=False))            
    
        model.add(layers.Flatten())
        
        model.add(layers.Dense(4096))
        model.add(layers.Dense(4096))
        model.add(layers.Dense(2048))
        model.add(layers.Dense(2048))
        model.add(layers.Dense(1024))
        model.add(layers.Dense(1024))
        
        model.add(layers.Dense(1))
        
        return model
    
    def _make_discriminator_model_big(self):
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(128,128,3)))
        model.add(layers.Reshape((128,128,3,1)))

        model.add(layers.Conv3D(16, (9, 9, 1), strides=(2, 2, 1), padding='valid'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        #60,60,3,16
        model.add(layers.Reshape((60,60,3*16,1)))
        
        model.add(layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),padding='valid'))
        #30,30,24,1
        model.add(layers.Conv3D(8, (7, 7, 7), strides=(3, 3, 3), padding='valid'))
        #8,8,6,8
        model.add(layers.Reshape((8,8,6*8,1)))
        
        model.add(layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2),padding='valid'))

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='tanh'))

        return model
    
    def _make_discriminator_model_huge(self):
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(128,128,3)))
        model.add(layers.Reshape((128,128,3,1)))

        model.add(layers.Conv3D(32, (5, 5, 1), strides=(2, 2, 1), padding='valid'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv3D(32, (5, 5, 1), strides=(2, 2, 1), padding='valid'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model