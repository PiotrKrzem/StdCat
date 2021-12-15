import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.ops.gen_array_ops import Reshape

DIR = "./images/Images/n02110958-pug"

def load_data():
    directory = os.listdir(DIR)
    output = np.zeros((len(directory),128,128,3),dtype=np.float32)
    iterator = 0
    for pug in directory:
      img = PIL.Image.open(DIR+"/"+pug)
      scaled_img = img.resize((128,128),PIL.Image.ANTIALIAS)
      output[iterator,:,:,:] = np.array(scaled_img,dtype=np.float32)
      iterator+=1
    return output
  
train_images = load_data()
train_image_plot = plt.figure()
#train_image_plot.set_title("train image")
train_image_plot.figimage(train_images[0, :, :, :])
#plt.imshow(train_images[0, :, :, :])
plt.show()
train_images = (train_images - 127.5)/127.5



BUFFER_SIZE = 256
BATCH_SIZE = 16

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_mock_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(128,128,3)))
    model.add(layers.Activation(tf.nn.tanh))
    model.add(layers.Reshape((128,128,3)))

    assert model.output_shape == (None, 128, 128, 3)

    return model

def make_generator_model():
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

generator = make_generator_model()
generator.summary()

noise = tf.random.normal([1, 128, 128, 3])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, :])

def make_mock_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(128,128,3)))
    model.add(layers.Dense(1))
    return model

def make_discriminator_model():
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

discriminator = make_discriminator_model()
discriminator.summary()

decision = discriminator(generated_image)
print(decision)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, 128*128*3])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 128, 128, 3])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow((predictions[i, :, :, :]+1)/2)
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

train(train_dataset, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)