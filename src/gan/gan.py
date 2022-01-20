import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import PIL
import time

from re import I
from IPython import display
from tensorflow.python.data.ops.dataset_ops import BatchDataset

from src.generator.generator import *
from src.discriminator.discriminator import *
from src.utils.data_provider import *
from src.const import *


class Gan:

    generator: Generator
    discriminator: Discriminator
    data_provider: DataProvider

    generator_model: tf.keras.Sequential
    discriminator_model: tf.keras.Sequential
    checkpoint: tf.train.Checkpoint

    data: BatchDataset

    seed = tf.random.normal([GIF_SIZE_SQUARED, IMAGE_SIZE * IMAGE_SIZE * 3])

    def __init__(self, generator: Generator, discriminator: Discriminator, data: DataProvider):
        self.generator = generator
        self.discriminator = discriminator
        self.data_provider = data

        self.generator_model = generator.get()
        self.discriminator_model = discriminator.get()

        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator.generator_optimizer,
            discriminator_optimizer=discriminator.discriminator_optimizer,
            generator=generator.get(),
            discriminator=discriminator.get(),
        )

    def load_data(self):
        self.data = self.data_provider.load_data()

    def train(self):
        print('Training started.')
        for save in range(SAVES):
            for epoch in range(EPOCHS):
                start = time.time()

                for image_batch in self.data:
                    self._train_step(image_batch)

                # Produce images for the GIF as you go
                display.clear_output(wait=True)
                self._generate_and_save_images(epoch + 1)

                # Save the model every 15 epochs
                if (epoch + 1) % 25 == 0:
                    self.checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

                print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        self._generate_and_save_images(EPOCHS)
        # self.checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
        self._display_image(EPOCHS)

    def make_gif(self):
        anim_file = GIF_NAME

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob('./test/image*.png')
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

    @tf.function
    def _train_step(self, images):
        noise = tf.random.normal([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator_model(noise, training=True)

            real_output = self.discriminator_model(images, training=True)
            fake_output = self.discriminator_model(generated_images, training=True)

            gen_loss = self.generator.loss(fake_output)
            disc_loss = self.discriminator.loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator_model.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator_model.trainable_variables)

        self.generator.apply_gradients(gradients_of_generator)
        self.discriminator.apply_gradients(gradients_of_discriminator)

    def _generate_and_save_images(self, epoch):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self.generator_model(self.seed, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((predictions[i, :, :, :] + 1) / 2)
            plt.axis('off')

        plt.savefig('./test/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show(block=WAIT)

    # Display a single image using the epoch number
    def _display_image(self, epoch_no):
        return PIL.Image.open('./test/image_at_epoch_{:04d}.png'.format(epoch_no))
