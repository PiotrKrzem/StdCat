import os

ALL_DIR = "./training_data/Images/"
PUG_DIR = "./training_data/Images/n02110958-pug/"
CHECKPOINT_DIR = './training_checkpoints'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")
GIF_NAME = "dcgan.gif"

BUFFER_SIZE = 22500
BATCH_SIZE = 2048
EPOCHS = 400
SAVES = 1

IMAGE_SIZE = 64

GIF_SIZE = 4 #GIF_SIZE = S means SxS examples per gif frame
GIF_SIZE_SQUARED = GIF_SIZE*GIF_SIZE

WAIT = False #wait when gif is shown