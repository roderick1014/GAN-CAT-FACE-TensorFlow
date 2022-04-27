import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Reshape, Conv2DTranspose, Dense, Dropout, Conv2D , Flatten, LeakyReLU, ReLU
from tensorflow.keras.preprocessing.image import array_to_img
import time as time

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tf.config.list_physical_devices('GPU'):
    print('\nUsing GPU for training (つ´ω`)つ\n')
else:
    print('\nNo GPU! 。･ﾟ･(つд`)･ﾟ･\n')

latent_dim = 128
BATCH_SIZE = 64
LOAD_MODEL = True
GEN_WEIGHT_PATH = 'GEN_WEIGHTS/epoch_500/'

generator = Sequential(
    [
        Input(shape=(latent_dim,)),
        Dense(8 * 8 * 128),
        Reshape((8, 8, 128)),
        Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        ReLU(),
        Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        ReLU(),
        Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        ReLU(),
        Conv2DTranspose(3, kernel_size=5, padding="same", activation="sigmoid"),
    ],
    name="generator",
)

if LOAD_MODEL:
    print("Loading weights...")
    generator.load_weights(GEN_WEIGHT_PATH)

random_latent_vectors = tf.random.normal(shape = (BATCH_SIZE, latent_dim))
print('Inferencing...')
current_time = time.time()
fake = generator(random_latent_vectors)
print('Time consuming: ', (time.time() - current_time),' s')
print('Finished')
fake_imgs_1 = fake[0]
fake_imgs_2 = fake[8]
fake_imgs_3 = fake[16]
fake_imgs_4 = fake[24]
for imgs in range(1, 8):
    fake_imgs_1 = np.hstack((fake_imgs_1, fake[imgs]))
    fake_imgs_2 = np.hstack((fake_imgs_2, fake[imgs + 8]))
    fake_imgs_3 = np.hstack((fake_imgs_3, fake[imgs + 8 * 2]))
    fake_imgs_4 = np.hstack((fake_imgs_4, fake[imgs + 8 * 3]))
            
fake_imgs = np.vstack((fake_imgs_1, fake_imgs_2))
fake_imgs = np.vstack((fake_imgs, fake_imgs_3))
fake_imgs = np.vstack((fake_imgs, fake_imgs_4))

img = array_to_img(fake_imgs)
img.show()
img.save("generated_img.png")