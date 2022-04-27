import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Reshape, Conv2DTranspose, Dense, Dropout, Conv2D , Flatten, LeakyReLU, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tqdm import tqdm
import numpy as np

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tf.config.list_physical_devices('GPU'):
    print('\nUsing GPU for training (つ´ω`)つ\n')
else:
    print('\nNo GPU! 。･ﾟ･(つд`)･ﾟ･\n')

GEN_WEIGHT_PATH = 'GEN_WEIGHTS'
DISC_WEIGHT_PATH = 'DISC_WEIGHTS'
SAVE_MODEL = True
LOAD_MODEL = False
BATCH_SIZE = 64
EPOCH = 2000

dataset = image_dataset_from_directory(
    directory="dataset",
    label_mode=None,
    image_size=(64, 64),
    batch_size=64,
    shuffle=True,
    seed=None,
    validation_split=None,
).map(lambda x: x / 255.0)

discriminator = Sequential(
    [
        Input(shape=(64, 64, 3)),
        Conv2D(64, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(alpha=0.2),
        Conv2D(128, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(0.2),
        Conv2D(256, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(0.2),
        Conv2D(512, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(0.2),
        Flatten(),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
print(discriminator.summary())                                                       # Shows the details of the discriminator

latent_dim = 128

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

generator.summary()                                                                  # Shows the details of the discriminator

opt_gen = Adam(1e-4)
opt_disc = Adam(1e-4)
loss_fn = BinaryCrossentropy()

if LOAD_MODEL:
    print("Loading weights...")
    generator.load_weights(GEN_WEIGHT_PATH)
    discriminator.load_weights(DISC_WEIGHT_PATH)

random_latent_vectors = tf.random.normal(shape = (BATCH_SIZE, latent_dim))


for epoch in range(EPOCH):
    print('=========================================================================================================================')
    tqdm_bar = tqdm(dataset)
    for idx, (real) in enumerate(tqdm_bar):
        batch_size = real.shape[0]
        with tf.GradientTape() as gen_tape:
            random_latent_vectors_train = tf.random.normal(shape = (batch_size, latent_dim))
            
            fake = generator(random_latent_vectors_train)
        
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        with tf.GradientTape() as disc_tape:
            loss_disc_real = loss_fn(tf.ones((batch_size, 1)), discriminator(real))
            loss_disc_fake = loss_fn(tf.zeros((batch_size, 1)), discriminator(fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

        grads = disc_tape.gradient(loss_disc, discriminator.trainable_weights)
        opt_disc.apply_gradients(zip(grads, discriminator.trainable_weights))

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with tf.GradientTape() as gen_tape:
            fake = generator(random_latent_vectors_train)
            output = discriminator(fake)
            loss_gen = loss_fn(tf.ones(batch_size, 1), output)

        grads = gen_tape.gradient(loss_gen, generator.trainable_weights)
        opt_gen.apply_gradients(zip(grads, generator.trainable_weights))

        if (idx % 245 == 0 and idx > 0):
            fake = generator(random_latent_vectors)
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
            img.save("gen_images/generated_img_epoch_%03d.png" % epoch)
            
            if SAVE_MODEL:
                generator.save_weights(GEN_WEIGHT_PATH + '/epoch_' + str(epoch) + '/')
                discriminator.save_weights(DISC_WEIGHT_PATH + '/epoch_' + str(epoch) + '/')
                
        tqdm_bar.set_description(f'Epoch [{epoch}/{EPOCH}]')
        tqdm_bar.set_postfix(Gen_loss = float(loss_gen), Disc_loss = float(loss_disc))

