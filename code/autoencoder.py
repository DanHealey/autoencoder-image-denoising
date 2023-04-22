import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, LeakyReLU

# Noise parameters
NOISE_MEAN = 0
NOISE_STD = 30/255

def noiser(original_image):
    noise = tf.random.normal(tf.shape(original_image), mean=NOISE_MEAN, stddev=NOISE_STD)
    noisy_image = tf.clip_by_value(original_image + noise, 0, 1)
    return noisy_image

class Autoencoder(Model):
    def __init__(self, latent_dim=512, input_size=(150, 150)):

        super(Autoencoder, self).__init__()

        # Latent Dimension and input size
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Encoder
        #self.encoder = Sequential([
            
        #])

        self.c1_1 = Conv2D(3, (3, 3), padding='SAME')
        self.c1_2 = LeakyReLU()
        self.c1_3 = Conv2D(16, (3, 3), padding='SAME')
        self.c1_4 = LeakyReLU()
        self.c1_5 = MaxPooling2D(padding='SAME')
        self.bn_1 = tf.keras.layers.BatchNormalization()

        #Conv2D(16, (3, 3), padding='SAME'),
        #LeakyReLU(),
        self.c2_1 = Conv2D(64, (3, 3), padding='SAME')
        self.c2_2 = LeakyReLU()
        self.bn_2 = tf.keras.layers.BatchNormalization()
        #MaxPooling2D(padding='SAME'),

        self.c3_1 = Conv2D(64, (3, 3), padding='SAME')
        self.c3_2 = LeakyReLU()
        self.c3_3 = Conv2D(256, (3, 3), padding='SAME')
        self.c3_4 = LeakyReLU()
        self.bn_3 = tf.keras.layers.BatchNormalization()
        #MaxPooling2D(padding='SAME'),

        self.c3_5 = Conv2D(256, (3, 3), padding='SAME')
        self.c3_6 = LeakyReLU()
        #Conv2D(256, (3, 3), padding='SAME'),
        #LeakyReLU(),

        # Decoder
        # self.decoder = Sequential([
            
        # ])

        self.c4_01 = Conv2DTranspose(256, (3, 3), padding='SAME')
        self.c4_02 = LeakyReLU()
        #Conv2DTranspose(256, (3, 3), padding='SAME'),

        #Conv2DTranspose(256, (3, 3), strides=2, padding='SAME'),
        #LeakyReLU(),
        self.c4_1 = Conv2DTranspose(256, (3, 3), padding='SAME')
        self.c4_2 = LeakyReLU()
        self.c4_3 = Conv2DTranspose(64, (3, 3), padding='SAME')
        self.c4_4 = LeakyReLU()
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_4 = tf.keras.layers.BatchNormalization()

        #Conv2DTranspose(64, (3, 3), strides=2, padding='SAME'),
        
        self.c5_1 = Conv2DTranspose(64, (3, 3), padding='SAME')
        self.c5_2 = LeakyReLU()
        self.bn_5 = tf.keras.layers.BatchNormalization()
        #Conv2DTranspose(16, (3, 3), padding='SAME'),

        self.c6_1 = Conv2DTranspose(16, (3, 3), strides=2, padding='SAME')
        self.c6_2 = LeakyReLU()
        self.bn_6 = tf.keras.layers.BatchNormalization()
        #Conv2DTranspose(16, (3, 3), padding='SAME'),
        #LeakyReLU(),
        self.c6_3 = Conv2DTranspose(3, (3, 3), padding='SAME', activation='sigmoid')
        
    def call(self, x):

        x = self.c1_1(x)
        x = self.c1_2(x)
        x = self.c1_3(x)
        x = self.c1_4(x)
        x = self.c1_5(x)
        x = self.bn_1(x)

        c2_1 = self.c2_1(x)
        x = self.c2_2(c2_1)
        x = self.bn_2(x)

        c3_1 = self.c3_1(x)
        x = self.c3_2(c3_1)
        x = self.c3_3(x)
        x = self.c3_4(x)
        x = self.bn_3(x)

        x = self.c3_5(x)
        x = self.c3_6(x)

        x = self.c4_01(x)
        x = self.c4_02(x)

        x = self.c4_1(x)
        x = self.c4_2(x)
        x = self.c4_3(x)
        x = self.c4_4(x) + c3_1
        x = self.bn_4(x)

        x = self.c5_1(x) + c2_1
        x = self.c5_2(x)
        x = self.bn_5(x)

        x = self.c6_1(x)
        x = self.c6_2(x)
        x = self.bn_6(x)
        x = self.c6_3(x) 


        return x

    def train_step(self, inputs):

        inputs, labels = inputs

        noisy = noiser(inputs)

        with tf.GradientTape() as tape:

            denoised = self(noisy)
            loss = self.compiled_loss(inputs, denoised)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.compiled_metrics.update_state(inputs, denoised)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):

        inputs, labels = inputs

        noisy = noiser(inputs)

        denoised = self(noisy)

        self.compiled_metrics.update_state(inputs, denoised)

        return {m.name: m.result() for m in self.metrics}