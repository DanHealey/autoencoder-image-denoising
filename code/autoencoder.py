import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensofrlow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose

class Autoencoder(Model):
    def __init__(self, latent_dim=512, input_size=(150, 150)):

        super(Autoencoder, self).__init__()

        # Latent Dimension and input size
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Encoder
        self.encoder = Sequential([
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), padding="same")
        ])

        # Decoder
        self.decoder = Sequential([
            Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")
        ])
        
    def call(self, x):
        return self.decoder(self.encoder(x))