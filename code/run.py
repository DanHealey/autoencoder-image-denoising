import tensorflow as tf

from autoencoder import Autoencoder

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image data 
DATA_DIRECTORY = "../data/"
IMAGE_SIZE = (128, 128)

# Noise parameters
NOISE_MEAN = 0
NOISE_STD = 30

# Model hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01
VALIDATION_SPLIT = 0.2
LATENT_DIM=512

def noiser(original_image):
    noise = tf.random.normal(original_image.shape, mean=NOISE_MEAN, stddev=NOISE_STD)
    noisy_image = tf.clip_by_value(original_image + noise, 0, 255)
    return noisy_image, original_image


def main():
    autoencoder = Autoencoder(LATENT_DIM, IMAGE_SIZE)

    train_datagen = ImageDataGenerator(
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=VALIDATION_SPLIT,
        preprocessing_function=noiser        
        )

    train_image_data = train_datagen.flow_from_directory(
        DATA_DIRECTORY,
        batch_size=BATCH_SIZE,
        target_size=IMAGE_SIZE,
        classes=None,
        class_mode=None,
        shuffle=True,
        subset="training",
    )

    test_image_data = train_datagen.flow_from_directory(
        DATA_DIRECTORY,
        batch_size=BATCH_SIZE,
        target_size=IMAGE_SIZE,
        classes=None,
        class_mode=None,
        shuffle=True,
        subset="validation"
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [
        tf.keras.metrics.MeanSquaredError()
    ]

    autoencoder.compile(
        optimizer=optimizer, 
        loss=loss,
        metrics=metrics
    )

    autoencoder.fit(
        train_image_data[0], 
        train_image_data[1],
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(test_image_data, test_image_data)
    )


if __name__ == "__main__":
    main()