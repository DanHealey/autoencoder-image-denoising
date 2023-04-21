import tensorflow as tf

from autoencoder import Autoencoder

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image data 
TRAIN_DATA_DIRECTORY = "../data/data/train"
TEST_DATA_DIRECTORY = "../data/data/test"
IMAGE_SIZE = (150, 150)

# Noise parameters
NOISE_MEAN = 0
NOISE_STD = 30

# Model hyperparameters
EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.01
VALIDATION_SPLIT = 0.2
LATENT_DIM=512

def noiser(original_image):
    noise = tf.random.normal(original_image.shape, mean=NOISE_MEAN, stddev=NOISE_STD)
    noisy_image = tf.clip_by_value(original_image + noise, 0, 1)
    return noisy_image

def main():
    autoencoder = Autoencoder(LATENT_DIM, IMAGE_SIZE)

    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=True,
        vertical_flip=True,  
        preprocessing_function=noiser     
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIRECTORY,
        batch_size=BATCH_SIZE,
        target_size=IMAGE_SIZE,
        classes=None,
        class_mode="input",
        shuffle=True,
    )


    test_datagen = ImageDataGenerator(
        rescale=1.0/255.0    
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_DIRECTORY,
        batch_size=BATCH_SIZE,
        target_size=IMAGE_SIZE,
        classes=None,
        class_mode="input",
        shuffle=True
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
        train_generator,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=test_generator
    )


if __name__ == "__main__":
    main()