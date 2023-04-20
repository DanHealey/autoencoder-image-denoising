import tensorflow as tf

from autoencoder import Autoencoder

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image data 
DATA_DIRECTORY = "../data/data/"
IMAGE_SIZE = (150, 150)

# Model hyperparameters
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 0.01
VALIDATION_SPLIT = 0.2
LATENT_DIM=512

def main():
    autoencoder = Autoencoder(LATENT_DIM, IMAGE_SIZE)

    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=VALIDATION_SPLIT,        
    )

    train_image_data = datagen.flow_from_directory(
        DATA_DIRECTORY,
        batch_size=BATCH_SIZE,
        target_size=IMAGE_SIZE,
        classes=None,
        class_mode=None,
        shuffle=True,
        subset="training",
    )

    test_image_data = datagen.flow_from_directory(
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
        train_image_data[0],
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(test_image_data[0], test_image_data[0])
    )


if __name__ == "__main__":
    main()