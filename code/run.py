import tensorflow as tf

from autoencoder import Autoencoder

from tf.keras.preprocessing.image import ImageDataGenerator


def main():
    # Image data 
    DATA_DIRECTORY = "../data/"
    IMAGE_SIZE = (128, 128)

    # Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    VALIDATION_SPLIT = 0.2

    LATENT_DIM=512

    autoencoder = Autoencoder(LATENT_DIM, IMAGE_SIZE)

    train_datagen = ImageDataGenerator(
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=VALIDATION_SPLIT
        )

    train_image_data = train_datagen.flow_from_directory(
        DATA_DIRECTORY,
        batch_size=BATCH_SIZE,
        target_size=IMAGE_SIZE,
        classes=None,
        class_mode=None,
        shuffle=True,
        subset="training"
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

    # ADD NOISE TO IMAGES
    
    autoencoder.compile(
        optimizer='adam', 
        loss=tf.keras.losses.MeanSquaredError()
    )

    autoencoder.fit(
        train_image_data, 
        train_image_data,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(test_image_data, test_image_data)
    )


if __name__ == "__main__":
    main()