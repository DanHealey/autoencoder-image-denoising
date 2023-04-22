import tensorflow as tf
from PIL import Image
import numpy as np

from autoencoder import Autoencoder

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

# Image data 
TRAIN_DATA_DIRECTORY = "C:/datasets/bim"
TEST_DATA_DIRECTORY = "C:/datasets/bim"
IMAGE_SIZE = (256, 256)

# Visualization data
VISUALIZE_IMAGE_PATH = "C:/datasets/intel/seg_pred/5.jpg"

# Noise parameters
NOISE_MEAN = 0
NOISE_STD = 30/255

# Model hyperparameters
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 0.0005
VALIDATION_SPLIT = 0.2
LATENT_DIM=512

def noiser(original_image):
    noise = tf.random.normal(tf.shape(original_image), mean=NOISE_MEAN, stddev=NOISE_STD)
    noisy_image = tf.clip_by_value(original_image + noise, 0, 1)
    return noisy_image


def resize(images):
    return tf.image.resize(images, (152, 152))

def visualize_results(model):
    image_path = "C:/datasets/intel/seg_pred/" + input("filename")
    image = load_img(image_path, target_size=IMAGE_SIZE)
    image = img_to_array(image)
    image_input = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))/255
    noisy_image = noiser(image_input)
    model_output = model(noisy_image)

    Image.fromarray((np.asarray(image_input).squeeze()*255).astype(np.uint8)).show()
    Image.fromarray((np.asarray(noisy_image).squeeze()*255).astype(np.uint8)).show()
    Image.fromarray((np.asarray(model_output).squeeze()*255).astype(np.uint8)).show()

def main():
    autoencoder = Autoencoder(LATENT_DIM, IMAGE_SIZE)

    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=True,
        vertical_flip=True,  
        validation_split=VALIDATION_SPLIT
        #preprocessing_function=resize     
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIRECTORY,
        batch_size=BATCH_SIZE,
        target_size=IMAGE_SIZE,
        classes=None,
        class_mode="input",
        shuffle=True,
        subset='training',
        keep_aspect_ratio=True,
    )


    test_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        #preprocessing_function=resize
    )

    test_generator = train_datagen.flow_from_directory(
        TEST_DATA_DIRECTORY,
        batch_size=BATCH_SIZE,
        target_size=IMAGE_SIZE,
        classes=None,
        class_mode="input",
        shuffle=True,
        subset='validation',
        keep_aspect_ratio=True,
    )
    
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
    def psnr_loss(y_true, y_pred):
        return -tf.image.psnr(y_true, y_pred, max_val=1.0)
    def ssim_loss(y_true, y_pred):
        return -tf.image.ssim(y_true, y_pred, 1)
    mse = tf.keras.losses.MeanSquaredError()
    def mix(y_true, y_pred):
        return 0.5 * ssim_loss(y_true, y_pred) + 0.5 * mse(y_true, y_pred)
    loss = ssim_loss
    metrics = [
        tf.keras.metrics.MeanSquaredError(),
        psnr_loss,
    ]

    autoencoder.compile(
        optimizer=optimizer, 
        loss=loss,
        metrics=metrics
    )

    autoencoder.build(input_shape=(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    autoencoder.summary()

    autoencoder.fit(
        train_generator,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=test_generator
    )

    autoencoder.save_weights('weights_ssim_only')

    for i in range(10):
        visualize_results(autoencoder)

if __name__ == "__main__":
    main()