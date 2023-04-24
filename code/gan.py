import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Dense, Flatten
from tensorflow.keras.constraints import Constraint

NOISE_MEAN = 0
NOISE_STD = 30/255

class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value
    
    # clip model weights to hypercube
    def __call__(self, weights):
        return tf.clip_by_value(weights, -self.clip_value, self.clip_value)
    
    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

class WGAN(Model):

    def __init__(self):

        super(WGAN, self).__init__()

        #self.c_clip = ClipConstraint(0.05)
        
        self.generator = Sequential([
            Conv2D(3, (3, 3), padding='SAME'),
            LeakyReLU(),
            #BatchNormalization(),

            Conv2D(16, (3, 3), padding='SAME'),
            LeakyReLU(),
            BatchNormalization(),

            Conv2D(16, (3, 3), padding='SAME'),
            LeakyReLU(),
            #BatchNormalization(),

            Conv2D(64, (3, 3), padding='SAME'),
            LeakyReLU(),
            BatchNormalization(),

            Conv2D(64, (3, 3), padding='SAME'),
            LeakyReLU(),
            #BatchNormalization(),

            Conv2D(128, (3, 3), padding='SAME'),
            LeakyReLU(),
            BatchNormalization(),

            Conv2D(128, (3, 3), padding='SAME'),
            LeakyReLU(),
            #BatchNormalization(),

            Conv2DTranspose(128, (3, 3), padding='SAME'),
            LeakyReLU(),
            BatchNormalization(),

            Conv2DTranspose(64, (3, 3), padding='SAME'),
            LeakyReLU(),
            #BatchNormalization(),

            Conv2DTranspose(64, (3, 3), padding='SAME'),
            LeakyReLU(),
            BatchNormalization(),

            Conv2DTranspose(16, (3, 3), padding='SAME'),
            LeakyReLU(),
            #BatchNormalization(),

            Conv2DTranspose(16, (3, 3), padding='SAME'),
            LeakyReLU(),
            BatchNormalization(),

            Conv2DTranspose(3, (3, 3), padding='SAME'),
            LeakyReLU(),
            #BatchNormalization(),

            Conv2DTranspose(3, (3, 3), padding='SAME'),
            tf.keras.layers.Activation('sigmoid')
        ])

        self.critic = Sequential([
            Conv2D(3, (3, 3)),
            LeakyReLU(),
            BatchNormalization(),

            Conv2D(16, (3, 3)),
            LeakyReLU(),
            MaxPooling2D(),
            BatchNormalization(),

            Conv2D(64, (5, 5)),
            LeakyReLU(),
            MaxPooling2D(),
            BatchNormalization(),

            Conv2D(64, (5, 5), strides=2),
            LeakyReLU(),
            BatchNormalization(),

            Flatten(),

            Dense(16, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        self.rec_loss = tf.keras.losses.MeanSquaredError()
        self.c_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
        self.g_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)

        self.crit_steps = 5

        #self.mse_metric = tf.keras.metrics.MeanSquaredError()
        self.critic_loss = tf.keras.metrics.Mean()
        self.gen_loss = tf.keras.metrics.Mean()

    def call(self, x):
        return self.critic(self.generator(x))

    def noise(self, x):
        noise = tf.random.normal(tf.shape(x), mean=NOISE_MEAN, stddev=NOISE_STD)
        noisy_image = tf.clip_by_value(x + noise, 0, 1)
        return noisy_image

    # STRAIGHT FROM TF, MAKE OUR OWN
    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.critic(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, x):

        images, labels = x
        x_real = images
        x_noisy = self.noise(x_real)
        x_fake = self.generator(x_noisy)

        for i in range(self.crit_steps):

            with tf.GradientTape() as tape:

                y_pred_real = self.critic(x_real)
                y_pred_fake = self.critic(x_fake)

                y_real = -tf.ones_like(y_pred_real)
                y_fake = tf.ones_like(y_pred_fake)

                y = tf.concat((y_real, y_fake), axis=0)

                loss = wasserstein_loss(y_real, y_fake) + self.gradient_penalty(tf.shape(x_real)[0], x_real, x_fake) * 10

            grads = tape.gradient(loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
            self.critic_loss.update_state(loss)

        
        with tf.GradientTape() as tape:
            denoised = self.generator(x_noisy)
            y_pred_fake = self.critic(denoised)
            y_fake = -tf.ones_like(y_pred_fake)

            loss = -tf.reduce_mean(y_fake)  + self.rec_loss(x_real, denoised)

        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        self.gen_loss.update_state(loss)

        return {'generator': self.gen_loss.result(), 'critic': self.critic_loss.result()}

    def test_step(self, x):

        images, labels = x
        x_real = images
        x_noisy = self.noise(x_real)
        x_fake = self.generator(x_noisy)

        y_pred_real = self.critic(x_real)
        y_pred_fake = self.critic(x_fake)

        y_real = -tf.ones_like(y_pred_real)
        y_fake = tf.ones_like(y_pred_fake)

        y = tf.concat((y_real, y_fake), axis=0)

        loss = wasserstein_loss(y, tf.concat((y_pred_real, y_pred_fake), axis=0))

        self.critic_loss.update_state(loss)

        denoised = self.generator(x_noisy)
        y_pred_fake = self.critic(denoised)
        y_fake = -tf.ones_like(y_pred_fake)

        loss = 0.5 * wasserstein_loss(y_fake, y_pred_fake) + self.rec_loss(x_real, denoised)
        
        self.gen_loss.update_state(loss)

        return {'generator': self.gen_loss.result(), 'critic': self.critic_loss.result()}

    @property
    def metrics(self):
        return [self.gen_loss, self.critic_loss]

def wasserstein_loss(real, fake):

    return tf.reduce_mean(fake) - tf.reduce_mean(real)