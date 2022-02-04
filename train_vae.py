#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys
import os.path
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K

from src.vae_sampler import *

def tversky(y_true, y_pred, smooth=1e-5, alpha=0.7):

    # make y_true per class
    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))

    # flatten per class
    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)
    
    true_pos = K.sum(y_true_pos * y_pred_pos, 1)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
    false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return K.sum(1 - tversky(y_true, y_pred))

def load_data(n, sampler, dim=(120,160,3)):
    X = np.empty((n, *dim))
    i = 0
    print('Loading data....')
    while True:
        image = sampler.random_image()
        if image is not None:
            image = cv2.resize(image, (dim[1],dim[0]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            X[i,:,:,:] = image
            i+=1
            sys.stdout.write('\r' + f'Loaded {i}/{n}')
        if i >= n:
            break
    X =  X.astype("float32") / 255
    print('\n data loading done')
    return X

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding an image."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def get_encoder(latent_dim):
    encoder_inputs = keras.Input(shape=(120, 160, 3))
    x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(encoder_inputs) # 80x60
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x) # 40x30
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x) # 20x15
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    #encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder = keras.Model(encoder_inputs, z_mean, name="encoder")
    encoder.summary()
    return encoder

def get_decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(15 * 20 * 128, activation="relu")(latent_inputs)
    x = layers.Reshape((15, 20, 128))(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
    #x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(x) 
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    #x = layers.Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            #reconstruction_loss = tversky_loss(data, reconstruction)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def plot_result(data, encoder, decoder):
    #z_mean, z_log_var, z = vae.encoder.predict(data)
    z = encoder.predict(data)
    reconstructed = decoder.predict(z)

    f, axs = plt.subplots(2,12, figsize=(20,7))
    i = 0
    for x in range(12):
        axs[0,x].imshow(data[i])
        axs[1,x].imshow(reconstructed[i])
        i+=1
    plt.show()

if __name__ == "__main__":
    latent_dim = 10
    n = 5000
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)

    if not os.path.isfile('data.dat'):
        sampler = VAE_Sampler('./track.png', image_res=(640,480))
        data = load_data(n, sampler)
        data.tofile('data.dat')
        print('saved data to data.dat')
    else:
        print('loading data.dat file')
        data = np.fromfile('data.dat', dtype=np.float32)
        data = np.reshape(data, (n,120,160,3))
        print('done loading')

    #vae = VAE(encoder, decoder)
    #vae.compile(optimizer=keras.optimizers.Adam())
    inputs = keras.Input(shape=(120,160,3), name='encoder_input')
    vae = keras.Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    vae.compile(loss=tversky_loss, optimizer=keras.optimizers.Adam())
    vae.fit(data, data, epochs=30, batch_size=128)

    encoder.save('encoder.h5')

    plot_result(data[:12,:,:,:], encoder,decoder)
