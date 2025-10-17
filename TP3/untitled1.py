import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.datasets import mnist
np.random.seed(123)  # for reproducibility

# Chargement et pré-traitement des données MNIST
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype("float32") / 255.
X_test = X_test.astype("float32") / 255.
X_train = np.reshape(X_train, (-1, 28, 28, 1))
X_test = np.reshape(X_test, (-1, 28, 28, 1))

# Couche personnalisée Sampling avec ajout de la loss KL
class Sampling(layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mu))
        z = mu + tf.exp(0.5 * log_var) * epsilon

        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
        self.add_loss(tf.reduce_mean(kl_loss))  # Ajout KL divergence loss au modèle

        return z

# Encoder
encoder_inputs = Input(shape=(28, 28, 1))
x = layers.Conv2D(16, 3, activation="relu", padding="same")(encoder_inputs)
x = layers.MaxPooling2D(2, padding="same")(x)
x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)

mu = layers.Dense(2, name="mu")(x)
log_var = layers.Dense(2, name="log_var")(x)
z = Sampling()([mu, log_var])

encoder = Model(encoder_inputs, [z, mu, log_var], name="encoder")
encoder.summary()

# Decoder
latent_inputs = Input(shape=(2,))
x = layers.Dense(16, activation="relu")(latent_inputs)
x = layers.Dense(7 * 7 * 32, activation="relu")(x)
x = layers.Reshape((7, 7, 32))(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)
decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

decoder = Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# VAE (Encoder + Decoder)
vae_inputs = Input(shape=(28, 28, 1))
z, mu, log_var = encoder(vae_inputs)
reconstructions = decoder(z)
vae = Model(vae_inputs, reconstructions, name="vae")

# Compilation (seulement reconstruction loss, KL loss ajoutée via add_loss dans Sampling)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy())

# Entraînement
history = vae.fit(X_train, X_train,
                  epochs=20,
                  batch_size=64,
                  validation_data=(X_test, X_test))

# Affichage des courbes d’apprentissage
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.legend()
plt.title("Loss pendant l'entraînement")
plt.show()

# Reconstruction
z_test, _, _ = encoder.predict(X_test)
decoded_imgs = decoder.predict(z_test)

for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.subplot(2, 5, i + 6)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
plt.show()

