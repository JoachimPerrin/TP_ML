import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Dropout, SpatialDropout2D, Convolution2D, Conv2D, MaxPooling2D, Reshape, Conv2DTranspose, Resizing, BatchNormalization, Lambda
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
from matplotlib import pyplot as plt

#%%utils
def affiche(history):
    metrics = [m for m in history.history.keys() if not m.startswith('val_')]
    for m in metrics:
        val_m = f'val_{m}'
        plt.figure()
        plt.plot(history.history[m], label=f'Train {m}')
        if val_m in history.history:
            plt.plot(history.history[val_m], label=f'Val {m}')
        plt.title(f'{m} evolution')
        plt.xlabel('Epoch')
        plt.ylabel(m)
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

def eval_classif(Y_true, Y_pred):

    y_true_classes = np.argmax(Y_true, axis=1)
    y_pred_classes = np.argmax(Y_pred, axis=1)
    
    # Matrice de confusion
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Matrice de confusion")
    plt.show()

    # Rapport de classification
    print("\nClassification report:")
    print(classification_report(y_true_classes, y_pred_classes))

#%% Data preparation
def load_dataset(data_path, categories, target_size=(224, 224), train_split=0.7):
    categories = [os.path.join(data_path, cat) for cat in categories]
    
    data = []
    for c, category in enumerate(categories):
        images = [os.path.join(dp, f) for dp, _, filenames in os.walk(category)
                  for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]
        for img_path in images:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
            x = np.array(img, dtype=np.float32)
            data.append({'x': x, 'y': c})
    
    num_classes = len(categories)
    random.shuffle(data)

    # Split train/test
    idx_train = int(train_split * len(data))
    train, test = data[:idx_train], data[idx_train:]

    # Séparation données/labels
    X_train = np.array([t['x'] for t in train])
    y_train = [t['y'] for t in train]
    X_test = np.array([t['x'] for t in test])
    y_test = [t['y'] for t in test]

    # Normalisation
    X_train /= 255.0
    X_test /= 255.0

    # Encodage one-hot
    Y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    Y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    print(f"Finished loading {len(data)} images from {num_classes} categories.")
    print(f"Train/Test split: {len(X_train)} / {len(X_test)}")

    return (X_train, Y_train), (X_test, Y_test), num_classes

def load_mnist_with_noise(noise_train=0.2, noise_test=0.4):
    from tensorflow.keras.datasets.mnist import load_data

    # Load MNIST
    (X_train, Y_train), (X_test, Y_test) = load_data()

    # Preprocess input data
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Add noise
    X_train_noise = X_train + 0.2 * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_test_noise = X_test + 0.4 * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

    # Clip to [0,1]
    X_train_noise = np.clip(X_train_noise, 0.0, 1.0)
    X_test_noise = np.clip(X_test_noise, 0.0, 1.0)

    return (X_train, Y_train, X_train_noise), (X_test, Y_test, X_test_noise)


#%% Model definition
data_augmentation = tf.keras.Sequential([
    # tf.keras.layers.RandomFlip("vertical"),
    # tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    # tf.keras.layers.RandomZoom(0.1),
    # tf.keras.layers.RandomContrast(100)
])

def CNN(shape, nb_classes=4):
    inputs = Input(shape)
    x = data_augmentation(inputs)

    x=Conv2D(16,(3, 3), activation='relu',padding='same')(x)
    x=MaxPooling2D(pool_size=(2, 2), strides=2,padding='same')(x)
    # x=Dropout(0.1)(x)

    x=Conv2D(32,(3, 3), activation='relu',padding='same')(x)
    x=Conv2D(32,(3, 3), activation='relu',padding='same')(x)
    x=MaxPooling2D(pool_size=(2, 2), strides=2,padding='same')(x)
    x=Dropout(0.05)(x)

    x=Conv2D(64,(3, 3), activation='relu',padding='same')(x)
    x=Conv2D(64,(3, 3), activation='relu',padding='same')(x)
    x=Dropout(0.1)(x)

    x=Flatten()(x)
    x=Dense(100, activation='relu')(x)
    x=Dropout(0.35)(x)
    outputs=Dense(nb_classes, activation='softmax')(x)

    return Model(inputs, outputs)

def MLP_transfer(input_tensor, nb_classes):
    x = Dense(256, activation='relu')(input_tensor)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    return Dense(nb_classes, activation='softmax')(x)


def build_autoencoder(shape):
    inputs = layers.Input(shape)

    # Encodeur
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)

    # Code latent
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    
    # Décodeur
    x = layers.Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(16, (3,3), strides=2, activation='relu', padding='same')(x)
    
    outputs = layers.Conv2D(1, (3,3), activation='linear', padding='same')(x)

    return Model(inputs, outputs)



# def build_decoder_vae(latent_dim=2):
#     decoder_inputs = layers.Input(shape=(latent_dim,))
#     x = layers.Dense(7*7*32, activation='relu')(decoder_inputs)
#     x = layers.Reshape((7,7,32))(x)
#     x = layers.Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same')(x)
#     x = layers.Conv2DTranspose(16, (3,3), strides=2, activation='relu', padding='same')(x)
#     outputs = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
#     decoder = Model(decoder_inputs, outputs, name="decoder")
#     return decoder

# def build_vae(input_shape=(28,28,1), latent_dim=2, coeff=1.0):
#     encoder = build_encoder_vae(input_shape, latent_dim, coeff)
#     decoder = build_decoder_vae(latent_dim)
#     vae_inputs = layers.Input(shape=input_shape)
#     z, mu, log_var = encoder(vae_inputs)
#     reconstructions = decoder(z)
#     vae = Model(vae_inputs, reconstructions, name="vae")
#     return vae, encoder, decoder

class Sampling(layers.Layer):
    def __init__(self, coeff=1.0, **kwargs):
        super().__init__(**kwargs)
        self.coeff = coeff

    def call(self, inputs):
        mu, log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(mu))
        z = mu + tf.exp(0.5 * log_var) * epsilon
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
        self.add_loss(self.coeff * tf.reduce_mean(kl_loss))
        return z
    
def build_encoder(input_shape=(28,28,1), latent_dim=2, coeff=1.0):
    encoder_inputs = Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(encoder_inputs)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    mu = layers.Dense(latent_dim, name='mu')(x)
    log_var = layers.Dense(latent_dim, name='log_var')(x)
    z = Sampling(coeff=coeff)([mu, log_var])
    encoder = Model(encoder_inputs, [z, mu, log_var], name='encoder')
    return encoder

def build_decoder(latent_dim=2):
    decoder_inputs = Input(shape=(latent_dim,))
    x = layers.Dense(7*7*32, activation='relu')(decoder_inputs)
    x = layers.Reshape((7,7,32))(x)
    x = layers.Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(16, (3,3), strides=2, activation='relu', padding='same')(x)
    outputs = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
    decoder = Model(decoder_inputs, outputs, name='decoder')
    return decoder

def build_vae(input_shape=(28,28,1), latent_dim=2, coeff=0.01):
    encoder = build_encoder(input_shape, latent_dim, coeff)
    decoder = build_decoder(latent_dim)
    vae_inputs = Input(shape=input_shape)
    z, mu, log_var = encoder(vae_inputs)
    reconstructions = decoder(z)
    vae = Model(vae_inputs, reconstructions, name='vae')
    return vae, encoder, decoder
