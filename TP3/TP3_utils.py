# Transfert learning et Fine tuning sur caltech101

import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense, Flatten, Dropout, SpatialDropout2D, Conv2D, MaxPooling2D, Reshape, Conv2DTranspose, Resizing, BatchNormalization, Lambda
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

    global_acc = accuracy_score(y_true_classes, y_pred_classes)
    print(f"\nPrécision globale: {global_acc:.4f}")

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
    (X_train, y_train), (X_test, y_test) = load_data()

    # Reshape to (samples, height, width, channels)
    X_train = X_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    X_test  = X_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

    # One-hot encode labels
    Y_train = to_categorical(y_train, 10)
    Y_test  = to_categorical(y_test, 10)

    # Add Gaussian noise
    X_train_noisy = X_train + noise_train * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_test_noisy  = X_test  + noise_test  * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

    # Clip to [0,1]
    X_train_noisy = np.clip(X_train_noisy, 0.0, 1.0)
    X_test_noisy  = np.clip(X_test_noisy, 0.0, 1.0)

    return (X_train, Y_train, X_train_noisy), (X_test, Y_test, X_test_noisy)


#%% Model definition
data_augmentation = tf.keras.Sequential([
    # tf.keras.layers.RandomFlip("vertical"),
    tf.keras.layers.RandomFlip("horizontal"),
    # tf.keras.layers.RandomRotation(0.2),
    # tf.keras.layers.RandomZoom(0.2),
    # tf.keras.layers.RandomContrast(0.1)
])

def CNN(shape, nb_classes=4):
    inputs = Input(shape)

    x=Conv2D(32,(3,3), activation='relu',padding='same')(inputs)
    x=MaxPooling2D(pool_size=(3, 3), strides=2,padding='same')(x)

    x=Conv2D(64,(3, 3), activation='relu',padding='same')(x)
    x=MaxPooling2D(pool_size=(3, 3), strides=2,padding='same')(x)
    
    x=Flatten()(x)
    x=Dense(100, activation='relu')(x)
    outputs=Dense(nb_classes, activation='softmax')(x)

    return Model(inputs, outputs)

def MLP_transfer(shape, nb_classes):
    inputs = Input(shape)
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x) # To be tuned (number of neurons, layers etc.)
    # Softmax activation for multi-class classification
    outputs = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

def encoder(shape):
    inputs = Input(shape)
    
    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2), name='p1')(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2), name='p2')(c2)

    # Backbone
    b = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    
    model = Model(inputs, b)
    return model

def decoder(shape):
    inputs = Input(shape)    
    # Decoder 
    u2 = Conv2DTranspose(32, (3,3), activation='relu', strides=(2,2), padding='same', name='u2')(inputs)
    u1 = Conv2DTranspose(16, (3,3), activation='relu', strides=(2,2), padding='same', name='u1')(u2)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u1)

    model = Model(inputs, outputs)
    return model


def auto_encoder(shape):
    inputs = Input(shape)   
    x = encoder(shape)(inputs)
    outputs = decoder(x.shape[1:])(x)
    model = Model(inputs, outputs)
    return model

## VAE
def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tf.keras.backend.exp(log_variance/2) * epsilon
    return random_sample

def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss


    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + 0*kl_loss
        #loss = reconstruction_loss 
        return loss

    return vae_loss

def vae(input_shape, latent_dim=2):

    encoder_model = encoder(input_shape)
    x = Flatten()(encoder_model.output)
    x = Dense(64, activation='relu')(x)

    mu = Dense(latent_dim, name='mu')(x)
    log_var = Dense(latent_dim, name='log_var')(x)
    z = Lambda(sampling, name="sampling")([mu, log_var])

    decoder_input_shape = encoder_model.output_shape[1:]
    units = np.prod(decoder_input_shape)
    x_decoded = Dense(units, activation='relu')(z)
    x_decoded = Reshape(decoder_input_shape)(x_decoded)

    decoder_model = decoder(decoder_input_shape)
    outputs = decoder_model(x_decoded)

    vae_model = Model(encoder_model.input, outputs, name="VAE")

    return vae_model, mu, log_var