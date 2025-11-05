# Transfert learning et Fine tuning sur caltech101

import numpy as np
import os
import random
import time
np.random.seed(123)  # for reproducibility
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense, Flatten, Dropout, SpatialDropout2D, Convolution2D, MaxPooling2D, Reshape, Conv2DTranspose, Resizing, BatchNormalization
from tensorflow.keras.datasets.mnist import load_data
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def affiche(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

##########################
# 1.1 Chargement et mise en forme des données
##########################

DATA_PATH = 'Data'
categories=["accordion", "anchor", "barrel", "binocular"]
for i in range(len(categories)):
    categories[i]=DATA_PATH + '/'+categories[i]

data = []
for c, category in enumerate(categories):
    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(category)
              for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]
    for img_path in images:
	    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
	    x=np.array(img)
	    x = np.expand_dims(x, axis=0)
	    data.append({'x': np.array(x[0]), 'y':c})

num_classes = len(categories)
random.shuffle(data)

#create train / val / test split
train_split = 0.7
idx_train = int(train_split * len(data))
train = data[:idx_train]
test = data[idx_train:]

#separate data and labels
X_train, y_train = np.array([t['x'] for t in train]), [t['y'] for t in train]
X_test, y_test = np.array([t['x'] for t in test]), [t['y'] for t in test]

#normalize data
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

#convert labels to one-hot vectors
Y_train = tf.keras.utils.to_categorical(y_train, num_classes)
Y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print('finished loading',len(data),'images from',num_classes,'categories')
print('train / test split:',  len(X_train), len(X_test))
print('training data shape: ', X_train.shape)
print('training label shape: ', len(y_train))

shape = X_train[0].shape

##########################
# 1.2. Apprentissage from scratch d'un modèle convolutionnel
##########################

def CNN(shape):
    inputs = Input(shape)
    x= inputs
    #x = Resizing(128,128,interpolation='bilinear')(x)
    x=Convolution2D(32,(3,3), activation='relu',padding='same')(x)
    x=MaxPooling2D(pool_size=(3, 3), strides=2,padding='same')(x)
    
    x=Convolution2D(64,(3, 3), activation='relu',padding='same')(x)
    x=MaxPooling2D(pool_size=(3, 3), strides=2,padding='same')(x)
    
    x=Flatten()(x)
    x=Dense(100, activation='relu')(x)
    outputs=Dense(4, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

model = CNN(shape)
model.summary()

# III.2. Apprentissage
lr=0.0001
batch_size=min([X_train.shape[0], 256])
epochs=40

ad= tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(loss='categorical_crossentropy', optimizer=ad, metrics=['accuracy'])
tps1 = time.time()
history =model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))
tps2 = time.time()

affiche(history)
print('lr=',lr,'batch_size=',batch_size, 'epochs=',epochs)
print('Temps d apprentissage',tps2 - tps1)
print('val_accuracy',history.history['val_accuracy'][-1])

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)




'''



########################
# Auto-encodeur
########################

(X_train, y_train), (X_test, y_test) = load_data()

#Ne conserve que 10% de la base
X_train, pipo, y_train, pipo = train_test_split(X_train, y_train, test_size=0.90)
X_test, pipo, y_test, pipo = train_test_split(X_test, y_test, test_size=0.90)

# Preprocess input data
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255



X_train_noise = X_train + 0.2 * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noise = X_test + 0.4 * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)


X_train_noise = np.clip(X_train_noise, 0.0, 1.0)
X_test_noise = np.clip(X_test_noise, 0.0, 1.0)

# Display the train data and a version of it with added noise
for i in range(5):
  plt.subplot(2,5,i+1)
  plt.imshow(X_train[i,:].reshape([28,28]), cmap='gray')
  plt.axis('off')
  plt.subplot(2,5,i+6)
  plt.imshow(X_train_noise[i,:].reshape([28,28]), cmap='gray')
  plt.axis('off')
plt.show()




########################
# Auto-encodeur variationnel
########################

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

vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=loss_func(mu, log_variance))


vae.fit(X_train, X_train, epochs=70, batch_size=32, shuffle=True, validation_data=(X_test, X_test))



'''