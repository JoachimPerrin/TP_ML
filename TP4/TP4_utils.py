#%% imports
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
# from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import time



#%% utils
# def dice_loss(y_true, y_pred, smooth=1e-6):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# def bce_dice_loss(y_true, y_pred):
#     bce = K.mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
#     dl = dice_loss(y_true, y_pred)
#     return bce + dl

# def dice_coef(y_true, y_pred, smooth=1e-6):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def display_results(X_test, Y_test, preds):
    plt.figure(figsize=(12, 12))
    for i in range(5):
        plt.subplot(5, 3, 3*i+1)
        plt.imshow(X_test[i])
        plt.title("Image")
        
        plt.subplot(5, 3, 3*i+2)
        plt.imshow(Y_test[i], cmap='gray')
        plt.title("True Mask")
        
        plt.subplot(5, 3, 3*i+3)
        plt.imshow(preds[i]>0.5, cmap='gray')
        plt.title("Predicted Mask")
    
    plt.tight_layout()
    plt.show()

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

def eval_classif(Y_true, Y_pred, threshold=0.5):
    # Binarisation des prédictions
    Y_pred_bin = (Y_pred > threshold).astype(int)
    
    # Aplatir pour comparer pixel à pixel
    y_true_flat = Y_true.flatten()
    y_pred_flat = Y_pred_bin.flatten()
    
    # Matrice de confusion
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Background", "Object"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Matrice de confusion (pixels)")
    plt.show()

    # Rapport de classification
    print("\nClassification report (par pixel):")
    print(classification_report(y_true_flat, y_pred_flat, target_names=["Background", "Object"]))


#############################################################
#%% QUESTION2 
#############################################################
def unet_simple(input_shape):
    inputs = layers.Input(input_shape)
    # x = layers.RandomFlip("horizontal")(inputs)

    # Encodeur
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='c2')(inputs)
    # c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='c2_2')(c2)
    p2 = layers.MaxPooling2D((2, 2), name='p2')(c2)

    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='c1')(p2)
    # c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='c1_2')(c1)
    c1 = layers.Dropout(0.3)(c1)
    p1 = layers.MaxPooling2D((2, 2), name='p1')(c1)

    # Bottleneck
    b = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='b')(p1)
    b = layers.Dropout(0.5)(b)
    # b = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='b_2')(b)
    # b = layers.Dropout(0.5)(b)

    # # Decodeur
    u1 = layers.UpSampling2D((2, 2), name='u1')(b)
    u1 = layers.concatenate([u1, c1], name='u1b')
    cd1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='cd1')(u1)
    # cd1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='cd1_2')(cd1)

    u2 = layers.UpSampling2D((2, 2), name='u2')(cd1)
    u2 = layers.concatenate([u2, c2], name='u2b')
    cd2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='cd2')(u2)
    # cd2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='cd2_2')(cd2)

    # Output
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", name='outputs')(cd2)

    model = models.Model(inputs, outputs)
    return model



#############################################################
#%% QUESTION4 
#############################################################
    
def affiche_deep(history):
    # summarize history for accuracy
    plt.plot(history.history['output1_accuracy'])
    plt.plot(history.history['val_output1_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_output1_loss'])
    plt.plot(history.history['val_output2_loss'])
    plt.plot(history.history['val_output3_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'test1', 'test2', 'test3'], loc='upper left')
    plt.show()
    








