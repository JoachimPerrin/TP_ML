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
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
import time



#%% utils

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

    global_acc = accuracy_score(y_true_flat, y_pred_flat)
    print(f"\nPrécision globale (pixel accuracy): {global_acc:.4f}")

#%% Model definition



#############################################################
#%% QUESTION4 
#############################################################
    
def unet_simple(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='c1')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='c1_2')(c1)
    p1 = layers.MaxPooling2D((2, 2), name='p1')(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='c2')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='c2_2')(c2)
    p2 = layers.MaxPooling2D((2, 2), name='p2')(c2)
    
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='c3')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='c3_2')(c3)
    c3 = layers.Dropout(0.3)(c3)
    p3 = layers.MaxPooling2D((2, 2), name='p3')(c3)

    # Bottleneck
    b = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='b')(p3)
    b = layers.Dropout(0.5)(b)

    # Decoder
    u3 = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', name='u3')(b)
    u3 = layers.concatenate([u3, c3], name='u3b')
    cd3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='cd3')(u3)
    cd3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='cd3_2')(cd3)

    u2 = layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', name='u2')(cd3)
    u2 = layers.concatenate([u2, c2], name='u2b')
    cd2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='cd2')(u2)
    cd2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='cd2_2')(cd2)

    u1 = layers.Conv2DTranspose(16, (3,3), strides=(2,2), padding='same', name='u1')(cd2)
    u1 = layers.concatenate([u1, c1], name='u1b')
    cd1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='cd1')(u1)
    cd1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='cd1_2')(cd1)

    # Outputs at multiple scales
    output = layers.Conv2D(1, (1, 1), activation='sigmoid', name='output1')(cd1)

    model = models.Model(inputs, output)
    return model

def unet_deep(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='c1')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='c1_2')(c1)
    p1 = layers.MaxPooling2D((2, 2), name='p1')(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='c2')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='c2_2')(c2)
    p2 = layers.MaxPooling2D((2, 2), name='p2')(c2)
    
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='c3')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='c3_2')(c3)
    c3 = layers.Dropout(0.3)(c3)
    p3 = layers.MaxPooling2D((2, 2), name='p3')(c3)

    # Bottleneck
    b = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='b')(p3)
    b = layers.Dropout(0.5)(b)

    # Decoder
    u3 = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', name='u3')(b)
    u3 = layers.concatenate([u3, c3], name='u3b')
    cd3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='cd3')(u3)
    cd3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='cd3_2')(cd3)

    u2 = layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', name='u2')(cd3)
    u2 = layers.concatenate([u2, c2], name='u2b')
    cd2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='cd2')(u2)
    cd2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='cd2_2')(cd2)

    u1 = layers.Conv2DTranspose(16, (3,3), strides=(2,2), padding='same', name='u1')(cd2)
    u1 = layers.concatenate([u1, c1], name='u1b')
    cd1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='cd1')(u1)
    cd1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='cd1_2')(cd1)

    # Outputs at multiple scales
    output1 = layers.Conv2D(1, (1, 1), activation='sigmoid', name='output1')(cd1)
    output2 = layers.Conv2D(1, (1, 1), activation='sigmoid', name='output2')(cd2)
    output3 = layers.Conv2D(1, (1, 1), activation='sigmoid', name='output3')(cd3)

    model = models.Model(inputs=inputs, outputs=[output1, output2, output3])
    return model


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
    








