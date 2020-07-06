from pathlib import Path
# from tqdm import tqdm
from matplotlib import image as img
import numpy as np
from tensorflow.keras.utils import Sequence
from random import shuffle
from utils import my_load_img, My_Generator
from preparing_data import modifications
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D,concatenate,Flatten,SeparableConv2D,Activation
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import RMSprop, SGD,Adam,Adagrad,Adadelta
from tensorflow.keras.regularizers import l2,l1,l1_l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input
import tensorflow.keras as keras
import pickle
import multiprocessing as mp
from random import sample
from tensorflow.keras.models import Model


def rgbt2gray(rgb):
    return np.dot(rgb[...,:3],[0.2989,0.5870,0.1140])

def load_data(data_path='augmented_data/only_mass_train',size=4398, img_size=(256,256)):
    p = Path(data_path)
    X = np.zeros(shape=(size,img_size[0],img_size[1]))
    y = np.zeros(shape=(size))
    last_index= 0
    for i in p.iterdir():
        if i.is_dir():
            current_class = int(i.name)
            for j in tqdm(i.iterdir()):
                if j.is_file():
                    X[last_index] = rgbt2gray(img.imread(str(j)))
                    y[last_index] = current_class
                    last_index += 1
    return X,y

def buildInceptionModel():
    inputX = Input(shape=(256,256,1))
    x = BatchNormalization()(inputX)
    x = Conv2D(32, (3,3),padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2),padding='same',strides=(2,2))(x)
    x = Conv2D(32, (3,3),padding='same', activation='relu')(x)
    x = Conv2D(64, (3,3),padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2),padding='same',strides=(2,2))(x)
    x = Conv2D(64, (3,3),padding='same', activation='relu')(x)
    convAvg1 = Conv2D(192, (3,3),padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2),strides=(2,2),padding='same')(convAvg1)
    # _______________________ToAverage_______________________
    average1 = AveragePooling2D((1,1),strides=(2,2),padding='same')(convAvg1)

    #--------------------Inception6a-----------------------------------------------------------------------------------------------------------------
    inception_6a_x_branch1 = Conv2D(64, (1,1),padding='same', activation='relu')(x)
    # inception_6a_x_branch1 = keras.models.Model(inputs=input_inception_6a,outputs=inception_6a_x_branch1)

    inception_6a_x_branch2 = Conv2D(48, (1,1),padding='same', activation='relu')(x)
    inception_6a_x_branch2 = Conv2D(64, (3,3),padding='same', activation='relu')(inception_6a_x_branch2)
    # inception_6a_x_branch2 = keras.models.Model(inputs=input_inception_6a,outputs=inception_6a_x_branch2)

    inception_6a_x_branch3 = Conv2D(32, (1,1),padding='same', activation='relu')(x)
    inception_6a_x_branch3 = Conv2D(64, (3,3),padding='same', activation='relu')(inception_6a_x_branch3)
    inception_6a_x_branch3 = Conv2D(96, (3,3),padding='same', activation='relu')(inception_6a_x_branch3)
    # inception_6a_x_branch3 = keras.models.Model(inputs=input_inception_6a,outputs=inception_6a_x_branch3)

    inception_6a_x_branch4 = MaxPooling2D((2,2),padding='same',strides=(1,1))(x)
    inception_6a_x_branch4 = Conv2D(64, (1,1),padding='same', activation='relu')(inception_6a_x_branch4)
    # inception_6a_x_branch4 = keras.models.Model(inputs=input_inception_6a,outputs=inception_6a_x_branch4)

    inception_6a_x_concatenation = concatenate([inception_6a_x_branch4,inception_6a_x_branch3,inception_6a_x_branch2,inception_6a_x_branch1],axis=3)
    #--------------------Inception6a-----------------------------------------------------------------------------------------------------------------

    #--------------------Inception6a2-----------------------------------------------------------------------------------------------------------------
    inception_6a2_x_branch1 = Conv2D(64, (1,1),padding='same', activation='relu')(inception_6a_x_concatenation)
    # inception_6a_x_branch1 = keras.models.Model(inputs=input_inception_6a,outputs=inception_6a_x_branch1)

    inception_6a2_x_branch2 = Conv2D(48, (1,1),padding='same', activation='relu')(inception_6a_x_concatenation)
    inception_6a2_x_branch2 = Conv2D(64, (3,3),padding='same', activation='relu')(inception_6a2_x_branch2)
    # inception_6a_x_branch2 = keras.models.Model(inputs=input_inception_6a,outputs=inception_6a_x_branch2)

    inception_6a2_x_branch3 = Conv2D(32, (1,1),padding='same', activation='relu')(inception_6a_x_concatenation)
    inception_6a2_x_branch3 = Conv2D(64, (3,3),padding='same', activation='relu')(inception_6a2_x_branch3)
    inception_6a2_x_branch3 = Conv2D(96, (3,3),padding='same', activation='relu')(inception_6a2_x_branch3)
    # inception_6a_x_branch3 = keras.models.Model(inputs=input_inception_6a,outputs=inception_6a_x_branch3)

    inception_6a2_x_branch4 = MaxPooling2D((2,2),padding='same',strides=(1,1))(inception_6a_x_concatenation)
    inception_6a2_x_branch4 = Conv2D(64, (1,1),padding='same', activation='relu')(inception_6a2_x_branch4)
    # inception_6a_x_branch4 = keras.models.Model(inputs=input_inception_6a,outputs=inception_6a_x_branch4)

    inception_6a2_x_concatenation = concatenate([inception_6a2_x_branch4,inception_6a2_x_branch3,inception_6a2_x_branch2,inception_6a2_x_branch1],axis=3)
    #--------------------Inception6a2-----------------------------------------------------------------------------------------------------------------

    # _______________________ToAverage_______________________
    average2 = AveragePooling2D((1,1),strides=(1,1),padding='same')(inception_6a2_x_concatenation)


    #--------------------Inception6b-----------------------------------------------------------------------------------------------------------------
    inception_6b_branch1 = Conv2D(64,(3,3),padding='same',activation='relu')(inception_6a2_x_concatenation)

    inception_6b_branch2 = Conv2D(48,(1,1),padding='same',activation='relu')(inception_6a2_x_concatenation)
    inception_6b_branch2 = Conv2D(64,(3,3),padding='same',activation='relu')(inception_6b_branch2)
    inception_6b_branch2 = Conv2D(64,(3,3),padding='same',activation='relu')(inception_6b_branch2)

    inception_6b_branch3 = MaxPooling2D((3,3),strides=(1,1),padding='same')(inception_6a2_x_concatenation)

    inception_6b_concatenation = concatenate([inception_6b_branch1,inception_6b_branch2,inception_6b_branch3],axis=3)
    #--------------------Inception6b-----------------------------------------------------------------------------------------------------------------

    #--------------------Inception7a-----------------------------------------------------------------------------------------------------------------
    inception_7a_x_branch1 = Conv2D(64, (1,1),padding='same', activation='relu')(inception_6b_concatenation)
    # inception_7a_x_branch1 = keras.models.Model(inputs=input_inception_7a,outputs=inception_7a_x_branch1)

    inception_7a_x_branch2 = Conv2D(48, (1,1),padding='same', activation='relu')(inception_6b_concatenation)
    inception_7a_x_branch2 = Conv2D(64, (1,3),padding='same', activation='relu')(inception_7a_x_branch2)
    inception_7a_x_branch2 = Conv2D(64, (3,1),padding='same', activation='relu')(inception_7a_x_branch2)
    # inception_7a_x_branch2 = keras.models.Model(inputs=input_inception_7a,outputs=inception_7a_x_branch2)

    inception_7a_x_branch3 = Conv2D(32, (1,1),padding='same', activation='relu')(inception_6b_concatenation)
    inception_7a_x_branch3 = Conv2D(64, (3,1),padding='same', activation='relu')(inception_7a_x_branch3)
    inception_7a_x_branch3 = Conv2D(64, (1,3),padding='same', activation='relu')(inception_7a_x_branch3)
    inception_7a_x_branch3 = Conv2D(64, (3,1),padding='same', activation='relu')(inception_7a_x_branch3)
    inception_7a_x_branch3 = Conv2D(96, (1,3),padding='same', activation='relu')(inception_7a_x_branch3)
    # inception_7a_x_branch3 = keras.models.Model(inputs=input_inception_7a,outputs=inception_7a_x_branch3)

    inception_7a_x_branch4 = AveragePooling2D((2,2),padding='same',strides=(1,1))(inception_6b_concatenation)
    inception_7a_x_branch4 = Conv2D(64, (1,1),padding='same', activation='relu')(inception_7a_x_branch4)
    # inception_7a_x_branch4= keras.models.Model(inputs=input_inception_7a,outputs=inception_7a_x_branch4)

    inception_7a_x_concatenation = concatenate([inception_7a_x_branch4,inception_7a_x_branch3,inception_7a_x_branch2,inception_7a_x_branch1],axis=3)
    #--------------------Inception7a-----------------------------------------------------------------------------------------------------------------

    #--------------------Inception7a2-----------------------------------------------------------------------------------------------------------------
    inception_7a2_x_branch1 = Conv2D(64, (1,1),padding='same', activation='relu')(inception_7a_x_concatenation)
    # inception_7a_x_branch1 = keras.models.Model(inputs=input_inception_7a,outputs=inception_7a_x_branch1)

    inception_7a2_x_branch2 = Conv2D(48, (1,1),padding='same', activation='relu')(inception_7a_x_concatenation)
    inception_7a2_x_branch2 = Conv2D(64, (1,3),padding='same', activation='relu')(inception_7a2_x_branch2)
    inception_7a2_x_branch2 = Conv2D(64, (3,1),padding='same', activation='relu')(inception_7a2_x_branch2)
    # inception_7a_x_branch2 = keras.models.Model(inputs=input_inception_7a,outputs=inception_7a_x_branch2)

    inception_7a2_x_branch3 = Conv2D(32, (1,1),padding='same', activation='relu')(inception_7a_x_concatenation)
    inception_7a2_x_branch3 = Conv2D(64, (3,1),padding='same', activation='relu')(inception_7a2_x_branch3)
    inception_7a2_x_branch3 = Conv2D(64, (1,3),padding='same', activation='relu')(inception_7a2_x_branch3)
    inception_7a2_x_branch3 = Conv2D(64, (3,1),padding='same', activation='relu')(inception_7a2_x_branch3)
    inception_7a2_x_branch3 = Conv2D(96, (1,3),padding='same', activation='relu')(inception_7a2_x_branch3)
    # inception_7a_x_branch3 = keras.models.Model(inputs=input_inception_7a,outputs=inception_7a_x_branch3)

    inception_7a2_x_branch4 = AveragePooling2D((2,2),padding='same',strides=(1,1))(inception_7a_x_concatenation)
    inception_7a2_x_branch4 = Conv2D(64, (1,1),padding='same', activation='relu')(inception_7a2_x_branch4)
    # inception_7a_x_branch4= keras.models.Model(inputs=input_inception_7a,outputs=inception_7a_x_branch4)

    inception_7a2_x_concatenation = concatenate([inception_7a2_x_branch4,inception_7a2_x_branch3,inception_7a2_x_branch2,inception_7a2_x_branch1],axis=3)
    #--------------------Inception7a2-----------------------------------------------------------------------------------------------------------------

    # _______________________ToAverage_______________________
    average3 = AveragePooling2D((1,1),strides=(1,1),padding='same')(inception_7a2_x_concatenation)


    #--------------------Inception7b-----------------------------------------------------------------------------------------------------------------
    inception_7b_branch1 = Conv2D(64,(3,3),padding='same',activation='relu')(inception_7a2_x_concatenation)

    inception_7b_branch2 = Conv2D(48,(1,1),padding='same',activation='relu')(inception_7a2_x_concatenation)
    inception_7b_branch2 = Conv2D(64,(3,3),padding='same',activation='relu')(inception_7b_branch2)
    inception_7b_branch2 = Conv2D(64,(3,3),padding='same',activation='relu')(inception_7b_branch2)

    inception_7b_branch3 = MaxPooling2D((3,3),strides=(1,1),padding='same')(inception_7a2_x_concatenation)

    inception_7b_concatenation = concatenate([inception_7b_branch1,inception_7b_branch2,inception_7b_branch3],axis=3)
    #--------------------Inception7b-----------------------------------------------------------------------------------------------------------------
    
    output = AveragePooling2D((1,1),strides=(1,1),padding='same')(inception_7b_concatenation)
    output = concatenate([output,average1,average2,average3],axis=3)
    
    output = GlobalAveragePooling2D()(output)
    output = Flatten()(output)
#     output = Dense(64, activation='relu',kernel_regularizer=l2(1e-6))(output)
#     output = Dense(32, activation='relu',kernel_regularizer=l2(1e-6))(output)
    output = Dense(1, activation='sigmoid')(output)
    model = keras.models.Model(inputX,output)

    model.compile(optimizer=Adam(1e-2),
                  loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model

def buildSepcvn():
    model = models.Sequential()
    
    model.add(SeparableConv2D(32,(3,3),padding='same',input_shape=(256,256,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    
    model.add(SeparableConv2D(64,(3,3),padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(64,(3,3),padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    
    model.add(SeparableConv2D(128,(3,3),padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(128,(3,3),padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(128,(3,3),padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
#     model.add(Dense(512, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
    
#     model.add(Dense(128, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
    optimizer=Adadelta(lr=1e-2),
    metrics=['accuracy'])
    
    model.summary()
    
    return model