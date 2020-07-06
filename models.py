import numpy as np
from matplotlib import pyplot as plt
from utils import my_load_img, stats,multiple_model_stats
from preparing_data import modifications
import os
from tensorflow.keras import layers
from pathlib import Path

from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.regularizers import l2,l1,l1_l2
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from multiprocessing import Process

from vggGrayScale.convert_vgg_grayscale import load_grayscale_vgg_model

from load_model import buildInceptionModel

        

from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score,roc_auc_score, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class EnsembleClassifier:
    def __init__(self, build_cnn_function, train_cnn_function, classifiers):
        self.classifiers = classifiers
        self.build_cnn_model = build_cnn_function
        self.train_cnn_model = train_cnn_function
        
        self.cnn_model = None
        self.cropped_model = None
        
    def build_model(self,gray_scale_model=False):
        self.cnn_model = self.build_cnn_model(gray_scale_model)
        
    def train_model(self,x_train, y_train,x_val,y_val,batch_size,normalize=False,log_stats=True):
        self.train_cnn_model(self.cnn_model,x_train, y_train,x_val,y_val,batch_size,normalize=normalize)
        layer_dict = dict([(layer.name,layer) for layer in self.cnn_model.layers])

    #     x = layer_dict['flatten'].output
        x = self.cnn_model.layers[-2].output

        self.cropped_model = Model(self.cnn_model.input,x)

        cropped_model_train_output = self.cropped_model.predict(x_train)
        
        for sklearn_model, _ in self.classifiers:
            sklearn_model.fit(cropped_model_train_output,y_train)
        
        ensemble_model_predictions = self.predict(x_val)
        if log_stats:
            print("Ensemble predicted proba: ", ensemble_model_predictions)

            print("Validation ensemble Model result: ")
            stats(y_val, ensemble_model_predictions,'Ensemble Method')

    def predict(self,x_val):
        
        prediction = self.cnn_model.predict(x_val).ravel()
        cnn_selected_features = self.cropped_model.predict(x_val)
        for cl,_ in self.classifiers:
            prediction += cl.predict_proba(cnn_selected_features)[:,1]
        prediction = prediction/(len(self.classifiers)+1)
        
        return prediction
    
    def get_predictions(self, x_val):
        prediction = self.cnn_model.predict(x_val).ravel()
        yield prediction, "CNN"
        cnn_selected_features = self.cropped_model.predict(x_val)
        for cl, name in self.classifiers:
            yield cl.predict_proba(cnn_selected_features)[:,1], name
            
def build_model(gray_scale_model=True):
    if gray_scale_model:
        conv_base = load_grayscale_vgg_model()
    else:
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    
    for layer in conv_base.layers:
        if "block5" in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
    
    layer_dict = dict([(layer.name,layer) for layer in conv_base.layers])
    
    if gray_scale_model:
        x = layer_dict['256_block5_pool'].output
    else:
        x = layer_dict['block5_pool'].output

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    
    if gray_scale_model:
        x = layers.Dense(2048, activation='relu',name='dense_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(2048, activation='relu',name='dense_2' )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
    else:
        x = layers.Dense(300, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(300, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=conv_base.input,outputs=out)

    model.summary() 
    model.compile(loss='binary_crossentropy',
    optimizer=Adam(lr=1e-5,decay=5*1e-4,amsgrad=True),
    metrics=['accuracy'])
    
    return model
                    
# This method assume that x_val and x_train are images
def train_model_with_Keras_ImageDataGenerator(model:models.Sequential,x_train, y_train,x_val=None,y_val=None,batch_size=10,normalize=False, epochs=60, data_augmentation=True,save_model=False):
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10,verbose=0,mode='min')
#     mcp_save = ModelCheckpoint('./trained_models/vgg16/vgg16_{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5',save_best_only=True,monitor='val_loss',mode='min')
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=5*1e-2,patience=4, verbose=7, mode='min')
    
    callbacks = [earlyStopping]
#     if reduce_lr:
#         print("added reduce_lr callback")
#         callbacks.append(reduce_lr)
    
    
    TOTAL = len(x_train)
    
    print("Normalize: ", normalize)
#     print("Reduce lr: ", reduce_lr)
    if data_augmentation:
        print("Data Augmentation")
        train_datagen = ImageDataGenerator(
          featurewise_center=normalize,
          featurewise_std_normalization=normalize,
          rotation_range=10,
          width_shift_range=0.02,
          height_shift_range=0.02,
          shear_range=0.02,
          zoom_range=0.05,
          horizontal_flip=True,
          vertical_flip=True,
          fill_mode='nearest'
        )
    else:
        print("No Data Augmentation")
        train_datagen = ImageDataGenerator()
        
    
    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(featurewise_center=normalize,
      featurewise_std_normalization=normalize)
    
    if normalize:
        print('Normalizing training data')
        train_datagen.fit(x_train)
        if x_val is not None:
            test_datagen.fit(x_val)
    
    train_generator = train_datagen.flow(x_train,y_train,batch_size=batch_size)#, save_to_dir='out/')
    if x_val is not None:
        validation_generator = test_datagen.flow(x_val,y_val,batch_size=batch_size)

        history = model.fit_generator(train_generator,
        steps_per_epoch=2*TOTAL/batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
                                     )
        plot_history(history)
    else:
        history = model.fit_generator(train_generator,
        steps_per_epoch=2*TOTAL/batch_size,
        epochs=epochs,
        callbacks=callbacks)
        
    if save_model:
        model.save_weights('model_weights.h5')
        # Save the model architecture
        with open('model_architecture.json', 'x') as f:
            f.write(model.to_json())
    return model

def train_model_sklearn(model:models.Sequential,x_train, y_train,x_val,y_val,sklearn_model,sklearn_model_name,batch_size,normalize=False):
    
    model = train_model_with_Keras_ImageDataGenerator(model,x_train, y_train,x_val,y_val,batch_size,normalize=normalize)
    layer_dict = dict([(layer.name,layer) for layer in model.layers])
    
#     x = layer_dict['flatten'].output
    x = model.layers[-2].output

    from tensorflow.keras.models import Model
    cropped_model = Model(model.input,x)
    del model
    
    
    cropped_model_train_output = cropped_model.predict(x_train)
    del x_train
    
    sklearn_model.fit(cropped_model_train_output,y_train)
    
    cropped_model_val_output = cropped_model.predict(x_val)
    sklearn_model_predicted = sklearn_model.predict_proba(cropped_model_val_output)[:,1]
    print("SKlearn predicted proba: ", sklearn_model_predicted)
    
    print("Validation sklearn Model result: ")
    stats(y_val,sklearn_model_predicted,'CNN + '+sklearn_model_name)
    return sklearn_model,cropped_model
    
def plot_history(history):
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    

def _train_ensemble(classifiers,X_train,y_train,X_val,y_val,X_test,y_test,batch_size,gray_scale_model,normalize=False,mode='single',cnn_file_results_path = './histories/ensemble_norm.pickle'):
    model = EnsembleClassifier(build_model,train_model_with_Keras_ImageDataGenerator,classifiers)
    model.build_model(gray_scale_model)
    model.train_model(X_train, y_train,X_val,y_val,batch_size,normalize=normalize)
    
    if mode == 'single':
        y_pred = model.predict(X_test)

        path_p = Path(cnn_file_results_path)
        if not path_p.exists():
            print("Initialize log file")
            with open(cnn_file_results_path,'xb') as file:
                pickle.dump([],file)

        stats(y_test,y_pred,"Ensemble",cnn_file_results_path)
    else:
        
        for y_pred, model_name in model.get_predictions(X_test):
            results_path = '/histories/'+'norm_'+model_name+'.pickle'
            path_p = Path(results_path)
            if not path_p.exists():
                print("Initialize log file")
                with open(results_path,'xb') as file:
                    pickle.dump([],file)

            stats(y_test,y_pred,model_name,str(path_p))

def _train(build_model,train_model,X_train,y_train,X_val,y_val,X_test,y_test,batch_size,gray_scale_model,normalize=False,cnn_file_results_path = './histories/cnn_norm.pickle'):
    model = build_model(gray_scale_model)
    train_model(model,X_train,y_train,X_val,y_val,batch_size,normalize)
    
    y_pred = model.predict(X_test).ravel()
    
    path_p = Path(cnn_file_results_path)
    if not path_p.exists():
        print("Initialize log file")
        with open(cnn_file_results_path,'xb') as file:
            pickle.dump([],file)
        
    stats(y_test,y_pred,"CNN",cnn_file_results_path)
    return model
    
def _trainV2(build_model,train_model,X_train_paths,y_train,X_val_paths,y_val,X_test_paths,y_test,batch_size,gray_scale_model,normalize=False,cnn_file_results_path = './histories/cnn_norm.pickle'):
    
    dict_path_image = load_images(X_train_paths+X_val_paths+X_test_paths,y_train+y_val+y_test,gray_scale_model,normalize)
    X_train_ = [value for path,value in dict_path_image.items() if path in X_train_paths]
    X_val_ = [value for path,value in dict_path_image.items() if path in X_val_paths]
    X_test_ = [value for path,value in dict_path_image.items() if path in X_test_paths]
    
    del dict_path_image

    X_train = np.array(map(lambda x: x[0], X_train_))
    y_train = np.array(map(lambda x: x[1], X_train_))
    
    del X_train_
    
    X_val = np.array(map(lambda x: x[0], X_val_))
    y_val = np.array(map(lambda x: x[1], X_val_))
    
    del X_val_
    
    X_test = np.array(map(lambda x: x[0], X_test_))
    y_test = np.array(map(lambda x: x[1], X_test_))
    
    del X_test_
    
    model = build_model(gray_scale_model)
    train_model(model,X_train,y_train,X_val,y_val,batch_size,normalize)
    
    y_pred = model.predict(X_test).ravel()
    
    path_p = Path(cnn_file_results_path)
    if not path_p.exists():
        print("Initialize log file")
        with open(cnn_file_results_path,'xb') as file:
            pickle.dump([],file)
        
    stats(y_test,y_pred,"CNN",cnn_file_results_path)
    

from pathlib import Path
def _trainSK(build_model,train_model,X_train,y_train,X_val,y_val,X_test,y_test,sklearn_model,sklearn_model_name,batch_size,gray_scale_model,normalize,cnn_file_results_path = './histories/norm_'):
    print("TrainingModelSK")
    model = build_model(gray_scale_model)
    sklearn_model,cropped_model = train_model(model,X_train,y_train,X_val,y_val,sklearn_model,sklearn_model_name,batch_size,normalize)
    
    y_pred = sklearn_model.predict_proba(cropped_model.predict(X_test))[:,1]
    
    path = cnn_file_results_path+sklearn_model_name+".pickle"
    path_p = Path(path)
    if not path_p.exists():
        print("Initialize log file")
        with open(path,'xb') as file:
            pickle.dump([],file)
        
    stats(y_test,y_pred,"CNN + "+sklearn_model_name,path)
    
import random

def run_kfold_cnn(z,labels,build_cnn_model_function,train_function,batch_size,k=5,gray_scale_model=True,normalize=False):
    # z,labels = shuffle(z,labels)
    kf = KFold(n_splits=k)
    z_train,z_test,y_label_train,y_label_test = train_test_split(z,labels,random_state=42,test_size=0.15,shuffle=True)
    for train_index, test_index in kf.split(z_train):
        _execute_kfold(build_cnn_model_function,train_function,z_train,z_test,y_label_train,y_label_test,gray_scale_model=gray_scale_model,	train_index=train_index,test_index=test_index,batch_size=batch_size,normalize=normalize)
                                  						      											        
def run_kfold_svm(z,labels,build_cnn_model_function,train_function,batch_size,k=5,gray_scale_model=True,normalize=False):
    # z,labels = shuffle(z,labels)
    kf = KFold(n_splits=k)
    z_train,z_test,y_label_train,y_label_test = train_test_split(z,labels,random_state=42,test_size=0.15,shuffle=True)
    for train_index, test_index in kf.split(z_train):
        sklearn_model = SVC(C=2**5,gamma=2*1e-12,random_state=42,probability=True)
        _execute_kfold(build_cnn_model_function,train_function,z_train,z_test,y_label_train,y_label_test,gray_scale_model=gray_scale_model,train_index=train_index,test_index=test_index,sklearn_model=sklearn_model,sklearn_model_name="SVM",batch_size=batch_size,normalize=normalize)
        
def run_kfold_rf(z,labels,build_cnn_model_function,train_function,batch_size,k=5,gray_scale_model=True,normalize=False):
    # z,labels = shuffle(z,labels)
    kf = KFold(n_splits=k)
    z_train,z_test,y_label_train,y_label_test = train_test_split(z,labels,random_state=42,test_size=0.15,shuffle=True)
    for train_index, test_index in kf.split(z_train):
        sklearn_model = RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=42)
        _execute_kfold(build_cnn_model_function,train_function,z_train,z_test,y_label_train,y_label_test,gray_scale_model=gray_scale_model,train_index=train_index,test_index=test_index,sklearn_model=sklearn_model,sklearn_model_name="RF",batch_size=batch_size,normalize=normalize)

def run_kfold_classifiers(z,labels,batch_size,k=5,gray_scale_model=True,normalize=False):
    classifiers = [(SVC(C=2**5,gamma=2*1e-12,random_state=42,probability=True), "SVM"),(RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=42),"RF"),(KNeighborsClassifier(30), "KNN")]
    kf = KFold(n_splits=k)
    z_train,z_test,y_label_train,y_label_test = train_test_split(z,labels,random_state=42,test_size=0.15,shuffle=True)
    for train_index, test_index in kf.split(z_train):
        _execute_kfold(build_model,train_model_with_Keras_ImageDataGenerator,
	z_train,z_test,y_label_train,
	y_label_test,gray_scale_model=gray_scale_model,train_index=train_index,
	test_index=test_index,batch_size=batch_size,normalize=normalize)
        for sklearn_model, sklearn_model_name in classifiers:
            _execute_kfold(build_model,train_model_sklearn,	z_train,z_test,y_label_train,y_label_test,gray_scale_model=gray_scale_model,train_index=train_index,test_index=test_index,sklearn_model=sklearn_model,sklearn_model_name=sklearn_model_name,batch_size=batch_size,normalize=normalize)


def run_kfold_single_ensemble(z,labels,batch_size,k=5,gray_scale_model=True,normalize=False):
    classifiers = [(SVC(C=2**5,gamma=2*1e-12,random_state=42,probability=True), "SVM")]
#     classifiers = [SVC(C=2**5,gamma=2*1e-12,random_state=42,probability=True), RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=42),KNeighborsClassifier(30)]
    kf = KFold(n_splits=k)
    z_train,z_test,y_label_train,y_label_test = train_test_split(z,labels,random_state=42,test_size=0.15,shuffle=True)
    for train_index, test_index in kf.split(z_train):
        _execute_kfold(None,None,z_train,z_test,y_label_train,y_label_test,classifiers=classifiers,gray_scale_model=gray_scale_model,train_index=train_index,test_index=test_index,batch_size=batch_size,normalize=normalize)


def _execute_kfold(build_cnn_model_function,train_function,z_train,z_test,y_label_train,y_label_test,train_index,test_index,sklearn_model=None,sklearn_model_name="",batch_size=10,gray_scale_model=True,normalize=False,classifiers=None,mode='single'):
    X_train, y_train = z_train[train_index], y_label_train[train_index]
    X_val, y_val = z_train[test_index], y_label_train[test_index]
    val0 = len(y_val[y_val == 0])
    val1 = len(y_val[y_val == 1])
    print("train_len ->",len(y_train))
    print("class 0 validation: ",val0)
    print("class 1 validation: ",val1)
    print("random baseline: max/total: ", max(val0,val1)/(val0+val1))
    if classifiers is not None:
        p = Process(target=_train_ensemble,args=(classifiers,X_train,y_train,X_val,y_val,z_test,y_label_test,batch_size,gray_scale_model,normalize,mode))
    elif sklearn_model is None:
        p = Process(target=_train,args=(build_cnn_model_function,train_function,X_train,y_train,X_val,y_val,z_test,y_label_test,batch_size,gray_scale_model,normalize))
    else:
        p = Process(target=_trainSK,args=(build_cnn_model_function,train_function,X_train,y_train,X_val,y_val,z_test,y_label_test,sklearn_model,sklearn_model_name,batch_size,gray_scale_model,normalize))
    p.start()
    p.join()
