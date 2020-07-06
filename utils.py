from tensorflow.keras import models
import os
from pathlib import Path
import numpy as np
#from tqdm import tqdm
from matplotlib import image as img
from preparing_data import modifications
import pickle
from matplotlib import pyplot as plt
from tensorflow.keras.utils import Sequence

class My_Generator(Sequence):
    def __init__(self, images_filenames, labels,batch_size,image_data_generator:bool=False,load_gray=False):
        super(My_Generator)
        self.images_filenames = images_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.current_index = 0
        self.image_data_generator = image_data_generator
        self.load_gray = load_gray
        
    def __len__(self):
        return (np.ceil(len(self.images_filenames)/float(self.batch_size))).astype(np.int)
    
    def __getitem__(self,idx):
        batch_x = self.images_filenames[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]
        
        return np.array([ my_load_img(i,to_gray=self.load_gray)*(1/255.0) for i in batch_x]), np.array(batch_y)
    
    def __next__(self):
        if len(self) == self.current_index:
            raise StopIteration
        idx = self.current_index
        batch_x = self.images_filenames[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]
        
        if not self.image_data_generator:
            next_ =  np.array([ my_load_img(i,to_gray=self.load_gray)*(1/255.0) for i in batch_x]), np.array(batch_y)
        else:
            loaded_images = []
            for image_path in barch_x:
                temp_image = my_load_img(image_path,to_gray=self.load_gray)*(1/255.0)
                i = sample(range(len(modifications),1))
                mod = modifications[i]
                loaded_images.append(mod(temp_image)[0])
                del temp_image
            next_ =  np.array(loaded_images), np.array(batch_y)
        self.current_index +=1
        return next_
    

#todo: configure data path aan validation path
def train_model(model:models.Sequential,callbacks,X,y,train_size,val_size):
    assert train_size+ val_size <= len(y)
    train_half = int(train_size/2)
    val_half = int(val_size/2)
    X_train = np.concatenate([X[:train_half],X[-train_half:]])
    y_train = np.concatenate([y[:train_half],y[-train_half:]])
    
    val_top = train_half + val_half
    x_val = np.concatenate([X[train_half:val_top],X[-val_top:-train_half]])
    y_val = np.concatenate([y[train_half:val_top],y[-val_top:-train_half]])
    
    print('Training model with size ',(train_size,val_size),' ...')
    history = model.fit(X_train,
                        y_train,
                        epochs=60,
                        batch_size=20,
                        validation_data=(x_val, y_val),shuffle=True, callbacks=callbacks)
    print('Trained model')
    return history

from pathlib import Path
import numpy as np
from matplotlib import image as img
from random import sample

import tensorflow.keras as keras
from PIL import Image
def my_load_img(fname,size=(256,256),to_gray=False, normalize = False):
    img = Image.open(fname)
    img = img.resize(size)
    rbg_img = img.convert('RGB')
    array = keras.preprocessing.image.img_to_array(rbg_img) 
    if to_gray:
        array = rgbt2gray(array)
        array = np.reshape(array, size+(1,))
    if normalize:
        print("normalizing!!")
        return (array - 133.64513448)/48.22728248
    return array

def rgbt2gray(rgb):
    return np.dot(rgb[...,:3],[0.2989,0.5870,0.1140])

def load_data(data_path='augmented_data/train_set',size=4398, img_size=(256,256,3),cmap='rgb',load_all_data=False):
    '''
        Load from data_path size random images 
    '''
    

    sample_0_size = size - int(size/2)
    sample_1_size = size - sample_0_size
    
    negatives_path = data_path+'/0/'
    positives_path = data_path+'/1/'
    
    pn = Path(negatives_path)
    pp = Path(positives_path)
    p_list = list(pp.iterdir())
    n_list = list(pn.iterdir())
    p_len = len(p_list)
    n_len = len(n_list)

    if load_all_data:
        sample_0_size = n_len
        sample_1_size = p_len
    else:
        assert p_len+n_len >= size
        if n_len<sample_0_size:
            sample_1_size += sample_0_size - n_len
            sample_0_size = n_len


        if p_len<sample_1_size:
            sample_0_size+=sample_1_size-p_len
            sample_1_size = p_len

    p_list = sample(p_list,sample_1_size)
    n_list = sample(n_list,sample_0_size)
    X = np.zeros(shape=(size,img_size[0],img_size[1],img_size[2]))
    y = np.zeros(shape=(size))
    last_index= 0  
    
    for l,current_class in [(n_list,0),(p_list,1)]:
        for j in l:
            if j.is_file():
                if cmap == 'gray':
                    X[last_index] = rgbt2gray(img.imread(str(j)))*(1./255)
                elif cmap == 'rgb':
                    X[last_index] = my_load_img(str(j),img_size[:-1])
                y[last_index] = current_class
                last_index += 1

    

    return X,y
    
#todo: configure data path and validation path
def learning_curves(model,callbacks,train_path,validation_path='',train_size=4328, validation_size=123,sequence_of_data_sizes=[100,300,600,900,1200,1500,1800,2100,2400,2746]):
    X_train,y_train = load_data(train_path)        
            
    histories = []
    for size in sequence_of_data_sizes:
        history = train_model(model,callbacks,X,y,size,int(size/2))
        histories.append((size,history))
    try:
        with open('histories_learning_curves.pickle','wb') as file:
            pickle.dump(histories,file)
    except:
        with open('histories_learning_curves.pickle','wb') as file:
            pickle.dump([(size,x['acc'],x['val_acc']) for size,x in histories],file)
    plt.plot(sequence_of_data_sizes,[x for _,x,_ in histories],'b',label='training error')
    plt.plot(sequence_of_data_sizes,[y for _,_,y in histories],'r',label='validation error')
    plt.title('Training and validation error')
    plt.legend()
    plt.show()

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score,roc_auc_score, roc_curve, auc



import numpy as np

from itertools import groupby

def multiple_model_stats(y_preds, y_tests,models_names,xlim=(0,0.2),ylim=(0.8,1)):
    
    results = []
    
    fprs_tprs_aucs_modelName = [(0,0,0,model_name) for model_name in models_names]
    
    for i in range(len(models_names)):
        fpr, tpr,_ = roc_curve(y_tests[i],y_preds[i])
        _auc = auc(fpr,tpr)
        
        fprs_tprs_aucs_modelName[i] = (fpr,tpr,_auc,models_names[i])
    
    
    for k, g in groupby(fprs_tprs_aucs_modelName,lambda x: x[-1]):
        h = max(g,key=lambda a: a[2])
        results.append(h)
        
    plt.figure(1)
    if xlim is not None and ylim is not None:
        plt.xlim(xlim[0],xlim[1])
        plt.ylim(ylim[0],ylim[1])
    
    plt.plot([0,1],[0,1],'k--')
    
    for i in results:
        plt.plot(i[0],i[1], label="{} (area = {})".format(i[-1],i[2]))
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curves')
    plt.legend(loc='best')
    plt.show()

    
    

def stats(y_test,y_pred,model_name,history_file_name=None):
    
#     y_test = [float(i) for i in y_test]
#     print(y_test)

    if history_file_name is not None:
        p = Path(history_file_name)
        if not p.exists():
            histories = [(y_test,y_pred,model_name)]
            print('Creating history file')
            with open(history_file_name,'xb') as file:
                pickle.dump(histories,file)
                print("saving model predictions")
        else:
            with open(history_file_name,'rb') as file:
                histories = pickle.load(file)
            with open(history_file_name,'wb') as file:
                histories.append((y_test,y_pred,model_name))
                pickle.dump(histories,file)
                print("saving model predictions")

    print(model_name, ' stats!')
    print('confusionM ',confusion_matrix(y_test,np.round(y_pred)))
    print('acc ',accuracy_score(y_test,np.round(y_pred)))
    print('recall',recall_score(y_test,np.round(y_pred)))
    print('auc',roc_auc_score(y_test,np.round(y_pred)))
    
    
    y_test = [float(i) for i in y_test]
    fpr,tpr,threshold = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    
    plt.title("ROC curve")
    plt.plot(fpr,tpr,'b',label='AUC = %0.2f'%roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    
def load_dataset(img_dataset_path = 'train_test_cropped_clahe_meanblur/',load_gray=True,normalize=True,rescale=False):
    
    negatives_path = img_dataset_path + '0/'
    positives_path = img_dataset_path + '1/'

    negatives = os.listdir(negatives_path)
    positives = os.listdir(positives_path)
    data = []
    labels = []
    
    for image_path in negatives:
        image = my_load_img(negatives_path+image_path,to_gray=load_gray)
        data.append(image)
        labels.append(0)
    
    for image_path in positives:
        image = my_load_img(positives_path+image_path,to_gray=load_gray)
        data.append(image)
        labels.append(1)

    data = np.array(data)
    labels = np.array(labels)

    if normalize:
        mean_data = data.mean(axis=(0,1,2),keepdims=True)
        std_data = data.std(axis=(0,1,2),keepdims=True)
        print('mean: ', mean_data, ' shape: ',mean_data.shape)
        print('std: ', std_data)
        z = (data - mean_data)/std_data
    elif rescale:
        z =data* 1/255.0
    else:
        z = data
        
    
    del data
    del negatives
    del positives
    return z,labels


def load_images_paths(img_dataset_path = 'train_test_cropped_clahe_meanblur/'):
        
    negatives_path_folder = Path(img_dataset_path + '0/')
    positives_path_folder = Path(img_dataset_path + '1/')
    
    negatives = list(negatives_path_folder.iterdir())
    positives = list(positives_path_folder.iterdir())
    
    return negatives+positives,np.array([0 for _ in range(negatives)]+[1 for _ in range(positives)])

def load_images(images_paths,labels,to_gray,normalize):
    result = {}
    data = []
    for image_path,label in zip(images_paths,labels):
        image = my_load_img(image_path,to_gray=to_gray)
        result[image_path] = (image,label)
        data.append(image)
        
    if normalize:
        data = np.array(data)
        mean_data = data.mean(axis=(0,1,2),keepdims=True)
        std_data = data.std(axis=(0,1,2),keepdims=True)
        del data
        print('mean: ', mean_data, ' shape: ',mean_data.shape)
        print('std: ', std_data)
#         z = (data - mean_data)/std_data
        for k,v in result.items():
            result[k] = ((v[0]-mean_data)/std_data,v[1])
        
    return result