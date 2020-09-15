
#model for SAR data(sentinel 1) classification with tenserflow using optical data(sentinel 2)

import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import random
from random import shuffle
from skimage.transform import rotate
import scipy.ndimage as ndimage
from PIL import Image
import glob
from scipy import misc
import spectral
import imageio

import h5py
from keras.models import load_model
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def loaddata():
    data_path=os.path.join(os.getcwd())
    data=imageio.imread(os.path.join(data_path,'sar.png'))
    labels=imageio.imread(os.path.join(data_path,'optical.png'))
    return data,labels

def splitImage():
    #i = 0
    #j = 0
    data_path=os.path.join(os.getcwd())
    arr1 = np.zeros([256,256,3])
    arr2 = np.zeros([256,256,3])
    data = imageio.imread(os.path.join(data_path,'sar.png'))
    label = imageio.imread(os.path.join(data_path,'optical.png'))
    #print(arr.shape)
    data_path1 = os.path.join(os.getcwd(),'/sar')
    data_path2 = os.path.join(os.getcwd(),'/optical')
    num = 1

    for i in range(0,data.shape[0],256):
        for j in range(0,data.shape[1],256):
            arr1 = data[i:i+256,j:j+256]
            arr2 = label[i:i+256,j:j+256]
            #print(j+256)
            #print(sar1[i:i+256,j:j+256].shape)
            if (arr1.shape == (256,256,3) and arr2.shape == (256,256,3)):
                res1 = Image.fromarray(arr1, mode = 'RGB')
                res2 = Image.fromarray(arr2, mode = 'RGB')
                with open(os.path.join(data_path1, '{}.png'.format(num)), 'w') as f:
                    res1.save(f)
                with open(os.path.join(data_path2, '{}.png'.format(num)), 'w') as f:
                    res2.save(f)    
                num = num + 1

#converting an RGB image to a groundtruth classifier image i.e., x,y pixel location will have the class number
def label_creator(y):
    y_final = np.zeros([y.shape[0],y.shape[1]]) #y.shape[0]=height(2376), y.shape[1]=width(2793)
    #print(y_final.shape)
    cls = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i][j][1] == 0 :
                cls = 1
            elif y[i][j][1] ==104:
                cls = 2
            else:
                cls = 3    
            #if(j==5):
                #break
            """
            print("--------------------------------")
            print('maximum of y= ',max(y[i][j]))
            print("y[i,j]=",y[i][j])
            print("y[i][j][0]",y[i][j][0])
            print("y[i][j][1]",y[i][j][1])
            print('i',i)
            print('j',j)
            print("--------------------------------")
            """
            y_final[i][j] = cls
        #print(y_final[i][j])
        #if(i==5 and j==5):
           # break
        print(i)
    return y_final

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def AugmentData(X_train):
    for i in range(int(X_train.shape[0]/2)):
        patch = X_train[i,:,:,:]
        num = random.randint(0,2)
        if (num == 0):
            
            flipped_patch = np.flipud(patch)
        if (num == 1):
            
            flipped_patch = np.fliplr(patch)
        if (num == 2):
            
            no = random.randrange(-180,180,30)
            flipped_patch = ndimage.interpolation.rotate(patch, no,axes=(1, 0),
                                                               reshape=False, output=None, order=3, mode='constant', cval=0.0, prefilter=False)
    
    
    patch2 = flipped_patch
    X_train[i,:,:,:] = patch2
    
    return X_train


#Creating patches of 5x5x3 so that each patch contains 1 pixel at the centre, surrounded by zero padded pixels

def createPatches(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

#model training and construction

channels=3
windowsize=5
#testratio=0.25

x,y=loaddata()

with open(os.getcwd()+'Y_lable.npy','bw') as outfile: 
    np.save(outfile,y)
    
y=np.load(os.getcwd()+'Y_lable.npy')

for i in range(2376):
    print(y[i])

Xpatches,ypatches=createPatches(x,y,windowSize=windowsize)
X_train,X_test,Y_train,Y_test=train_test_split(Xpatches,ypatches,test_size=0.25)
with open(os.getcwd()+'X_train.npy','bw') as outfile:
    np.save(outfile,X_train)
with open(os.getcwd()+'Y_train.npy','bw') as outfile:
    np.save(outfile,Y_train)
with open(os.getcwd()+'X_test.npy','bw') as outfile:
    np.save(outfile,X_train)
with open(os.getcwd()+'Y_test.npy','bw') as outfile:
    np.save(outfile,Y_train)
X_train=np.load(os.getcwd()+'X_train.npy') 
Y_train=np.load(os.getcwd()+'Y_train.npy') 
X_test=np.load(os.getcwd()+'X_test.npy') 
Y_test=np.load(os.getcwd()+'Y_test.npy') 
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[3],X_train.shape[1],X_train.shape[2]))
Y_train=np_utils.to_categorical(Y_train)
input_shape=X_train[0].shape

"""
print(Xpatches.shape)
print(X_train.shape)
print(Y_train.shape)
"""

#input_shape = (256,256,3)
#WORKING BUT EACH EPOCH TAKES 3MINUTES
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2),dim_ordering="th"))
model.add(Conv2D(32, (3, 3), activation='relu',padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2),dim_ordering="th"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=2048,epochs=25)

model.save(os.getcwd()+'mymodel.h5')



def trainNN():
    data_path = os.path.join(os.getcwd(),'/sar')
    label_path = os.path.join(os.getcwd(),'/optical')
    input_images = os.listdir(data_path)
    label_images = os.listdir(label_path)
    for image in input_images:
        #print(image.type)
        X = imageio.imread(os.getcwd()+'/sar/{}'.format(image))
        y = imageio.imread(os.getcwd()+'/optical/{}'.format(image))
        #print(y)
        y = label_creator(y)
        #print(y)
        #X, y = createPatches(X, y, windowSize=windowSize)
        #X,y = oversampleWeakClasses(X, y)
        #X = AugmentData(X)
        X = np.reshape(X, (X.shape[0],X.shape[3], X.shape[1], X.shape[2]))
        #X = X.reshape(X.shape[0], 256, 256, 3)
        #X = X.astype('float32')
        #X /= 255
        print(X.shape)
        '''
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                print(y[i][j])
        '''
        y = np_utils.to_categorical(y)
        #print(min(y))
        #print(X[0].shape)
        model.fit(X, y, batch_size = 4096,epochs=50)
    model.save(os.getcwd()+'my_model_NN.h5')

trainNN()



def reports (X_test,y_test):
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    target_names = ['land', 'coconut trees', 'water bodies']

    
    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss =  score[0]*100
    Test_accuracy = score[1]*100
    
    return classification, confusion, Test_Loss, Test_accuracy

data = ndimage.imread(os.getcwd()+'/sar/rand.png')
labels = ndimage.imread(os.getcwd()+'/optical/rand.png')
X_test = data
y_test = label_creator(labels)

X_test,y_test= createPatches(X_test, y_test, windowSize=windowsize)

X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[3], X_test.shape[1], X_test.shape[2]))
y_test = np_utils.to_categorical(y_test)

model = load_model(os.getcwd()+'mymodel.h5')

os.chdir(os.getcwd())

classification, confusion, Test_loss, Test_accuracy = reports(X_test,y_test)
classification = str(classification)
confusion = str(confusion)
file_name = 'report2' +".txt"
with open(file_name, 'w') as x_file:
    x_file.write('{} Test loss (%)'.format(Test_loss))
    x_file.write('\n')
    x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{}'.format(classification))
    x_file.write('\n')
    x_file.write('{}'.format(confusion))

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

img = np.reshape(y_pred,(256,256))

pred_img = np.zeros((256,256,3))
for i in range(img.shape[0]):
    for j in range(img.shape[0]):
        if img[i][j] == 1:
            pred_img[i][j] = (0,0,0)
        elif img[i][j]==2:
            pred_img[i][j] = (10, 104, 10)
        #else:
         #   pred_img[i][j]=(0,0,0)
        elif img[i][j] == 3:
            pred_img[i][j] == (39,39,244)
res1 = Image.fromarray(pred_img, mode = 'RGB')
with open(os.path.join(os.getcwd(), 'pred.png'), 'w') as f:
    res1.save(f)

predict_image = spectral.imshow(classes = img.astype(int),figsize =(5,5))

def Patch(data,height_index,width_index):
    #transpose_array = data.transpose((2,0,1))
    #print transpose_array.shape
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    
    return patch

def avg_accuracy():
    total = 0
    loss = 0
    model = load_model(os.getcwd()+'my_model.h5')
    data_path = os.path.join(os.getcwd())
    label_path = os.path.join(os.getcwd())
    input_images = os.listdir(data_path)
    label_images = os.listdir(label_path)
    i = 0
    for image in input_images:
        #print(image.type)
        X_test = imageio.imread(os.getcwd()+'/sar/{}'.format(image))
        y_test = imageio.imread(os.getcwd()+'/optical/{}'.format(image))
        #print(y)
        y_test = label_creator(y_test)

        X_test,y_test= createPatches(X_test, y_test, windowSize=windowSize)

        X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[3], X_test.shape[1], X_test.shape[2]))
        y_test = np_utils.to_categorical(y_test)

        os.chdir(os.getcwd())

        classification, confusion, Test_loss, Test_accuracy = reports(X_test,y_test)
        if i==0:
            acc = Test_accuracy
            loss = Test_loss
        acc = float((acc+Test_accuracy)/2)
        loss = float((loss+Test_loss)/2)
        print(image)
        print('Average Test Accuracy: ' + str(acc))
        print('Average Test Loss: ' + str(loss))
        i+=1
    #acc = float(total/200)
    #avg_Loss = float(loss/200)
    return acc,loss

accuracy,loss = avg_accuracy()

accuracy

accuracy,loss












