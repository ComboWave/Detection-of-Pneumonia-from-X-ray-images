# Import Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import cv2 # OpenCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
import tensorflow as tf
import os
from google.colab import drive
from zipfile import ZipFile


# Get data

LABELS = ('PNEUMONIA','NORMAL')
IMG_SIZE=200
def get_data(data_dir):
    data = []
    for label in LABELS:
        path = os.path.join(data_dir, label)
        class_num = LABELS.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE)) # Reshaping Images
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)
    
    
"""## Loading Dataset"""

drive.mount('/content/drive')
file_name="/content/drive/MyDrive/Electrical Engineering | Roy Ben Avraham | 2017/deep learning/chest_xray.zip"
with ZipFile (file_name,'r') as zip:
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')
    
    
train_raw = get_data('../content/chest_xray/train')
test_raw = get_data('../content/chest_xray/test')
val_raw = get_data('../content/chest_xray/val')


#Splitting data

X_train = []
y_train = []

X_val = []
y_val = []

X_test = []
y_test = []

for image, label in train_raw:
    X_train.append(image)
    y_train.append(label)
    
for image, label in val_raw:
    X_val.append(image)
    y_val.append(label)
    
for image, label in test_raw:
    X_test.append(image)
    y_test.append(label)
    
    
# Normalize the data
X_train = np.array(X_train) / 255
X_val = np.array(X_val) / 255
X_test = np.array(X_test) / 255


# Resize data for deep learning 
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array(y_train)

X_val = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_val = np.array(y_val)

X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array(y_test)


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
thresholds_set = list(np.arange(0.1,0.91,0.05))

accuracy = tf.keras.metrics.BinaryAccuracy(name='accuracy',
                                           dtype=None,
                                           threshold=0.5
                                           )

recall = tf.keras.metrics.Recall(thresholds=thresholds_set,
                                 top_k=None,
                                 class_id=None,
                                 name='recall',
                                 dtype=None
                                 )

precision = tf.keras.metrics.Precision(thresholds=thresholds_set,
                                       top_k=None,
                                       class_id=None,
                                       name='precision',
                                       dtype=None
                                       )
np.random.seed = 16
NUM_EPOCHS = 16

model = Sequential()

model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (IMG_SIZE,IMG_SIZE,1)))
#model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
#model.add(Dropout(0.4))

model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
#model.add(Dropout(0.1))
#model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
#model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
#model.add(Dropout(0.3))
#model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

#model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
#model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
#model.add(Conv2D(192 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())
#model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))


## Q3.1.3
  # option 1:
    # model.add(Conv2D(32 , (2,2) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (img_size,img_size,1)))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    # model.add(Conv2D(32 , (2,2) , strides = 1 , padding = 'same' , activation = 'relu'))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    # model.add(Conv2D(32 , (2,2) , strides = 1 , padding = 'same' , activation = 'relu'))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    # model.add(Conv2D(32 , (2,2) , strides = 1 , padding = 'same' , activation = 'relu'))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

  # option 2:
    # model.add(Conv2D(32 , (4,4) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (img_size,img_size,1)))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    # model.add(Conv2D(32 , (4,4) , strides = 1 , padding = 'same' , activation = 'relu'))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    # model.add(Conv2D(32 , (4,4) , strides = 1 , padding = 'same' , activation = 'relu'))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    # model.add(Conv2D(32 , (4,4) , strides = 1 , padding = 'same' , activation = 'relu'))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

  # option 3:
    # model.add(Dense(units = 32 , activation = 'relu'))

  # option 4:
    # model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (img_size,img_size,1)))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    # model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    # model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    # model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

  # option 5:
    # model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (img_size,img_size,1)))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    # model.add(Conv2D(32 , (4,4) , strides = 1 , padding = 'same' , activation = 'relu'))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    # model.add(Conv2D(32 , (5,5) , strides = 1 , padding = 'same' , activation = 'relu'))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    # model.add(Conv2D(32 , (6,6) , strides = 1 , padding = 'same' , activation = 'relu'))
    # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))


## Q3.1.1
  # depth_of_conv = 32 # 64/128
  # model.add(Conv2D(depth_of_conv , (3,3) , strides = 1 , padding = 'same' , activation = 'relu')) 
  # model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))


## Q3.1.2
#depth_of_conv_1 = 32 # 64
#depth_of_conv_2 = 32 # 64/128
#model.add(Conv2D(depth_of_conv_1 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu')) 
#model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
#model.add(Conv2D(depth_of_conv_2 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu')) 
#model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

#Fully connected
model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
#model.add(Dropout(0.4))
model.add(Dense(units = 1 , activation = 'sigmoid'))

model.compile(optimizer = optimizers.Adam() , loss = 'binary_crossentropy' , metrics = ['accuracy', recall, precision])

## Q3.2
  # l_r = 0.001 # 0.05/0.0001
  # epochs_num = 15 # 5/25

## Q3.2.1 + Q3.2.2
  # m_m = 0.0 # 0.4/0.8
  # n_n = False # True
  # model.compile(optimizer = optimizers.SGD(lr = l_r,momentum = m_m,nesterov = n_n) , loss = 'binary_crossentropy' , metrics=[accuracy, recall, precision])

## Q3.2.3
  # model.compile(optimizer = optimizers.Adam(lr = l_r) , loss = 'binary_crossentropy' , metrics=[accuracy, recall, precision])

## Q3.2.4
  # model.compile(optimizer = optimizers.RMSProp(lr = l_r) , loss = 'binary_crossentropy' , metrics=[accuracy, recall, precision])


model.summary()

# Reduce learning rate when a metric has stopped improving
learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy',
                                            patience = 2,
                                            verbose=1,
                                            factor=0.3,
                                            min_lr=0.000001)

## Q3.3.6
callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy',mode='max', patience=2,restore_best_weights=True)

history = model.fit(datagen.flow(X_train, y_train),
                    epochs = NUM_EPOCHS,
                    validation_data = datagen.flow(X_val, y_val),
                    callbacks = [learning_rate_reduction, callback]
                   )
                   
                   
print("Loss of the model is - " , model.evaluate(X_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(X_test,y_test)[1]*100,"%")


epochs = [i for i in range(NUM_EPOCHS)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()


## Q2.1
  # Calculating F score and Precision-Recall Graph 

#f_score =[]
#plt.plot(recall.result(),precision.result())
#plt.xlabel("recall")
#plt.ylabel("precision")
#plt.grid(which='both')
#plt.scatter(recall.result(),precision.result())
#for i in range(np.size(thresholds_set)):
  #plt.annotate(str(i+1),(recall.result()[i],precision.result()[i]))
  #f_score.append(2*(recall.result().numpy()[i]*precision.result().numpy()[i])/(recall.result().numpy()[i]+precision.result().numpy()[i]))
  #print('recall-', recall.result()[i], 'precision-', precision.result()[i], 'f-score-', f_score[i])
#plt.show()
