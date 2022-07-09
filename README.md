# Detection-of-Pneumonia-from-X-ray-images

Pneumonia is an infectious disease of the lungs that mainly affects the pulmonary vasculature and causes the oxygen not to pass into the blood. Symptoms usually include a combination of cough, chest pain, high fever and difficulty breathing. Pneumonia is most often caused by a bacterial or viral infection and sometimes by autoimmune diseases. Diagnosis of inflammation is based on performing an X-ray of the chest and blood tests.
During this period of the corona virus (Covid-19) whose patients are in a higher risk group to develop pneumonia the importance of accuracy and decoding time of the test increases.
In light of this period there is a high increase in patients and therefore a quick and accurate solution is needed to decipher the X-rays. This project presents a solution for deciphering photographs using Deep Learning technology. This solution brings results almost in real time and with an accuracy of over 90% and thus will be used to reduce the load of radiologists who decode a lot of photographs of patients.
The mechanism will "learn" from a pre-existing X-ray database of healthy and sick people (the training set), using the CNN (Convolutional Neural Network) algorithm we "taught". He will know how to classify new x-rays that he has never encountered (test set) and determine if there is a presence of pneumonia in the photo.

# Tasks that this article deal with:

1) You need to design a deep network to solve the problem: the entry of the deep network is a picture and the starting point is the probability
That the picture represents a positive case of pneumonia. The probability is between 0 and 1 where 1 represents a certain case of
Pneumonia.
Performance will be measured on the test image set, which is completely separated from the training set. Good performance will be considered a chance
Error lower than 10% (i.e. Accuracy higher than 90%)

2) 1. For the network you selected, draw the PRECISION-RECALL performance graph, with each point in the graph calculated for
Different threshold level) in relation to the probability produced by the network (to decide on a positive example) i.e. with pneumonia (.
The threshold will be in the range 1.0 to 9.0 in jumps of 05.0. Also mark on the graph the SCORE-F points that will be calculated from each pair
PRECISION-RECALL values.
2. For which threshold was the highest SCORE-F value obtained?

3) 1. For the network you selected, try to improve Accuracy performance by making the following changes:
A. Addition of one layer according to your decision - the choice must be justified.
B. Addition of two layers according to your decision - the choice must be justified.
C. Examine five changes in the depth of existing layers and / or the number of convolution kernels (no additional layers)
- The choice must be justified.
2. Check the network performance with the following training algorithms Check the effect of the EPOCHS and LEARNING number
RATE per algorithm :
A. SGD algorithm
B. SGD algorithm with MOMENTUM and MOMENTUM NESTEROV.
C. ADAM Algorithm
D. RMSPROP algorithm
3. Examine the effect of the following changes:
E. DROPOUT - check the effect of changing the probability (this is the parameter of DROPOUT) if there is a layer .DROPOUT
F. Activate the STOPPING EARLY mechanism. Does the performance deteriorate beyond a certain EPOCHS number?
The convergence graph of the training process should be presented for each of the sections including LOSS TRAIN and VALIDATION
.VALIDATION ACCURACY -and TRAIN ACCURACY and, LOSS


In this project, a database from the Kaggle website was used which contains 5,863 real X-rays of pneumonia patients and healthy people.
The images in the database are divided into three folders:

Train - The training set is a database of examples (tagged images) used during the learning process to adjust parameters, including weights.

Validation - The control set, is a separate set used for control during the training process and allows to improve the classification level of the images by improving the hyperparameters that are not usually improved during the training process (such as Learning Rate). The set contains 8 images without pneumonia , And 8 with pneumonia.

Test - The test set, we were a set of untagged images that is used to actually test the system after the training process, and allows to characterize the quality of the network.

Principle of operation in brief: First, we train the system (Train). At this point the system learns how to identify images, by various parameters and weights them. We then move on to the test phase. At this point, we check image after image without tagging. Finally, we compare the control set (Validation) which actually criticizes the level of success of learning in the parameters learned. From here we come to a certain accuracy (Accuracy), in percent.

# Network training process:

![image](https://user-images.githubusercontent.com/105777016/178106746-89e61ff0-e5e4-4e96-87f4-e6c748802184.png)

From the accuracy graph it can be seen that in the 11th iteration we got a certain peak followed by a significant decrease in accuracy. It is hypothesized that this decrease is due to a change in the learning rate as a result of using an algorithm that changes the learning rate.
From the graph of losses it can be seen that the losses change, decrease to a local minimum and then increase which activates the mechanism of reducing the pace of learning.

The layers used:

● Conv2D - a two-dimensional convolution layer that produces a nucleus that is applied to the image at the entrance to the system to identify the image properties (edges, magnification, reduction and more). First we will define the N number of layers of the filters (we chose 32), then the kernel size Kernel Size, later we will need to select the size of the Stride window slider. We will set the image size (the image is two-dimensional so we will set 200X200, 1). We will then use a Relu activation layer (because it trains the network without significant penalty in the overall accuracy of the system) and finally we will use a Sigmoid activation layer that will give us a value of 0 or 1.

● Relu - is an Activation Function that resets the negative values and leaves the positive values as they are, the most common function due to its runtime efficiency and resource cost.

![image](https://user-images.githubusercontent.com/105777016/178106788-456b15c1-7cd5-45c0-bddd-755ed476712d.png)

● MaxPool2D - a layer that aims to reduce calculation costs and runtime by reducing the number of parameters that need to be studied.
This layer is done after convolution and actually reduces the Feature Map. The layer will take the maximum value from each Pool Size, we will make a 2X2 kernel with Stride 2 every time we use this operation.

![image](https://user-images.githubusercontent.com/105777016/178106804-4b06b463-cc55-47c0-8bb6-196abfc284ce.png)

● Flatten - A major step in CNN, we will use it when we want the output to be flat.
The intention is to turn a two-dimensional matrix into vector values that we can insert into the Classifier layer, this layer is located before the classification layer.

![image](https://user-images.githubusercontent.com/105777016/178106823-72048edd-a89f-41ef-a94b-bc7e96e17514.png)

● Dense - a layer in which each neuron in the layer receives input from all the neurons of the previous layer. The dense layer was found to be the most common layer in models. Behind the scenes the layer performs a product multiplication by a matrix, with the matrix values being verifiable weights.
The origin of the layer is a vector, which means that the role of the layer is to actually change the vector dimension.

![image](https://user-images.githubusercontent.com/105777016/178106842-29e654b2-be33-4cd6-b389-f79aa5bc3bc1.png)

# Data Analysis:

● Batch Size - Sets the size of the sample group used for training, for example the number of images in the control set.
● Epochs - the number of times we train with a different sample group (divides the set into groups in the amount of Epochs and compares to different sample groups).
● Loss Function - A loss function, is a function that maps values of one or more variables to an actual number that represents the "cost" of an event. This function is used to perform optimization, while want to keep its output as small as possible.

![image](https://user-images.githubusercontent.com/105777016/178106868-e20b72ee-7a0d-410c-a86b-1f865e602e79.png)


● Train / Val Accuracy - The accuracy of a learning system is determined by the proximity of the measurements of a particular quantity, to the true actual value of that quantity.

![image](https://user-images.githubusercontent.com/105777016/178106888-8eecddd2-1f87-4416-812a-04d4344ecbee.png)


Q1 Results:

Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Extracting all the files now...
Done!
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:40: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
Model: "sequential_12"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_62 (Conv2D)           (None, 200, 200, 32)      320       
_________________________________________________________________
max_pooling2d_62 (MaxPooling (None, 100, 100, 32)      0         
_________________________________________________________________
conv2d_63 (Conv2D)           (None, 100, 100, 32)      9248      
_________________________________________________________________
max_pooling2d_63 (MaxPooling (None, 50, 50, 32)        0         
_________________________________________________________________
conv2d_64 (Conv2D)           (None, 50, 50, 32)        9248      
_________________________________________________________________
max_pooling2d_64 (MaxPooling (None, 25, 25, 32)        0         
_________________________________________________________________
conv2d_65 (Conv2D)           (None, 25, 25, 32)        9248      
_________________________________________________________________
max_pooling2d_65 (MaxPooling (None, 13, 13, 32)        0         
_________________________________________________________________
flatten_12 (Flatten)         (None, 5408)              0         
_________________________________________________________________
dense_24 (Dense)             (None, 128)               692352    
_________________________________________________________________
dense_25 (Dense)             (None, 1)                 129       
=================================================================
Total params: 720,545
Trainable params: 720,545
Non-trainable params: 0
Loss of the model is -  0.29045939445495605
20/20 [==============================] - 1s 30ms/step - loss: 0.2905 - accuracy: 0.9038 - recall: 0.8060 - precision: 0.8141
Accuracy of the model is -  90.38461446762085 %


 # Recall-Precision Graph (Q2)
 
 Description of the performance of the basic model by the Recall-Precision graph where each point in the graph is calculated for a different threshold level in relation to the probability the network produces for deciding on a personal example.

![image](https://user-images.githubusercontent.com/105777016/178107125-28bc4710-50f1-4d04-9a4a-0358d9d2c5c7.png)

![image](https://user-images.githubusercontent.com/105777016/178107167-f14612b6-521a-49c9-934e-292f366d4265.png)

![image](https://user-images.githubusercontent.com/105777016/178107171-3587d3d0-2817-4d3d-a010-2ab920c34eda.png)

From the table  it can be seen that the highest harmonic mean (F1 -Score) is 0.8556825 and it occurs for a probability threshold level of 0.4 i.e. if we define any sample whose probability is positive (positive for pneumonia) will be above 0.4 we will get the best performance .



