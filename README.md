# CNN-for-ROCK-ANALYSIS
CODE FOR ROCK ANALYSIS
1.	Building the CNN model

1.1	Import the Libraries:

In this step, we import Keras library and packages for building the Convolutional Neural Network model. First, we import the Sequential module which is used for initializing our model. Second is Convolutiona2D is the package that we’ll use for the first step of making the CNN that is the convolution step in which we add convolutional layers So here we are working with images which are only 2 dimensions there we use convolution2D( incase of video we use 3D). Then the next package is MaxPooling2D that’s a package that we’ll use to proceed to step to the pooling step that will add our pooling layers. The next package is Flatten this is the package that we will use in step 3 flattening in which we convert all the pooled feature maps that we created through convolution and max-pooling into the large feature vector that is then becoming the input of our fully connected layers. And then finally the last package which Dense package this is the package we use to add the fully connected layer.
<img width="569" height="166" alt="image" src="https://github.com/user-attachments/assets/168326f7-f925-4589-b0c3-8a16ddb0e437" />

1.2	Initialize our model:

In this step, we initialize our Convolutional Neural Network model to do that we use sequential modules.
<img width="592" height="78" alt="image" src="https://github.com/user-attachments/assets/8a8a338d-824d-4c8a-93e6-35d42c5e9177" />


1.3	Convolution Step:

In this step, we add a convolution step and this is the first step of CNN. Here in Convolution2D, we pass several arguments in this module. First arguments which we pass here feature detector (32, 3, 3) that’s mean we create 32 feature detectors of three by three dimensions and therefore our convolutional layers composed of 32 feature maps. Next arguments are input shape that’s the shape of input image on which we are going to apply feature detectors through the convolution operation in this arguments we convert all images into same formats here 64 and 64 are dimension and 3 is number of channels because we are working with colored images that’s why we use 3 number channels correspond to RGB ( incase of black & white image we use 1 number of channel). And the last argument which we pass here is relu activation function the relu function is used to replace the negate pixel value by 0(As we understand in mathematical part of CNN intuition)
<img width="635" height="44" alt="image" src="https://github.com/user-attachments/assets/550140cd-94af-404e-a62c-d684ce294f27" />


1.4	Pooling Step:

In this step, we add the pooling layer of our CNN model pooling layer is reducing the size of the feature map( the size of the feature map is divided by 2 when we apply the pooling layer). Here we passed only one argument which is pool size and define 2 by 2 dimension because we don’t want to lose any information about the image there we take the minimum size of the pool.
 
 <img width="579" height="67" alt="image" src="https://github.com/user-attachments/assets/eb871150-13f0-4784-bbfc-d8e2a7b74ad7" />


1.5	Add a second convolution layer and pooling layer:

In this step, we add a second convolution layer and pooling layer to make a mode more efficient and produce some good accuracy.
<img width="635" height="79" alt="image" src="https://github.com/user-attachments/assets/64d985a7-16e7-4387-b1a3-e2ef51596e06" />


1.6	Flatten Step:

In this step, we convert Max pooling 2D into a single long continuous linear vector to make an input node of fully connected layers.

<img width="318" height="68" alt="image" src="https://github.com/user-attachments/assets/9489af56-ee92-4c2f-b389-def4b1c37e35" />

1.7	Full Connection layer:

In this step, we use the Dense model to add a different layer. The parameter which we pass here first is ouptu_dim=6 which defines hidden layer=128 and last arguments Third is activation= relu here in hidden layer we use relu activation.
<img width="635" height="47" alt="image" src="https://github.com/user-attachments/assets/f081e076-2601-409c-950d-6dae34dfaf25" />


After adding a Hidden layer we add an output layer in full connection layer out_dim= 1 which means one output node here we use the sigmoid function because our target attribute has a binary class which is charnockite or geneiss that’s why we use sigmoid activation function.
<img width="635" height="23" alt="image" src="https://github.com/user-attachments/assets/2ffa16e3-6a68-45fc-aad8-e2cf8bd35ca7" />


1.8	Compiling the CNN

In this step we compile the CNN to do that we use the compile method and add several parameters the first parameter is optimizer = Adam here use the optimal number of weights. So for choosing the optimal number of weights, there are various algorithms of Stochastic Gradient Descent but very efficient one which is Adam so that’s why we use Adam optimizer here. The second parameter is loss this corresponds to loss function here we use binary_crossentropy because if we see target attribute our dataset which contains the binary value that’s why we choose the binary cross-entropy. The final parameter is metrics its list of metrics to be evaluated by the model and here we choose the accuracy metrics.

<img width="635" height="39" alt="image" src="https://github.com/user-attachments/assets/73ed94a8-4132-4181-906a-e5699b1aa27f" />

Part 2: Image Preprocessing

2.1	Import the library
#from keras.src.legacy.preprocessing.image import ImageDataGenerator
#from tf_keras.src.legacy.preprocessing.image import ImageDataGenerator

# 1. Import the modern utility
from keras.utils import image_dataset_from_directory
import tensorflow.keras.layers as layers
import tensorflow as tf
# 2. Load your dataset directly from a folder
train_set = image_dataset_from_directory(
"C:/Users/USER/AppData/Local/Programs/Python/Python311/Scripts/NEWTRAINSET/", 
image_size=(224, 224), # Set your target image size
batch_size=32	# Set your batch size
)
normalized_trainset = train_set.map(lambda x,y:(x/255.0,y)) augmentation_layers =tf.keras.Sequential(
[ layers.RandomRotation(0.2),
layers.RandomTranslation(0.2,0.2), layers.RandomZoom(0.2)
] )


test_set=image_dataset_from_directory(
"C:/Users/USER/AppData/Local/Programs/Python/Python311/Scripts/NEWTESTSET/", image_size=(224, 224), # Set your target image size
batch_size=32	# Set your batch size
)

normalized_testset = test_set.map(lambda x,y:(x/255.0,y)) augmentation_layers =tf.keras.Sequential(
[ layers.RandomRotation(0.2),
layers.RandomTranslation(0.2,0.2), layers.RandomZoom(0.2)
] )

Found 70 files belonging to 2 classes. 
Found 16 files belonging to 2 classes.




2.2	Training set & Test set

In this step, we import the images that we want to train. Remember our dataset contains 1000 images 8000 for train and 2000 for test and here first we import our images from the working directory that we want to train. Then target size here 64 by 64 that’s expected CNN which we defined earlier in Building the CNN part. Then batch size that the size of batch in which some random sample of our image will be included and that contains the number of images that will go through the CNN after which the weight will be updated. And finally, the class mode that the parameter indicating if your class is dependent variable is binary or has more than two categories and therefore since two-class here charnockite & geniess that’s why class mode is binary here that we use here.



2.3	Fit the CNN model

First arguments which we pass here is train set which contains 8000 images belongs to 2 class second is stepper epoch that is the number of the image we in our training set because remember all the observation of the training set pass through CNN during each epoch since we have 8000 images that’s why 8000 here. Then the number of epoch you want to choose to train our CNN here we take only 1 because our purpose is just to understand we suggest to take more than 50 for good accuracy. And then validation data that correspond to the test set on which we want to evaluate the performance of our CNN. And last are validation steps that correspond to the number of the image in our test set and that is 2000.

history1=classifier.fit(normalized_trainset,
steps_per_epoch=409, epochs=50,
validation_data=normalized_testset, validation_steps=107)


Here we go we obtain 80% for the training set and 81% for the test set.




test_loss, test_acc = classifier.evaluate(normalized_testset,	verbose=5)

evaluation = classifier.evaluate(normalized_testset)
print(f'Test Accuracy : {evaluation[1] * 100:.2f}%')


import numpy as np
from keras.preprocessing import image


test_image= image.load_img('C:/Users/USER/AppData/Local/Programs/Python/Python311/Scripts/PREDICT/rocksample4.jpg',target_size test_image=image.img_to_array(test_image)


# 2. Extract the class names BEFORE chaining prefetch
class_names = train_set.class_names
print(class_names)  # Outputs: ['CHARNOKITESTONE', 'gneissbest', 'granitebest', 'sandstonebest', 'shalebest', 'siltstonebest']

# 3. Create a class_indices dictionary manually if you need it
class_indices = {name: idx for idx, name in enumerate(class_names)}
print(class_indices)  # Outputs: {'CHARNOKITESTONE': 0, 'gneissbest': 1, 'granitebest': 2, 'sandstonebest': 3, 'shalebest': 4,


testset_names = test_set.class_names
print(testset_names)  # Outputs: ['charnockitetestbest', 'gneisstestbest', 'granitetestbest', 'sandstonetestbest', 'shalestone

# 3. Create a class_indices dictionary manually if you need it
class_indices = {name: idx for idx, name in enumerate(testset_names)}
print(class_indices)  # Outputs: {'charnockitetestbest': 0, 'gneisstestbest': 1, 'granitetestbest': 2, 'sandstonetestbest': 3,


result= classifier.predict(test_image)
#train_set.class_indices
#{  'CHARNOKITESTONE':0,'gneissbest':1,'granitebest':2,'sandstonebest':3,'shalebest':4,'siltsonebest':5}



<_PrefetchDataset element_spec=(TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>



final_array = test_image.reshape(1, 224, 224, 3)
 ind= np.argmax(classifier.predict(final_array)) 
print(ind)



print(result)



#code for chart creation

from matplotlib import pyplot as plt
from keras.utils import to_categorical

plt.plot(history1.history['accuracy'], label='accuracy')
plt.plot(history1.history['val_accuracy'], label = 'val_accuracy') plt.xlabel('Epoch')
plt.ylabel('Accuracy') 
plt.ylim([0.2, 1])
plt.legend(loc='upper left')
 
Part 3: Making a new prediction

3.1	Import the library

In this step, we import libraries the first library which we import here is numpy used for multidimensional array and the second is image package from Keras used for import the image from the working directory.

<img width="467" height="95" alt="image" src="https://github.com/user-attachments/assets/1c92e279-6025-448f-9281-e6770ef53d8a" />
3.2	Import the Image

In this step, we import our single image we don’t know either is charnockite image or geneiss image (like in mathematical part when we train images 0 and X and then make a prediction X). Here we pass two arguments first is image load and second is target size. Here we set target size 64 by 64 as expected our CNN model.

<img width="635" height="111" alt="image" src="https://github.com/user-attachments/assets/66179e5d-d618-46e7-b10a-c718e9816331" />
<img width="199" height="31" alt="image" src="https://github.com/user-attachments/assets/33130860-ff2b-45ff-a5f9-dd7f86aea8e9" />


3.3	Set the Dimension of image

In this step, we set the dimension of our image 2D (64, 64) into the 3D array(64, 64, 3) exactly as input shape that’s the architecture of input shape.

<img width="536" height="70" alt="image" src="https://github.com/user-attachments/assets/bc80b1ea-e834-448b-a0ff-1a9898e0bd66" />
<img width="227" height="28" alt="image" src="https://github.com/user-attachments/assets/79130489-d38b-4f89-b8cd-0204a9b518d6" />


3.4	Set the Dimension of image

We are working with the single image we directly make a prediction then will raise an error which expected convolution 2d input_1 to have 4 dimensions to do that we modify the test image into 4 dimensions.

<img width="606" height="68" alt="image" src="https://github.com/user-attachments/assets/3307b250-3152-4fca-b8fc-bf9a60d38542" />
<img width="259" height="24" alt="image" src="https://github.com/user-attachments/assets/e9dd847a-eb17-4260-ae6d-d97dfb0d1313" />


3.5	Make a prediction

In this step, we predict a single image

So here we go charnockite  corresponds to 0 and gneisstestbest corresponds to 1 so our image is a dog. Perfect so that’s mean the prediction of our CNN model for the first image of go charnockite   and gneisstestbest  is correct because the image contains charnockite
 

<img width="483" height="66" alt="image" src="https://github.com/user-attachments/assets/ec10fe60-6c43-4927-aff3-de45ebb4f2b8" />

 
 

