#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/center.jpg "Center Image"
[image2]: ./report_images/clockwise.jpg "Clockwise Image"
[image3]: ./report_images/recover.jpg "Recover Image"
[image4]: ./report_images/track2.jpg "Track 2 Image"
[image5]: ./report_images/full.jpg "Full Image"
[image6]: ./report_images/cropped.jpg "Cropped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 with a lap recorded
* report_images folder with images of this report

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based 100% on the [Nvidia "End to End Learning for Self-Driving Cars" paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) and consists of:
- Cropping2D layer: this remove 50px of the top and 30px of the bottom of each image. The images input has a shape of 320x160 and the output 320x80

![alt text][image5]
![alt text][image6]

- Lambda layer: do the image normalization
- 3 Convolution2D layers: 3 convolution neural networks with 5x5 filter
- 2 Convolution2D layers: 2 convolution neural networks with 3x3 filter
- Flatten layer
- 5 Dense layers: fully conected layers with a output arrays of shape: 1164, 100, 50, 10 and 1

The Convolutional2D and Dense layers uses RELU as activation layers. 

####2. Attempts to reduce overfitting in the model

This model has no overfitting and the use of the Dropout layers not increase the validation results. This is probably beacause I am using a small dataset.

I tried to implement after the Convolutional2D layers, and then after the Dense layers. In both cases, the results were not good.

Besides, the car drive worst when I uses the Dropout layers.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 

I created 4 datasets for this purpose:
- 2 laps center lane driving, 
- Some tries recovering from the left and right sides of the road
- 1 lap clockwise
- 1 lap in the track2

But this 4 datasets not help in the training and validation step. The car drives unstable when I uses it. 

For sure, thew reason if this behavior is how I drived the car. I did it with the keywoard and the results are always an angle of 1 or -1. With this values, the training and validation works well, but in the practise, the car moves suddenly. Then I tried to use the mouse, but was impossible to drive a lap with a good result.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was created a model based on the Nvidia paper. Finally, this was the best option.

When I got a working model, I worked with different sources of the data and parameters (with or without flipping, using some of my data, modifying parameters, ... ), I tried some changes with this results:

- Increasing the batch the validation increase, but the car drives worst. Finally I used a batch of 64
- Removing the 2 first Dense layers, the overfitting decrease. But again, the car crives worst.
- I tried to remove the images with angle 0. The learning and validation works better, but the car drives worst.
- Adding dropout (after the Convolutional2D layers, and then aferter the Dense layers) the validations works better, but again the car drives worst. How I said before, this is probably beacause I am using a small dataset. When I uses my data, trhe car drives better, but not using only the Udacity data (beasides, the model does not have overfitting in this situation)
- I added the right and left images and flipping the images. Working on it, I finally uses a correction of 0.05 for this images and the result was a similar training and validation with better driving


####2. Final Model Architecture

The final model architecture (model.py lines 114-144) is described in the "An appropriate model architecture has been employed" section.


####3. Training and validation results of the final model

38568/38568 [==============================] - 97s - loss: 0.0128 - val_loss: 0.0103
Epoch 2/5
38568/38568 [==============================] - 89s - loss: 0.0095 - val_loss: 0.0099
Epoch 3/5
38568/38568 [==============================] - 89s - loss: 0.0090 - val_loss: 0.0102
Epoch 4/5
38568/38568 [==============================] - 89s - loss: 0.0088 - val_loss: 0.0093
Epoch 5/5
38568/38568 [==============================] - 89s - loss: 0.0088 - val_loss: 0.0102


####4. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving:

![alt text][image1]

Then, I recorded the vehicle doing 1 lap clockwise:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center:

![alt text][image3]

Then I repeated this process on track two in order to get more data points.

![alt text][image4]

How I had good results only with the Udacity images, I augmented the dataset with the right and left images and then flipping the images. With this action, the dataset increase 6 times with 38568 images to learning and validation process.