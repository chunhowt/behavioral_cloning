#**Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./examples/original.png "Original"
[image2]: ./examples/cropped.png "Cropped"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model is modified based on LeNet architecture, with the following layers.
- Conv2D of 3x3 filter and stride 1, with 10 feature maps + ReLU + Max pooling of 2x2 + Dropout of 0.5.
- Conv2D of 3x3 filter and stride 1, with 24 feature maps + ReLU + Max pooling of 2x2 + Dropout of 0.5.
- Fully connected layer of 120 nodes + ReLU
- Fully connected layer of 84 nodes + ReLU
- Fully connected layer of 10 nodes + ReLU
- Single node representing the steering angle.

Data is normalized using the Keras lambda layer.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets (20% validation size) to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
Batch size was 32 to fit into memory.
Epochs (20) were chosen until the training mse gets < 0.01.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the original udacity data, and then
flipping the images to adjust for the left turn bias. I also used the left / right camera with steering
adjustment to model recovering from the side.
For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

Before I begin training the model, I tried to analyze the udacity dataset to understand the training data. I figured out how I can downsample and crop the images that still keep all the information needed to make a choice between left / right turns. See analysis.ipynb for example of analysis I performed.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because our task essentially just need to differentiate between left turn / right turn and to the middle, which is relatively simple compared to most object recognition tasks. I tweaked the model a bit such as having a smaller convolution filter size since I already downsample the image and have slightly bigger # feature maps to get my training error to be < 0.010. I also added dropout layers after convolution layers so that my validation error is also < 0.010.

Next, I figured out the appropriate training data needed to make my model drive well on the simulator (see the section below).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture
My model is modified based on LeNet architecture, with the following layers.
- Conv2D of 3x3 filter and stride 1, with 10 feature maps + ReLU + Max pooling of 2x2 + Dropout of 0.5.
- Conv2D of 3x3 filter and stride 1, with 24 feature maps + ReLU + Max pooling of 2x2 + Dropout of 0.5.
- Fully connected layer of 120 nodes + ReLU
- Fully connected layer of 84 nodes + ReLU
- Fully connected layer of 10 nodes + ReLU
- Single node representing the steering angle.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I tried to record my own data. However, I noticed that each time I used my own data, even though I get good training and validation mse, my car will crash into the side. I suspect I am a bad driver simulating my own data. :) Most likely it is because I use keystroke which causes some noises whenever I collect data. Because of that I opted to use udacity provided data.

To augment the data sat, I also flipped images and angles thinking that this would allow the model to not be biased to the left-turn in track 1.

I also use the left and right camera to simulate recovering from the side with a steering adjustment of 0.2. I didn't use my own simulated images because of previous discovery that it doesn't yield good driving behavior.

After the collection process, I had 22050 number of data points. I then preprocessed this data by downsampling by 2 and then cropping away the top 25 pixels and bottom 10 pixels.
![Original image][image1]
![Cropped image][image2]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 which was chosen so that the training error and validation error stop decreasing (~0.01). I used an adam optimizer so that manually training the learning rate wasn't necessary.
