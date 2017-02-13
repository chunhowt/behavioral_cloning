#**Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./examples/original.png "Original"
[image2]: ./examples/cropped.png "Cropped"
[image3]: ./examples/recovery_curve.jpg "Recovery curve 1"
[image4]: ./examples/recovery_curve2.jpg "Recovery curve 2"
[image5]: ./examples/downsample.png "Downsample"

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
* run_v2.mp4 video showing trial run on the track 1 simulation.

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model is modified based on LeNet architecture, see the following section with the final details.

Data is normalized using the Keras lambda layer.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets (20% validation size) to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
Batch size was 32 to fit into memory.
Epochs (30) were chosen until the training mse gets < 0.01.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the original udacity data, and then
flipping the images to adjust for the left turn bias. I also used the left / right camera with steering
adjustment to model recovering from the side.
I also collect some of my own recovery data for a 2 curves after the bridge where my car has trouble with.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

Before I begin training the model, I tried to analyze the udacity dataset to understand the training data.
I figured out how I can downsample (by 2) and crop the images (cut the top 60 pixels and bottom 20 pixels
of original image) while still keeping all the information needed to make a choice between left / right turns.
This actually makes my training faster since the input size is smaller. See analysis.ipynb for example
of analysis I performed.

My first step was to use a convolution neural network model similar to the LeNet. I thought this
model might be appropriate because our task essentially just need to differentiate between left
turn / right turn and to the middle, which is relatively simple compared to most object recognition tasks.
I tweaked the model a bit such as having a smaller convolution filter size since I already downsample the
image and have slightly bigger # feature maps to get my training error to be < 0.010. I also added dropout
layers after convolution layers so that my validation error is closer to training error and not increase
after a lot of epochs. See the model section on the final architecture.

Next, I figured out the appropriate training data needed to make my model drive well on the simulator
(see the section below on the training data).

At the end of the process, the vehicle is able to drive autonomously around track 1 without leaving the
road. It is able to drive around track 2 a bit but ultimately still crashes to the side.

Throughout the project, I have the sense that the training data is the single most important factor
deciding the success of the final model (compared to model architecture and hyperparameters). I have to
iterate by adding more data whenever my car has trouble with certain part of the track. This is slightly
unsatisfactory though because:
1. validation error doesn't really correlate well with the final performance given the potential noise
in the training / validation data collection.
2. optimizing the model / training data based on the simulator seems to be peeking at the final test set
which isn't a good practice for machine learning.

####2. Final Model Architecture
My model is modified based on LeNet architecture, with the following layers.
- Conv2D of 3x3 filter and stride 1, with 10 feature maps + ReLU + Max pooling of 2x2 + Dropout of 0.5.
- Conv2D of 3x3 filter and stride 1, with 24 feature maps + ReLU + Max pooling of 2x2 + Dropout of 0.5.
- Fully connected layer of 256 nodes + ReLU
- Fully connected layer of 64 nodes + ReLU
- Fully connected layer of 16 nodes + ReLU
- Single node representing the steering angle.

See line 71 - 86 of model.py.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I initially tried to record my own data. However, I noticed that
each time I used my own data, even though I get good training and validation mse, my car will crash
into the side. I suspected that I am a bad driver (or maybe a bad gamer :) Most likely it is because
I use keystroke which causes some noises whenever I collect data as the steering angle is not
continuous. Because of that I opted to use udacity provided data.

To augment the data sat, I also flipped images and angles thinking that this would allow the model
to not be biased to having too many left-turn in track 1. While my analysis in analysis.ipynb shows that
the steering angle distribution isn't quite biased to left or right, training a model without flipping
the image seems to cause my car to have a bias towards doing left turn.

I also use the left and right camera to simulate recovering from the side with a steering adjustment of 0.2.

After training, I noticed that my car still have problem with the two curves after the bridge. To fix that,
I manually collect more recovery data targeting those two curves. Example images are:

![Recovery image 1][image3]
![Recovery image 2][image4]

After the collection and data augmentation process, I had 25806 number of data points.
I then preprocessed this data by downsampling by 2 and then cropping away the top 30 pixels and
bottom 10 pixels. I did this so that my input image is smaller allowing training to be faster, and then
removing useless information such as sky and ground since the camera on the car is fixed.

Original image:

![Original image][image1]

Downsample image:

![Downsample image][image5]

Cropped image:

![Cropped image][image2]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was
over or under fitting. The ideal number of epochs was 30 which was chosen so that the training error and
validation error stop decreasing much (~0.01). I used an adam optimizer so that manually training the
learning rate wasn't necessary.
