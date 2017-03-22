**Behavioral Cloning**
*Creig Cavanaugh*
*March 2017*

**Behavioral Cloning Project**

The goals / steps of this project are the following:
- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/powerful_cnn.png "Model Visualization"
[image2]: ./images/center.jpg "Center Image"
[image3]: ./images/center_2017_03_06_20_52_26_665.jpg "Recovery Image"
[image4]: ./images/center_2017_03_06_20_52_28_700.jpg "Recovery Image"
[image5]: ./images/center_2017_03_06_20_52_30_717.jpg "Recovery Image"
[image6]: ./images/image_center_normal.jpg "Normal Image"
[image7]: ./images/image_center_flipped.jpg "Flipped Image"
[image8]: ./images/figure.png "MSE Loss Example"

**Rubric Points**
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
**Files Submitted & Code Quality**

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
- model.py containing the script to create and train the model
- drive.py for driving the car in autonomous mode
- model.h5 containing a trained convolution neural network 
- writeup_report.md or writeup_report.pdf summarizing the results
- video.mp4 shows the simulator in autonomous mode using the trained CNN

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5 convolutional layers and 3 fully connected layers. The first three convolutional layers use a 2x2 stride and 5x5 kernel (model.py lines 68-70) .  The last two convolutional layers have a 3x3 kernel size and no stride (model.py lines 71-72). 

The model includes RELU layers on each convolutional layer to introduce nonlinearity (code lines 68-72), and the data is normalized in the model using a Keras lambda layer (code line 64). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I noticed increasing the Epochs above 4 seemed to ovefit the network, so my final network is trained using 3 Epochs. 


####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 81).  The correction factor applied when using the left or right offset images was manually tuned to 0.25 (model.py line 18).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and right lane driving on the second track. I also made sure to drive the courses both clockwise and counter clockwise. The total number of images available to train and test the network was about 95,000, which included the center view, as well as the left and right angled views and image augmentation.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the Keras framework to implement and modify the architecture based on testing results using the simulator in autonomous mode.

My first step was to use a convolution neural network model similar to the LeNet architecture, since I have used this architecture for image classification.  

Although the LeNet style architecture did work to some extent, I found the convolutional network architecture defined in the Nvidia paper titled "End to End Learning for Self-Driving Cars (2016)" helped the car stay in the center of the lane better and allowed it to complete multiple laps on the course.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I used more images from the recordings, and used less Epochs.  I also recorded data from both tracks, and with the car driven in clockwise and counter-clockwise directions.

Then I added image cropping to reduce the size and complexity of the images used to train the network, in order to have the network train on the most relevant imagery.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, specifically the turn just prior to the bridge, and the turn that forks to the dirt path. To improve the driving behavior in these cases, I recorded additional data at those turns, and adjusted the correction factor used for left and right camera angle correction. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 64-77) consisted of a convolution neural network with the following layers and layer sizes:

1) Input 160x320x3
2) Normalization Layer (Lambda) (160x320x3)
3) Cropping (Output: 70x320x3)
4) Convolutional (2x2 stride, 5x5 kernel, relu activation) (33x158x24)
5) Convolutional (2x2 stride, 5x5 kernel, relu activation) (15x77x36)
6) Convolutional (2x2 stride, 5x5 kernel, relu activation) (6x37x48)
7) Convolutional (3x3 kernel, relu activation) (4x35x64)
8) Convolutional (3x3 kernel, relu activation) (2x33x64)
9) Flatten (4224)
10) Fully Connected Layer (100)  
11) Fully Connected Layer (50)  
12) Fully Connected Layer (10)  
13) Fully Connected Layer (1) 

Here is a visualization of the architecture used, which comes from the Nvidia paper titled "End to End Learning for Self-Driving Cars (2016)"

![CNN Architecture][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I also recorded two laps on track one driving in the opposite direction (clockwise). 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover back to the center of the lane.  These images show what a recovery looks like starting on the far left of the road back to the center of the lane:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would help reduce the network's bias to turn to the left since the first track consists primarily of left turns when driven counter-clockwise. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I also experimented with adding images to the data set that have processed with a bitwise-not, with the intent to invert the brightness of the image, but testing did not show any significant improvement to the mean squared error.  

After the collection process, I had 47,589 number of data points. With data augmentation that number increased to 95,178 images.  I added a random filter to limit the number of data points actually used. I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by any additional epochs resulted in an increase of the mean squared error of the validation set. I used an adam optimizer so that manually training the learning rate wasn't necessary.
