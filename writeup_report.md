# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* model_no_crop.h5 I spent most of my time training an architecture with no CroppingLayer because of an issue with Keras. This model drives the car much smoother but fails in bend 2 
* run1.mp4 a video of a successful lap using model.h5
* writeup_report.md summarizing the results, the file you are reading now

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64. 

After the final convolution layer, it flattens the previous output, adds dropout and adds 4 dense layers. The dense layers use ReLU activation to introduce nonlinearity.

Prior to going through the network, the images are normalized, and cropped using two lambda functions. Keras's built-in Cropping2D layer did not work on the deployment machine, which is why I implemented a Lambda function to achieve the s ame result.

#### 2. Attempts to reduce overfitting in the model

As mentioned before, I used a dropout layer (with 50% retention rate), to avoid overfitting. I introduced it rather early in the model, but I do remember it helping with overfitting.

In addition, I found it useful to reset the start learning rate of the optimizer when finetuning the model (see next section).

Also, when introducing new data to help the car in tricky situations (e.g., give new training data based on a tricky bend), I found it a must to train the network with all data. The model seems to learn a new function when only giving it this "exception handling training data". 

I used scikits's built-in function to split my data between train and validation sets. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, however, I did use a start learning of 0.001. In addition, when fine-tuning successful models, I found it useful to reset the the start learning rate to 0.001 to avoid overfitting.

I used 10 epochs for the initial model and 4 epochs for refined versions.

I played around with batch sizes as well and found that the largest number I tried (512) achieved significant better results.

I used 33% of all data as validation data.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture:
____________________________________________________________________________________________________
| Layer (type) |                    Output Shape   |       Num Params  |    Connected to |
|:---------------------:|:---------------------:|:---------------------:|:---------------------:|
| lambda_1 (Lambda: normalization)      |          (None, 160, 320, 3)|   0           | lambda_input_1[0][0] |
| lambda_2 (Lambda: custom cropper)      |          (None, 90, 320, 3) |   0           | lambda_1[0][0] |
| convolution2d_1  (Convolution2D) | (None, 43, 158, 24) |  1824      |  lambda_2[0][0] |
| convolution2d_2 (Convolution2D)  | (None, 20, 77, 36)    | 21636       | convolution2d_1[0][0] |
| convolution2d_3 (Convolution2D)  | (None, 8, 37, 48)     | 43248       | convolution2d_2[0][0] |
| convolution2d_4 (Convolution2D)  | (None, 6, 35, 64)     | 27712       | convolution2d_3[0][0] |
| convolution2d_5 (Convolution2D)  | (None, 4, 33, 64)     | 36928       | convolution2d_4[0][0] |
| flatten_1 (Flatten)              | (None, 8448)          | 0           | convolution2d_5[0][0] |
| dropout_1 (Dropout)              | (None, 8448)          | 0           | flatten_1[0][0] |
| dense_1 (Dense)                  | (None, 100)           | 844900      | dropout_1[0][0] |
| dense_2 (Dense)                  | (None, 50)            | 5050        | dense_1[0][0] |
| dense_3 (Dense)                  | (None, 10)            | 510         | dense_2[0][0] |
| dense_4 (Dense)                  | (None, 1)             | 11          | dense_3[0][0] |

Total params: 981,819


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
