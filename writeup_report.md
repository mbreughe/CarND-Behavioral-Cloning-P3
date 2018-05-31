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

#### 1. Solution Design Approach and Challenges

The overal approach was to create an incremental better network by 1) changing the architecture, 2) changing hyper-parameters and 3) supplying training data in tricky situations. I started out with a simple LeNet-network just to have the car do something. Soon I started implementing the Nvidia network for self-driving cars, documented here: https://devblogs.nvidia.com/deep-learning-self-driving-cars/, with some small adjustments.

The biggest challenge I had was the fact that Cropping2D didn't work on the deployment machine: https://discussions.udacity.com/t/cropping2d-tensorflow-python-framework-errors-impl-invalidargumenterror/704632. 
I spent around 20hours training this nvidia-based model without the Cropping2D layer, because I was destined to make it work. The biggest improvement in this model were the introduction of a MaxPooling layer to reduce the image size 4x. This was an attempt to "blur out" the details as an alternative to cropping them out. Another big improvement came from increasing the batch size from 32 to 512. By providing specialized training data for the first bend and the bridge, I was able to make it all the way through bend number 2. Supplying 3442 additional images for this tricky situation did not solve the problem. This first model is saved as **model_no_crop.h5**, which drives pretty smooth until that second bend. I'd suggest the reader to also try out this model.

Another challenge I had, was the fact that after multiple training runs (over 60) I wasn't able to find a good correction factor for the steering angles when using left and right images: https://discussions.udacity.com/t/multiple-cameras-how-to-tune/709391. I suspect the Cropping2D image was also preventing me from finding this factor. This however, significantly reduced the amount of training data I could have used (by 66% !!)

Instead of quiting with a pretty good model (for not being able to crop images), I investigated different strategies to fix the initial cropping issue. I ended up creating a custom Lambda layer in Keras, that does the cropping for me. Building the layer itself costed me about an hour, which was a very good investment. The car was able to drive through bend 2 and even bend 3 without providing specialized data for it. Using this pretrained model, I was able to get through the whole track by providing additional training data for bend 4. This model is saved as **model.h5**, is less smooth than model_no_crop.h5, but drives multiple laps successfully. Because of its shaky behavior, I would probably not assign it as my designated driver just yet :).

#### 2. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64. 

After the final convolution layer, it flattens the previous output, adds dropout and adds 4 dense layers. The dense layers use ReLU activation to introduce nonlinearity.

Prior to going through the network, the images are normalized, and cropped using two lambda functions. Keras's built-in Cropping2D layer did not work on the deployment machine, which is why I implemented a Lambda function to achieve the same result.

#### 3. Attempts to reduce overfitting in the model

As mentioned before, I used a dropout layer (with 50% retention rate), to avoid overfitting. I introduced it rather early in the model, but I do remember it helping with overfitting.

In addition, I found it useful to reset the start learning rate of the optimizer when finetuning the model (see next section).

Also, when introducing new data to help the car in tricky situations (e.g., give new training data based on a tricky bend), I found it a must to train the network with all data. The model seems to learn a new function when only giving it this "exception handling training data". This new function does not generalize to the old training data and you would suddenly see the car not being able to handle simple road sections anymore.

I used scikits's built-in function to split my data between train and validation sets. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 4. Model parameter tuning

The model used an adam optimizer, however, I did use a start learning of 0.001. In addition, when fine-tuning successful models, I found it useful to reset the the start learning rate to 0.001 to avoid overfitting.

I used 10 epochs for the initial model and 4 epochs for refined versions.

I played around with batch sizes as well and found that the largest number I tried (512) achieved significant better results.

I used 33% of all data as validation data.

I tried a couple of additional cropping layers and fully connected layers, but did not find this useful. The Nvidia architecture is already highly tuned for this task. The dropout of 0.5 that I added seemed better than other values I tried.

#### 4. Appropriate training data

I used the data provided by Udacity, in addition to lots of self-created training data that I recorded from the simulator. I initially tried without the Udacity data, but I think my driving skills in the simulator are not that smooth :) -- though I used a PS4 controller. 

The additional data that I very cautiously generated was some "recovery driving", half a track all the way till tricky bend number 2 and around 10K images of tricky situations. These situations include bend 1, the bridge, bend 2 and bend 4, as well as some situations steering away from the borders of the road.

I believe my recovery driving data was a bit aggressive as you'll sometimes see the car jerking away from an almost fatal situation. 

#### 5. Final Model Architecture

The final model architecture (model.py lines 147-183) consisted of a convolution neural network with the following layers and layer sizes:

Here is a visualization of the architecture:
____________________________________________________________________________________________________
| Layer (type) |                    Output Shape   |       Num Params  |    Connected to |
|:---------------------:|:---------------------:|:---------------------:|:---------------------:|
| lambda_1 (Lambda: normalization)      |          (None, 160, 320, 3)|   0           | lambda_input_1[0][0] |
| lambda_2 (Lambda: custom cropper)  (1)    |          (None, 90, 320, 3) |   0           | lambda_1[0][0] |
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

lambda_2 is my custom layer that I built to do the cropping from scratch (Cropping2D did not work on the deployment machine).
(1) Note: for model_no_crop.h5, lambda_2 was replaced by a MaxPooling2D layer to reduce the resolution of the image.
