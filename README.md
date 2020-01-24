# **Lane Line Detection Pipeline** 

### Completed for Udacity Self Driving Car Engineer - 2018/02

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./ReportPics/15TrafficSigns.png "Visualization"
[image2]: ./ReportPics/TrainingSetHist.png "Training Set Histogram"
[image3]: ./ReportPics/ValidationSetHist.png "Validation Set Histogram"
[image4]: ./ReportPics/TestSetHist.png "Test Set Histogram"
[image5]: ./ReportPics/OnlineImg.png "Images from the Web"
[image6]: ./ReportPics/ModifiedArchitecture.png "Modified Multi-Stage Architecture"
[image7]: ./ReportPics/PreProcess.png "Pre Processed Image Samples"
[image8]: ./ReportPics/Processed_TestImages.png "Pre Processed Test Images"
[image9]: ./ReportPics/Softmax1.png "Softmax Probabilities 1"
[image10]: ./ReportPics/Softmax2.png "Softmax Probabilities 2"
[image11]: ./ReportPics/Results.png "Results"



**Project Rubric : [Rubric Points](https://review.udacity.com/#!/rubrics/481/view)**


---
### Project Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/JamesDiDonato/CarND-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The default training, validation, and test sets are imported from their respective .pickle files. Each image is size 32x32x3 and represents 1 of a possible 43 image classifications.  The size of each data set is summarized below: 

|Data Set|Absolute Size|Relative Size|
|---|---|---|
|Training Set|34799|67%|
|Validation Set|4410|8.5%|
|Test Set|12630|24.5%|


#### 2. Include an exploratory visualization of the dataset.

First, 15 random images are displayed from the training set

![Image Sampling][image1]

An observation of the images :different contrast levels, brightness, and physical sign size, some noise factors network will need to be robust to.

Histograms are plotted and analyzed for each of the three data sets:

![Training Set Histogram][image2]
![Validation Set Histogram][image3]
![Test Set Histogram][image4]

The distribution for each of the sets follows a similar pattern, implying that the three were generated from a randomized master dataset . While I have never been to Germany, I can observe that the following signs are not very common : 

|Sign|ID|
|---|---|
|20 km/h|0|
|End of Speed Limit 80 km/h |6|
|Dangerous Curve to the Left|19|
|Dangerous Curve to the Right |20|
|Double Curve |21|
|Road Narrows on the Right|24|
|Pedestrians|27|
|Bicycles Crossing|29|
|End of all Speed and Passing Limits|32|
|Go Straight or Left|37|
|Roundabout Mandatory|40|
|End of No Passing|41|


 While the following are more popular : 

|Sign|ID|
|---|---|
|Speed limit (30km/h)|1|
|Speed limit (50km/h)|2|
|Speed limit (70km/h)|4|
|Speed limit (80km/h)|5|
|No passing for vehicles over 3.5 metric tons|10|
|Priority road|12|
|Yield|13|
|Keep Right|38|

In theory, the worst case accuracy of the model would be on the signs that appear less frequently in the data set. Later on, these under-represented image will be used to judge the robustness of the model.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

An untouched LeNet model was initially tested without any pre-processing to establish a baseline accuracy. Model parameters were also default (EPOCHS = 10, BATCH_SIZE = 128, RATE = 0.001, SIGMA = 0.1, MU = 0). The accuracy on the validation was 86.9%. The model will be tested in an iterative process as design changes are made to both pre-processing and architecture, in order to validate their effects.

Pre-processing steps:
* Images are converted to gray scale. It was noticed that the training time is significantly shorter due to shrinking the input depth from (R,G,B) to (Gray) with a GTX 960 GPU being used. Validation set accuracy improves to 91.12%. It appears that color is not a useful feature for identifying the traffic signs and was confusing the feature set space. This is a suprising result to me, as you would think that color is an obvious feature that can be used to narrow down a traffic sign.
* The data sets are normalized to [-1,1] by applying the following formula to each pixel : normalized_pixel =(pixel - 128)/128 . The average of each dataset is roughly -0.35. Normalziation was performed so that features in the image are considered with equally by the network, leading to faster convergence of the network weights with less oscillations around minima. Validation set accuracy still sits at 91.18% after applying the normalization with original Le-Net architecture. 

A random sample of 15 pre-processed images is shown:

![Pre-Processed Images][image7]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


|Layer|Output Size|
|---|---|
|Input|32x32x1|
|Convolution 1|28x28x6|
|Relu Activation|28x28x6|
|Pooling|14x14x6|
|Convolution 2|8x8x10|
|Relu Activation|8x8x10|
|Convolution 3|4x4x20|
|Multi-Stage Feed Forward (conv2 + conv3)|1x960|
|Dropout - prob = 0.5|1x960|
|Fully Connected Layer 1|1x100|
|Relu Activation|1x100|
|Fully Connected Layer 2|1x43|


The architecture consists of a 3-stage convolutional layer feeding a 2-stage fully connected layer. Each layer output is activated using the non-linear relu function. The design also features multi-stage approach where the output of the 2nd convolutional layer is concatenated with the output of the 3rd to feed the fully connected layer. By implemeting the multi-stage approach, the model was able to learn both local as well as high level features of the image. 

I began this project with the Le-Net  and moved to this modified architecture in the spirit of improving validation accuracy above 93%. With the multi-stage model, I was able to reach an accuracy of 95%. 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The hyperparameters were tuned in an iterative fashion where each parameter was analyzed for its affect on model accuracy.  With a larger batch size, I noticed the model takes more epochs to become accurate, whereas training with smaller batch sizes leads to a faster convergence to the nominal accuracy. A value of 100 was settled upon. A dropout probability value of 0.5 was selected to randomly reduce the number of neurons and manage overfitting. I realized I could have spent hours tuning each of the hyperparameters to an optimal state, but instead I focused my time on improving the architecture. The final hyperparameter selections were :
* EPOCHS = 60
* BATCH_SIZE = 100
* rate = 0.0008
* dropout_prob = 0.5
* mu = 0
* sigma = 0.1


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

12/14/2018
* Tuning of the parameters to EPOCHS = 60, BATCH_SIZE = 100, rate = 0.008, including normalization and grayscaling, I was able to acheive a reasonable 93% accuracy on the validation set very quickly.  Increasing the number of EPOCHS has a positive effect on the accuracy given the model had more iterations to train upon. Not only did reducing the batch size from 120 improve the memory required to train my model, I noticed an immediate effect on the accuracy, raising to 94% for an iteration or two. However the accuracy settled down to 93% once the model was done training, which I attribute to the fact that smaller batch sizes are yielding an inacurate estimate of the gradient and overshooting the optimal.
* Looking to improve this, I suspected overfitting was occuring because  the training set accuracy was nearly 100% yet the validation set was stuck at 93%. I wanted to see the effect of max pooling so I removed one of the pooling operations from the LeNet architecture and doubled the size of my fully connected layer to 1600 on the input. Performance hardly  changed from 93%. 
* In addition, I tried the regularization technique dropout at the beggining of the fully connected layer, I was able to acheive 94% accuracy on the validation set. Dropout probabilty is set to 0.5 and represents a tunable hyper-parameter described above.

At this point, the linked paper written by Pierre Sermanet and Yann LeCun was consulted for design inspiration. A new archieture was born with a new name : DiDonato. In this modified architecture the multi-stage approach is utilized. The idea is to have convlolutional layers  fed-forward to the fully connected layer in order to capture the high and low level features in classifier weights. The early convolutional layers are thin and contain high level invariant features such as geometries whereas the later convolutional layers consists of local patterns. It would be possible to visulize this effect by plotting images at each of the convolutional layers, however for the sake of time this was skipped.

![alt text][image6]

The new architecture immediately showed an immediate 2% improvement  in the validation set accuracy. My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 95.3%
* test set accuracy of 93.7%


![Image Sampling][image11]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8 German traffic signs that I found on google images. The class ID is indicated on each. 

![alt text][image5]

Images are shown after applying pre-processing:

![alt text][image8]

The images selected have been downsampled to 32x32x3 using an online widget http://picresize.com/. In particular, there appears little that stands out about each image. They appear as normal : low sun glare, low blur, no vandalism, and fairly square viewing angle - all noise factors that would impact the performance of a real world system. After pre-processing, the images are still clear and differentiable to the human eye. The size of the traffic sign within the image appears to be slightly larger than the provided image databases, but I suspect this to have no effect on performance. However I will point out that I purposely decided to challenge the robustness of the model with images that were not as common in the training set. Specifically, I chose the following classes that had frequencies in the 0 - 500 range : 0, 19, 20, 40, 29.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to accurately predict 75% of the images ( 6/8 ). This is lower than the accuracy of the test set which had an accuracy of 93.7%, alebit the test set had more samples.

The image class ID's are compared with their predictions:


|Image|Predition|
| --- | --- |
|25|25|
|33|33|
|0|0|
|19|19|
|40|41|
|12|12|
|29|14|


The two images that were incorrectly classified were 40 - Roundabout Mandatory , and 29 - Bicycle Crossing. Both of these images were not well represented in the training set with counts of ~250,. A reccomendation for future work would be to augment these two class ID's with more data and then re-run this test. My hypothesis is that the model would predict these classes accurately if the data sets were increased to at least 800 images.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The softmax probabilities are displayed for each image below. The model confident predicts the 6/8 classes correctly with softmax probabilities greater than 0.9, however does a very poor job at identifying the Class 40 and 29.

![alt text][image9]
![alt text][image10]

