# **Traffic Sign Recognition** 

## James DiDonato
## February 2018

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


**Project Rubric : [Rubric Points](https://review.udacity.com/#!/rubrics/481/view)**


---
### Project Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/JamesDiDonato/CarND-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The default training, validation, and test sets are imported from their respective .pickle files. Each image is size 32x32x3 and represents 1 of a possible 43 image classifications.  The size of each data set is summarized below: 

|Data Set|Size|Relative Size|
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
|End of all Speed and Passing Limits|32|
|Go Straight or Left|37|
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

In theory, the worst case accuracy of the model would be on the signs that appear less frequently in the data set. Augmenting the data set with additional images will be explored as a technique to improve the models accuracy.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

An untouched LeNet model was initially tested without any pre-processing to establish a baseline accuracy. Model parameters were also default (EPOCHS = 10, BATCH_SIZE = 128, RATE = 0.001, SIGMA = 0.1, MU = 0). The accuracy on the validation was 86.9%. Ok, now we have a baseline to work from. The model will be tested in an iterative process as design changes are made to both pre-processing and architecture, in order to validate their effects.

Pre-processing steps:
* Images are converted to gray scale. It was noticed that the training time is significantly shorter due to shrinking the input depth from (R,G,B) to (Gray) with a GTX 960 GPU being used. Validation set accuracy improves to 91.12%. Color is not a useful feature for identifying the traffic signs and loosens the structure.
* The data sets are normalized to [-1,1] by applying the following formula to each pixel : normalized_pixel =(pixel - 128)/128 . The average of each dataset is roughly -0.35. Normalziation was performed so that features in the image are considered with equally by the network, leading to faster convergence of the network weights with less oscillations around minima. Validation set accuracy still sits at 91.18%.  	
* At this point, the training set accuracy is roughly 99% while the validation set is only 90%, and so I decided to generate additional data in order to improve the accuracy of the validation set . As illustrated above, there are certain images in the training dataset that are not very common and without sufficient data to  train on, this could lead to the model not being able to sufficiently learn the features to properly classify these images on fresh images it has not seen before. The goal will be to add images to each bin in the training data until the quantity reaches the average, 809.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


