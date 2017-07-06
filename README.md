#**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./images/hist.png "Histogram"
[image2]: ./images/grayscale.png "Grayscaling"
[image3]: ./images/init_distrib.png "Initial distribution"
[image4]: ./images/post_distrib.png "Post distribution"
[image5]: ./images/processed.png "Processed"
[image6]: ./extra_signs/sign1.jpg "sign1"
[image7]: ./extra_signs/sign2.jpg "sign2"
[image8]: ./extra_signs/sign3.jpg "sign3"
[image9]: ./extra_signs/sign4.jpg "sign4"
[image10]: ./extra_signs/sign5.jpg "sign5"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

Here is a link to my [project code](https://github.com/deepankarsharma/udacity_traffic_signs/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

I used matplotlib to examine the distribution of classes. Looking at this allowed me to realise that the distribution is highly non-uniform and there are several classes for which we do not have adequate training data. 

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing distribution of the training, validation and test datasets.

![alt text][image1]

###Design and Test a Model Architecture


As a first step, I decided to convert the images to grayscale so that the search space of the model is cut down. I also observed that performance improved a couple of percent by making this change. 

After grayscaling I normalized the model to further cut down the search space for the model.

Here is an example of a traffic sign image after preprocessing

![alt text][image5]

I decided to generate additional data because I observed that my models accuracy was stuck at about 87%. On looking at the exploratory data analysis I observed that many classes barely have any data.

To add more data to the the data set I ensured that every class has at least 1200 examples. I created the extra examples by doing random transformations on random selections of provided examples in the training data.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is a LeNet model with dropout.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten    	      	| outputs 400                    				|
| Fully Connected    	| outputs 120                    				|
| RELU					|												|
| Dropout				|												|
| Fully Connected    	| outputs 84                    				|
| RELU					|												|
| Dropout				|												|
| Fully Connected    	| outputs 43                    				|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer. The learning rate was 0.001, epochs were 100 and batch size was 128. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 97.2
* test set accuracy of 94.7

I followed the following methodolody to arrive at my final conclusion

1. Initial data exploration - notice shortage of test data for many classes
2. Proceed to implement lenet to serve as baseline
3. Initial version had test accuracy at ~80%
4. Switching to black and white normalized images, adding dropout and tweaking some knobs raises that to 87%.
5. Writing a data augmentation layer raised the test accuracy to ~94.7%.  

If a well known architecture was chosen:
* What architecture was chosen?
Lenet
* Why did you believe it would be relevant to the traffic sign application?
I was familiar with the architecture and coded it up to establish a baseline. Once I coded it I realised that performance was in the right ballpark (Initial version itself gave ~85% accuracy on test set). I felt with some engineering and tweaks I could get to ~95% accuracy on this architecture. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Even on test set data accuracy is over 94.7%. This means model is learning and generalising well.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6]
![alt text][image7]
![alt text][image8] 
![alt text][image9]
![alt text][image10]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Accuracy here was 50% vs 94.7% for the test set. 

Predictions were (Predicted / Actual)
1. Children crossing / Children crossing
2. Priority road / Priority road
3. Stop / No passing 
4. 60 / Dangerous turn to left
5  Road work / Road Work
6. Unknown / Yield



####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


Image 0 
	probabilities: [  1.00000000e+00   3.37353497e-38   0.00000000e+00   0.00000000e+00
   0.00000000e+00] 
	classes: [28 34  0  1  2]
Image 1 
	probabilities: [  1.00000000e+00   1.49683227e-37   0.00000000e+00   0.00000000e+00
   0.00000000e+00] 
	classes: [12 19  0  1  2]
Image 2 
	probabilities: [ 1.  0.  0.  0.  0.] 
	classes: [9 0 1 2 3]
Image 3 
	probabilities: [ 1.  0.  0.  0.  0.] 
	classes: [19  0  1  2  3]
Image 4 
	probabilities: [ 1.  0.  0.  0.  0.] 
	classes: [31  0  1  2  3]
Image 5 
	probabilities: [ 1.  0.  0.  0.  0.] 
	classes: [13  0  1  2  3]


For the second image ... 

