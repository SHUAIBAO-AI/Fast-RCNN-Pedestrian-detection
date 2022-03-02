# Fast-RCNN-Pedestrian-detection
If we want to detect multiple objects like car and pedestrian. We should maximize the pooling of input with non-uniform dimensions to obtain a fixed size feature map. 
Problem Specifications: 
	Please describe the 2 key components in the Fast R-CNN framework: the RoIPooling layer and the loss functions in the framework. 
Answer：
About ROIPooling:
	If we want to detect multiple obje	cts like car and pedestrian. We should maximize the pooling of input with non-uniform dimensions to obtain a fixed size feature map. There are two steps of object detection:
	Region proposal: for an given image, find possible locations where objects may exist by sliding window or selective search algorithm. After this processing, we output the bounding box on possible location. We call it as region proposals or regions of interest (ROI).
	Final classification: determine whether each region proposals in the previous stage belong to the target class or background.
But in these operation, it’s difficult for us to achieve real-time target detection. SO that we get ROIPooling layer:
Two of key components of ROIPooling is that:
	Fixed size feature maps obtained from deep networks with multiple convolution kernels pooling.
	N*5 matrices for all ROI, where n is the number of ROI. The first column represents the index of the image, and the remaining four columns represent the coordinates of the upper left corner and the lower right corner.
The operation steps of ROIPooling is that:
	Map ROI to the corresponding location of feature map according to the input image.
	Divide the mapped area into sections of the same size (the number of sections is the same as the output dimension).
	Max pooling for every section.
About loss functions:
Loss function is used to evaluate the level of differences between predicted value and true value. Generally ,The better the loss function is, the better the performance of the model is. In different model we often apply for different loss function.
Loss function has been divided into two parts: empirical risk loss function and structural risk loss function. The empirical risk loss function refers to the difference between the predicted results and the actual results. The structural risk loss function refers to the empirical risk loss function plus the regular term. The normally loss functions: zero-one loss, Absolute loss function, log loss function, square loss function, exponential loss, Hinge loss, perceptron loss, Cross-entropy loss function.
Fast-R-CNN achieve classification and regression in same neural net,so the loss function must be multitasking. And in that, classification means log loss function.
 
![image](https://github.com/STPChenFang/Fast-RCNN-Pedestrian-detection/blob/main/Fast%20RCNN%20IMG/image001.png)

Log loss function, also named as log-likelihood loss, logistic loss or cross-entropy loss, which defined by statistics. We often apply it on probability output of classification.
Log loss function can quantize the accuracy of classifier by classifying the punish wrong. Minimize the log loss is equal to maximize the accuracy of classifier. It’s necessary for classifier to input probability of different type. And the equation of log loss function is below:
![image](https://github.com/STPChenFang/Fast-RCNN-Pedestrian-detection/blob/main/Fast%20RCNN%20IMG/image003.png)

In this equation, Y means output variable, X is input variable, L is loss function, N is number of samples, M is number of possible classes, yij is binary index, which represent class j is the class of input sample xi or not, pij is possibility of model or classifier predicted result that input xi belong to class j.
Logarithmic loss is used for maximum likelihood estimation (the result is judged by the maximum probability, which is called maximum likelihood estimation)
The likelihood value of a set of parameters under a pile of data is equal to the product of conditional probability of each data under this set of parameters. The loss function is generally the sum of the loss of each data. In order to change the product into sum, the logarithm is taken.
Add a minus sign to match the maximum likelihood with the minimum loss. It is used to judge the closeness of the actual output and the expected output, and describes the distance between the actual output probability and the expected output probability, that is, the smaller the cross entropy is, the closer the two probability distributions are.

2. Please describe the object detection performance metric, mAP (Mean Average Precision), and explain why it can well reflect the object detection accuracy. 
Answer：
Mean Average Precision (mAP) is the performance measure of this kind of algorithm to predict the target location and category. mAP is very useful for evaluating target location, target detection model and instance segmentation.
In the prediction, there are many bounding boxes, but most of them have very small confidence. We only need to output the bounding boxes whose confidence exceeds a certain threshold.
mAP(mean average precision),is index of accuracy of object detection. In multiple objects detection project, we can draw curve by recall and precision, average precision means the area below the curve, mAP means that we calculate the average value of each AP of each classes.
For recall and precision:
Ture positives (TP): The number of correct positive division, that is, the number of instances actually positive and classified as positive by the classifier.
False positives (FP): The number of wrong positive division, the number of instances actually negative and classified as positive by the classifier.
False negatives (FN): The number of wrong negative division, that is, the number of instances that are actually positive but are classified as negative by the classifier.
True negatives(TN): The number of correct negative division, that is, the number of instances that are actually negative but are classified as negative by the classifier.
How to calculate the mAP?
	Get the confidence score of all test samples with the trained model, and save the confidence score of each class (such as car) to a file (such as Comp1_ cls test_ car. txt)。 Assume that there are 20 samples, confidence score and ground truth lab for the test:
![image](https://github.com/STPChenFang/Fast-RCNN-Pedestrian-detection/blob/main/Fast%20RCNN%20IMG/image005.png)

We sort the confidence score:
![image](https://github.com/STPChenFang/Fast-RCNN-Pedestrian-detection/blob/main/Fast%20RCNN%20IMG/image007.png)

Calculate the precision and recall:
![image](https://github.com/STPChenFang/Fast-RCNN-Pedestrian-detection/blob/main/Fast%20RCNN%20IMG/image009.png)

The figure above is more intuitive. The elements in the circle (true positions + false positions) are the elements we selected, which correspond to the results we take out in the classification task. For example, for the test sample classification on the trained car model, we want to get the top-5 results, that is 

![image](https://github.com/STPChenFang/Fast-RCNN-Pedestrian-detection/blob/main/Fast%20RCNN%20IMG/image011.png)

In this example, true positions refers to the 4th and 2nd pictures, and false positions refers to the 13th, 19th and 6th pictures. The elements inside the box and outside the circle (false negatives and true negatives) are relative to the elements inside the selection. In this case, it means that the confidence score is listed after top-5.

![image](https://github.com/STPChenFang/Fast-RCNN-Pedestrian-detection/blob/main/Fast%20RCNN%20IMG/image013.png)

Among them, false negatives refer to the 9th, 16th, 7th and 20th pictures, and true negatives refer to the 1st, 18th, 5th, 15th, 10th, 17th, 12th, 14th and 8113th pictures.
So, in this example, precision = 2 / 5-40%, which means that for the car category, we have selected 5 samples, of which 2 are correct, i.e. the accuracy is 40%; recall = 2 / 6 = 30%, which means that there are 6 cars in all the test samples, but we just recall 2 of them, so that the recall rate is 30%.
In the actual multi category classification task, we usually don't meet the requirement of only measuring the quality of a model by top-5, but we need to know the precision and recall corresponding to top-1 to top-N (n is the number of all test samples, in this example is 20). Obviously, as we select more and more samples, recall will be higher and higher, and precision will be in a downward trend as a whole. If recall is used as abscissa and precision as ordinate, then precision-recall curve can be clear.
	In the end, AP measure the performance of bad or good in each trained model, mAP measure the performance of bad or good in all trained model. In other words, precision is TP/(TP+FN), how many useful percentage in the result we returned, in contrast, how many useless percentage in the result we have returned? Two of these number is combined, it’s meaningless that we discuss one of it without the other one. For example, we can increase the recall to 100%, which means no matter what the request is , the program always return the whole documents to output, it must has no loss but in the meantime the request cannot get useful result.

3. Please train and test the Fast R-CNN framework on one of the existing pedestrian detection datasets, and report the final AP performance that you have achieved.  The dataset could be CUHK-SYSU [2], Citypersons [3].   Please also report some pedestrian detection examples by including the images and bounding boxes.  (40%).  

Answer:
The final AP performance that I have achieved is 70%.

![image](https://github.com/STPChenFang/Fast-RCNN-Pedestrian-detection/blob/main/Fast%20RCNN%20IMG/image015.png)

Figure:Screen shoot of running

![image](https://github.com/STPChenFang/Fast-RCNN-Pedestrian-detection/blob/main/Fast%20RCNN%20IMG/image017.png)
Figure:Screen shoot of end of runing

![image](https://github.com/STPChenFang/Fast-RCNN-Pedestrian-detection/blob/main/Fast%20RCNN%20IMG/image0119.png)
Figure:Sample image with bounding boxes

![image](https://github.com/STPChenFang/Fast-RCNN-Pedestrian-detection/blob/main/Fast%20RCNN%20IMG/image021.png)
Figure: Sample image with bounding boxes



4.Propose your own method to further improve the pedestrian detection performance based on the Fast R-CNN framework. (20%) 
	Answer: The improvement of feature extraction network: using RESNET instead of VGG to extract features.
 
References
	https://blog.csdn.net/ff_xun/article/details/82354999 mAP explain
	https://www.ziiai.com/blog/755 
