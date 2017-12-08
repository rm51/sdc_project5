##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: image0109.png
[image2]: image3711.png
[image3]: test_image_output.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function extract_features in lines 51 through 100 of the file called `lessons.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

At first I didn't separate out the vehicle images into testing and training and did a random split. But I wasn't getting as good results as I would have liked. So I decided to separate them manually but taking approximately every tenth image for the testing data for the cars data. This results in a much lower accuracy. Of .49 for the SVC as opposed to the 98% or 99% that I was getting before. But my results were better in the video after manually split the images. I also realized I made a mistake and split them incorrectly. Because I was taking one of each set of images and putting that into the testing data. Then later I split it up so that the testing and training didn't have the same data. But I also didnt' separate out the non-vehicles so that might be why I'm not getting as good results as I could have.

![vehicle][image1]
![non-vehicle][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

#Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(4, 4)`:


#![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.


In the quizzes I was using a HOG parameter of 1. But when running my code on the video, I ran into errors and I looked at the forum and saw other people had the same issue and they 
solved it by changing to HOG=ALL. So I also decided to use HOG=ALL. For the color spaces I tried HSV, LUV, HLS, YUV and YCrCb. I found the best results with YCrCb so I decided to use that will all three color channels as some cars are different colors so the color could be used to identify potential cars as well. I also noticed that some students didn't use HOG at all. So I intend to try things out using something else as well. When I increased the orient from 9 to 18 I got better results and it seemed to find the car more. But I think once I have found the car, I could just just move those boxes since the cars aren't changing lanes so once we have a good reading, we know they will appear in the next frame as well. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVC. At first I moved some of the training data to the testing set. But then I moved the data so that the data set wasn't in the testing set and the accuracy went down to about 50%. Previously it was at around 99%. But I still got decent results in my video although the boxes are wobbly because I haven't yet done an average of the past ten boxes. I used all three HOG channels and also the YCrCb color space. I decided upon these after trying out different color spaces and finding that YCrCb gave the best results. I originally was using HOG channel 1 and was getting good results with that but I ran into an error and I saw on the forums that it could be fixed by using HOG channel ALL. This was in single_img_features of search_classify.py in lines 42-86. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search only at the bottom of the image as cars wouldn't be found in the sky. In find_cars lines 267 - 278 I implement a sliding window search. I chose 64 so that it would 8x8 and chose cells_per_step to be 2 because when I increased cells_per_step it didn't find the car in several frames. 

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [https://youtu.be/J_ic79yOv9M]


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I filtered out xleft < 400 as that would not be within the lane lines on the right side of the road. But that is not the best way to do it as it wouldn't work in other situations such as where there's no left side. Given the cars don't change lanes I also tried to see if a car was not detected in one frame to use the previous location to draw the bbox.

I don't have an example results showing the heatmap because I changed my code so that it would take the centroid of the bboxes so my heatmap is just a square with no differentiations in color.

I also don't have the resulting bounding boxes from the last frame in the series as it didn't identify the cars accurately there.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

From looking at the forums, it seems that some people have gotten better results through using more of a deep learning approach and things such as YOLO. If I had more time I would look into those things and also for future personal projects I will try one of those things. I believe once I got a good estimate of where a vehicle is, it's not going to disappear so I could keep moving the box in subsequent frames. I also would smooth the  boxes by keeping track of the previous ten frames. I don't think this is good enough to pass this project but this project is due now so I will submit it and then keep working on it. 

One strange thing I noticed and others in the forum also noticed this is that if we try to limit the x values we get much worse results. And I don't quite understand why this is. 

I believe pipeline might fail if there are many cars such as during rush hour or if the cars switched lanes. Also it doesn't seem to recognize the cars that are in the distance. 

# Should the find_cars X_scaler be the X_scaler_test or X_scaler which is the training data? If I use X_scaler_test it performs very badly. 

I wanted to save the previous frame and once I got a reliable detection draw that bbox on if it didn't find anything. But that part doesn't seem to be working. I also plan to use a deque to record the past ten bboxes so I can average them out and make them smoother. 

I also limited the x values so that if startx were less than 300 it wouldn't add them to the bboxes list. But that would only work for similar vidoes. If there were no left side of the road in the video then it would incorrectly filter out valid cars.


