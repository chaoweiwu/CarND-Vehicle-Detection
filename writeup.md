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
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

[all_windows]: ./writeup_images/all_windows.png
[candidate_detections]: ./writeup_images/candidate_detections.png
[car_images]: ./writeup_images/car-images.png
[no_car_images]: ./writeup_images/no-car-images.png
[detections]: ./writeup_images/detections.png
[detections_many]: ./writeup_images/detections_many.png
[heat_map]: ./writeup_images/heat_map.png
[hot_windows]: ./writeup_images/hot_windows.png
[overlapping_windows]: ./writeup_images/overlapping_windows.png
[yellow_candidates]: ./writeup_images/yellow_candidates yellow_candidates.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.


I selected and tuned my features by measuring the results of my classifier (`train_classifier.py` and `train_classifier.ipynb`)
I started by reading  in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car images][car_images]
![no car images][no_car_images]

I then created a function that would extract binned colors features, color histogram features, and hog (histogram of gradient) features and combine them horizontally.
The code for this step is in `features.py` in the function `extract_features`. To extract the hog features I used from `skimage.feature.hog`.


Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I chose parameters:

```python
ORIENTATIONS = 6
PIXELS_PER_CELL = 8
CELLS_PER_BLOCK = 2
```
I verified that they worked well by testing the classifier on a validation set using `sklearn.model_selection.cross_val_score`,
getting a score of over 99% accuracy.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in `train_classifier.py` in the function `train_model`. 
I selected the hyper-parameters for the SVM in `train_classifier.ipynb` using `GridSearchCV` under the cell `Search for better params`.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the sliding window search in `sliding_window.py` in the function `sliding_window_gen`.
I tuned and discovered the size and location of windows I would use in `VehicleDetecton.ipynb` under `###Sliding Windows.`
At first I tried to define windows of several different sizes at different points on the image, which looked something like this:

![Many different types of windows][all_windows]

But then I decided it was easier to tune using windows of several different sizes with roughly the same parameters. 
I checked these against my classifier and tuned them by eye against the results that looked like this:

![All the detections][candidate_detections]
I defined the final windows I would use in `detect_vehicles.py` in the function `all_windows`

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Finally, I created a heatmap using all of the windows with all of the positive detections in `detect_vehicles.py` and `heatmap.py` 
in the functions `detect` and `add_heat`. 

![hot_windows][hot_windows]
![heat map][heat_map]

I then selected a threshold for the minimum heat to appear then detected labels using `scipy.ndimage.measurements.label`.
I tweaked the threshold and number of windows to get the best balance of avoiding false positives and finding boxes that
fit the cars well. The end result looked like:

![All the detections][detections_many]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the biggest challenges in this project is finding the correct window size and shape to balance speed and performance with
ability to cover cars of all distaces and areas in the screen. You want to be able to find cars at any position, but you don't
want to have so many windows that your pipeline is too slow to actually, test, improve, and run in practice.
My pipeline will likely fail if the camera is pointed at a different angle than the one it was tested on. For example,
if the cars appear slightly higher up in the image, it may miss them. Also, the pipeline is untested on other vehicle types,
it could potentially have problems with buses or large trucks. It may also have trouble if it is moved over to the right lane after being tested on the leftmost lane.
The lowest hanging fruit for improvement at this point is speeding up the feature extraction so windows are less expensive. It would also
be very useful to spend more time carefully selecting the best windows so we can accomplish more with fewer of them.

