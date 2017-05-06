# **Advanced Lane Finding Project**

This is a resubmission of the project 4. I thank the reviewer for providing me with the constructive feedback and an opportunity to revise the code to improve the quality of my submission. I appreciate all the suggestions to improve my submission. To emphasize how I addressed the reviewer's concerens, I used **_italic and bold_** sentences in the text below to indicate major changes from the previous submission. Briefly, I imporved the quality of the final movie by 1) Incorporating smoothing of polynomial coefficients over 5 previous movie frames; 2) Adding R channel in RGB space to detect white lanes better, and 3) If the current polynomial coefficients are way off compared with the previous coefficients, the current coefficients are rejected and the previous coefficients are used instead. I hope the quality of the final movie meets the criteria for successful submission.

[//]: # (Image References)

[image1]: ./figure_1.png
[image2]: ./figure_2.png
[image3]: ./figure_3.png
[image4]: ./figure_4.png
[image5]: ./figure_5.png
[image6]: ./figure_6.png
[video1]: ./project_video_processed.mp4
 
## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
  
---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point. 

*SPECIFICATIONS: The writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled.*

You're reading it!

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

*SPECIFICATIONS: OpenCV functions or other methods were used to calculate the correct camera matrix and distortion coefficients using the calibration chessboard images provided in the repository (note these are 9x6 chessboard images, unlike the 8x6 images used in the lesson). The distortion matrix should be used to un-distort one of the calibration images provided as a demonstration that the calibration is correct. Example of undistorted calibration image is Included in the writeup (or saved to a folder).*

The code for this step is contained in lines 9 through 63 of the file called `P4_code.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

---
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

*SPECIFICATIONS: Distortion correction that was calculated via camera calibration has been correctly applied to each image. An example of a distortion corrected image should be included in the writeup (or saved to a folder) and submitted with the project.*

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

*SPECIFICATIONS: A method or combination of methods (i.e., color transforms, gradients) has been used to create a binary image containing likely lane pixels. There is no "ground truth" here, just visual verification that the pixels identified as part of the lane lines are, in fact, part of the lines. Example binary images should be included in the writeup (or saved to a folder) and submitted with the project.*

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 152 through 159 in `P4_code.py`).  Here's an example of my output for this step.

**_In addition to the S channel in HLS color space to identify yellow lanes, I used the R channel in RGB color space to identify white lanes. This is incorporated in 'color_thresh' function at lines 139 through 150 in `P4_code.py`._** 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

*SPECIFICATIONS: OpenCV function or other method has been used to correctly rectify each image to a "birds-eye view". Transformed images should be included in the writeup (or saved to a folder) and submitted with the project.*

The code for my perspective transform includes a function called `warper()`, which appears in lines 168 through 173 in the file `P4_code.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[263, 715],
     [584, 458],
     [700, 458],
     [1156, 715]])
dst = np.float32(
    [[320,720],
     [320,0],
     [960,0],
     [960,720]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 584, 458      | 320, 0        | 
| 263, 715      | 320, 720      |
| 1156, 715     | 960, 720      |
| 700, 458      | 960, 0        |

To improve the accuracy of perspective transform, I also added a function called `region_of_interest()`, which appears in lines 178 through 190 in the file `P4_code.py`.  The `region_of_interest()` function, alse used in Project 1, takes as inputs an image (`img`) and vertices that surround the region of interest (ROI) (`vertices`) to mask the image outside the ROI. I chose the hardcode the ROI in the following manner:

```
vertices = np.array([[(70,720),(550, 450), (700, 450), (1210,720)]], dtype=np.int32)
```

I verified that my perspective transform was working as expected by confirming that the lanes appear parallel in the warped image. Of note, the area circumscribed by the red line in the figure below indicates the ROI.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

*SPECIFICATIONS: Methods have been used to identify lane line pixels in the rectified binary image. The left and right line have been identified and fit with a curved functional form (e.g., spine or polynomial). Example images with line pixels identified and a fit overplotted should be included in the writeup (or saved to a folder) and submitted with the project.*

Then I used the histogram-based sliding window method that I learned in the course to find the lanes and fit the lane lines with a 2nd order polynomial. This process is described in lines 220 through 313 in the file `P4_code.py`.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

*SPECIFICATIONS: Here the idea is to take the measurements of where the lane lines are and estimate how much the road is curving and where the vehicle is located with respect to the center of the lane. The radius of curvature may be given in meters assuming the curve of the road follows a circle. For the position of the vehicle, you may assume the camera is mounted at the center of the car and the deviation of the midpoint of the lane from the center of the image is the offset you're looking for. As with the polynomial fitting, convert from pixels to meters.* 

I did this in lines 315 through 344 in my code in `P4_code.py`. In essence, I fitted the lanes to polynomials in the world coordinate system (in meters), and calculated the radius of the curvature. In this example, the left lane curve radius was 703 m, the right lane curve radius was 1,087 m, and the vehicle was 0.05 m off the center of the two lanes to the right. These numbers are also shown in the figure below.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

*SPECIFICATIONS: The fit from the rectified image has been warped back onto the original image and plotted to identify the lane boundaries. This should demonstrate that the lane boundaries were correctly identified. An example image with lanes, curvature, and position from center should be included in the writeup (or saved to a folder) and submitted with the project.*

I implemented this step in lines 348 through 378 in my code in `P4_code.py`.  Below is an example of my result on a test image. In the figure, `Radius of Curvature` indicates the mean radius between both lanes, and `Minimum Radius of Curvature` is the smaller of the two.

![alt text][image6]

---
### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

*SPECIFICATIONS: The image processing pipeline that was established to find the lane lines in images successfully processes the video. The output here should be a new video where the lanes are identified in every frame, and outputs are generated regarding the radius of curvature of the lane and vehicle position within the lane. The pipeline should correctly map out curved lines and not fail when shadows or pavement color changes are present. The output video should be linked to in the writeup and/or saved and submitted with the project.* 

In the video pipeline I used the convolution based sliding window method that I learned in the course. Here's a [link to my video result](./project_video_processed02.mp4) 

Of note, I can play the mp4 movie on my MacBook Air, but I cannot play it once I upload it to my Github repository (https://github.com/ashikagah/CarND-P4/). I don't know if it's my browser problem (Firefox 52.0.2) or the VideoFileClip function. Just to make sure the reviewer can play it, I will submit my project in a zip file this time.

**_The current polynomial coefficients for the lanes (lf_i, rf_i) are obtained from the convolution based sliding window method (line 415-416). These coefficients are rejected if they are very different from the previous coefficients (lf_i_minus_1, rf_i_minus_1) (line 417-423). When the current coefficients are rejected, the previous coefficients are used instead. Then the polynomial coefficients that are actually used to draw lanes and calculate curvature (left_fit, right_fit) are obtained by the mean of 5 movie frames (lf_i through lf_i_minus_4 and rf_i through rf_i_minus_4) to smooth the coefficients (line 424-426). At the end of the function, the coefficnets of the 4 movie frames (lf_i through lf_i_minus_4 and rf_i through rf_i_minus_4) are renewed in preparation for the next frame (line 451-459)._**

---
### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

*SPECIFICATIONS: Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail.*

The project was relatively straightforward until I started working on the video. My initial version of the code failed to identify the lanes accurately when the road color is bright. I modified the thresholds for gradient, magnitude and color but these did not improve the accuracy. Then I masked the image outside the ROI just like Project 1, and this made a big improvement. However, it took me several iterations to reach a reasonable ROI for the accuraet result. Another parameter that I had trouble with was the margin of search for the convolution-based sliding window method. The default was 100, and I modified it down to 10. I found that, if it is too large, the accuracy suffers. If it is too small, misidentification of the lane in a single frame could last until the end of the movie. I ended up using 50. I believe it's not showing 'catastrophc failures' but I am still not too happy with the final result.

Since my pipeline is sensitive to the ROI. In this project I made it relatively tight to remove any potential noise that could cause misidentification of the lanes. This means that my pipeline would not do well on the road off the highway that has more acute turns. If I were going to pursue this project further, I might improve it by spending more time in segmentation of the lanes in the binary images before fitting polynomials.  

**_While smoothing over several movie frames minimizes outliers due to a sudden change in road color and tree shades, it definitely introduces another issue, which is misregistration of lanes. As is clear from the final movie, the side edges of the green area is slightly off the lane lines. This is an inevitable consequence of moving avaerage. However, this is probably more acceptable than gross misregistration of the lanes that could lead the car to go off the road._**
