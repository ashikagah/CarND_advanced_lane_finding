import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

##############################    CAMERA CALIBRATION    ############################## 

# prepare object points
nx = 9
ny = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob("camera_cal/calibration*.jpg")

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        plt.imshow(img)

# Test undistortion on an image
img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('calibration_cal/calibration1_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)

##############################         PIPELINE         ############################## 

## 1. Distortion correction  

# Load undistorted road image
Img = cv2.imread('examples/signs_vehicles_xygrad.png')

# Perform distortion correction using the result from camera calibration above
ImgC = cv2.undistort(Img, mtx, dist, None, mtx)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
ax1.imshow(cv2.cvtColor(Img,cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(cv2.cvtColor(ImgC,cv2.COLOR_BGR2RGB))
ax2.set_title('Undistorted Image', fontsize=30)

## 2. Binary image
 
def abs_sobel_thresh(image,orient='x',sobel_kernel=3,thresh=(0, 255)):
    # Grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # Gaussian blur
    gray = cv2.GaussianBlur(gray,(9,9),0)
    # Apply cv2.Sobel()
    if orient == 'x':
        sobel = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    else:
        print("Error: orient must be either x or y.")
    # Take the absolute value of the output from cv2.Sobel()
    abs_sobel = np.absolute(sobel)
    # Scale the result to an 8-bit range (0-255)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Apply lower and upper thresholds
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Create binary_output
    return grad_binary
    
def mag_threshold(image, sobel_kernel=3, mag_thresh=(0, 255)):    
    # Grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # Gaussian blur
    gray = cv2.GaussianBlur(gray,(9,9),0)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # Gaussian blur
    gray = cv2.GaussianBlur(gray,(9,9),0)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary

def color_thresh(image,thr_r=(200, 255),thr_s=(170, 255)):
    # Gaussian blur
    image = cv2.GaussianBlur(image,(9,9),0)
    # Separate the R channel to extract white lanes
    r_channel = image[:,:,0]
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # Threshold color channel
    color_binary = np.zeros_like(s_channel)
    color_binary[((s_channel >= thr_s[0]) & (s_channel <= thr_s[1]))|((r_channel >= thr_r[0]) & (r_channel <= thr_r[1]))] = 1
    return color_binary

# Choose a Sobel kernel size
ksize = 15 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(ImgC,orient='x',sobel_kernel=ksize,thresh=(20,100))
grady = abs_sobel_thresh(ImgC,orient='y',sobel_kernel=ksize,thresh=(5,100))
mag_binary = mag_threshold(ImgC,sobel_kernel=ksize,mag_thresh=(30,100))
dir_binary = dir_threshold(ImgC,sobel_kernel=ksize,thresh=(0.7,1.3))
color_binary = color_thresh(ImgC,thr_s=(200, 255),thr_r=(170, 255))
combined_binary = np.zeros_like(dir_binary)
combined_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1) ] = 1

# Visualize binary image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
ax1.imshow(cv2.cvtColor(ImgC,cv2.COLOR_BGR2RGB))
ax1.set_title('Undistorted Image', fontsize=30)
ax2.imshow(combined_binary,cmap="gray")
ax2.set_title('Binary Image', fontsize=30)

## 3. Perspective transform

def warper(img,src,dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1],img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped

def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Manually get coordinates of 4 src corners
# image_sample = cv2.imread('examples/example_output.jpg')
# plt.imshow(image_sample) 

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

vertices = np.array([[(70,720),(550, 450), (700, 450), (1210,720)]], dtype=np.int32)
masked_image = region_of_interest(combined_binary, vertices)
warped = warper(masked_image,src,dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# Visualize perspective transform
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
ax1.imshow(cv2.cvtColor(ImgC,cv2.COLOR_BGR2RGB))
ax1.plot((vertices[0][0][0],vertices[0][1][0],vertices[0][2][0],vertices[0][3][0],vertices[0][0][0]),(vertices[0][0][1],vertices[0][1][1],vertices[0][2][1],vertices[0][3][1],vertices[0][0][1]),color='green',linewidth=1)
ax1.set_title('Undistorted Image',fontsize=30)
ax2.imshow(warped,cmap="gray")
ax2.set_title('Perspective Transform',fontsize=30)

## 4. Fitting lanes with polynomial
Img = cv2.imread('test_images/test2.jpg')
ImgC = cv2.undistort(Img, mtx, dist, None, mtx)
ksize = 15
gradx = abs_sobel_thresh(ImgC,orient='x',sobel_kernel=ksize,thresh=(20,100))
grady = abs_sobel_thresh(ImgC,orient='y',sobel_kernel=ksize,thresh=(5,100))
mag_binary = mag_threshold(ImgC,sobel_kernel=ksize,mag_thresh=(30,100))
dir_binary = dir_threshold(ImgC,sobel_kernel=ksize,thresh=(0,1.3))
color_binary = color_thresh(ImgC,thr_s=(200, 255),thr_r=(170, 255))
combined_binary = np.zeros_like(dir_binary)
combined_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1) ] = 1
masked_image = region_of_interest(combined_binary,vertices)
binary_warped = warper(masked_image,src,dst)

# Take a histogram of the bottom half of the image
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
# Create an output image to draw on and visualize the result
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# Find the peak of the left and right halves of the histogram
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(binary_warped.shape[0]/nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Visualize binary warped image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
ax1.imshow(cv2.cvtColor(ImgC,cv2.COLOR_BGR2RGB))
ax1.set_title('Undistorted Image', fontsize=30)
ax2.imshow(binary_warped, cmap="gray")
ax2.plot(left_fitx, ploty, color='red',linewidth=6.0)
ax2.plot(right_fitx, ploty, color='blue',linewidth=6.0)
ax2.xlim(0, 1280)
ax2.ylim(720, 0)
ax2.set_title('Fitted lanes', fontsize=30)

## 5. Calculate the radius of curvature

y_eval = np.max(ploty)
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters
print('Left lane curve radius: ', left_curverad, 'm')
print('Right lane curve radius: ', right_curverad, 'm')
curvature = (left_curverad + right_curverad) / 2
min_curverad = min(left_curverad, right_curverad)
# Calculate offset from the center
y_level = 719 # at the bottom of the image = nearest to the camera
img_center = img_size[0]/2
left_lanex = left_fit[0]*y_level**2 + left_fit[1]*y_level + left_fit[2]
right_lanex = right_fit[0]*y_level**2 + right_fit[1]*y_level + right_fit[2]
lane_center = (left_lanex + right_lanex)/2
offset = (lane_center-img_center)*xm_per_pix
print('Offset: ', offset, 'm off to the right')

## 6. Masking lane area

def show_parameters(img,curvature,offset,min_curverad):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Radius of Curvature = %d(m)' % curvature, (50, 50), font, 1.5, (255, 255, 255), 2)
    left_or_right = "left" if offset > 0 else "right"
    cv2.putText(img, 'Vehicle is %.2fm %s of center' % (np.abs(offset), left_or_right), (50, 100), font, 1.5,
                (255, 255, 255), 2)
    cv2.putText(img, 'Minimum Radius of Curvature = %d(m)' % min_curverad, (50, 150), font, 1.5, (255, 255, 255), 2)

# Create an image to draw the lines on
warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (ImgC.shape[1], ImgC.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(ImgC, 1, newwarp, 0.3, 0)
# Visualize binary warped image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
ax1.imshow(cv2.cvtColor(ImgC,cv2.COLOR_BGR2RGB))
ax1.set_title('Undistorted Image', fontsize=30)
show_parameters(result,curvature=curvature,offset=offset,min_curverad=min_curverad)
ax2.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
ax2.set_title('Masked Lane Area', fontsize=30)

lf_initial = left_fit
rf_initial = right_fit

## 6. Video code

# Initial condition to prepar for smoothing over 5 movie frames
lf_i_minus_1 = lf_initial
lf_i_minus_2 = lf_initial
lf_i_minus_3 = lf_initial
lf_i_minus_4 = lf_initial
rf_i_minus_1 = rf_initial
rf_i_minus_2 = rf_initial
rf_i_minus_3 = rf_initial
rf_i_minus_4 = rf_initial

def image_pipeline(Img):   
    global lf_i_minus_1,lf_i_minus_2,lf_i_minus_3,lf_i_minus_4,rf_i_minus_1,rf_i_minus_2,rf_i_minus_3,rf_i_minus_4
    ImgC = cv2.undistort(Img, mtx, dist, None, mtx)
    gradx = abs_sobel_thresh(ImgC,orient='x',sobel_kernel=ksize,thresh=(20,100))
    grady = abs_sobel_thresh(ImgC,orient='y',sobel_kernel=ksize,thresh=(5,100))
    mag_binary = mag_threshold(ImgC,sobel_kernel=ksize,mag_thresh=(30,100))
    dir_binary = dir_threshold(ImgC,sobel_kernel=ksize,thresh=(0,1.3))
    color_binary = color_thresh(ImgC,thr_s=(200, 255),thr_r=(170, 255))
    combined_binary = np.zeros_like(dir_binary)
    combined_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1) ] = 1
    masked_image = region_of_interest(combined_binary,vertices)
    binary_warped = warper(masked_image,src,dst)    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = ((nonzerox > (lf_i_minus_1[0]*(nonzeroy**2) + lf_i_minus_1[1]*nonzeroy + lf_i_minus_1[2] - margin)) & (nonzerox < (lf_i_minus_1[0]*(nonzeroy**2) + lf_i_minus_1[1]*nonzeroy + lf_i_minus_1[2] + margin))) 
    right_lane_inds = ((nonzerox > (rf_i_minus_1[0]*(nonzeroy**2) + rf_i_minus_1[1]*nonzeroy + rf_i_minus_1[2] - margin)) & (nonzerox < (rf_i_minus_1[0]*(nonzeroy**2) + rf_i_minus_1[1]*nonzeroy + rf_i_minus_1[2] + margin)))  
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # 2nd polynomial fit
    lf_i = np.polyfit(lefty, leftx, 2)
    rf_i = np.polyfit(righty, rightx, 2)
     # If the current fit is way off compared with the previous fit, use the previous fit
    lf_diff = abs(lf_i - lf_i_minus_1)
    if (lf_diff[0]>0.001 or lf_diff[1]>1.0 or lf_diff[2]>100.) and len(lf_i)>0:
        lf_i = lf_i_minus_1
    rf_diff = abs(rf_i - rf_i_minus_1)
    if (rf_diff[0]>0.001 or rf_diff[1]>1.0 or rf_diff[2]>100.) and len(rf_i)> 0:
        rf_i = rf_i_minus_1
    # Smooth over 5 movie frames
    left_fit = np.average([lf_i,lf_i_minus_1,lf_i_minus_2,lf_i_minus_3,lf_i_minus_4],axis=0)
    right_fit = np.average([rf_i,rf_i_minus_1,rf_i_minus_2,rf_i_minus_3,rf_i_minus_4],axis=0)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    curvature = (left_curverad + right_curverad) / 2
    min_curverad = min(left_curverad, right_curverad)
    img_center = img_size[0]/2
    left_lanex = left_fit[0]*y_level**2 + left_fit[1]*y_level + left_fit[2]
    right_lanex = right_fit[0]*y_level**2 + right_fit[1]*y_level + right_fit[2]
    lane_center = (left_lanex + right_lanex)/2
    offset = (lane_center-img_center)*xm_per_pix
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (ImgC.shape[1], ImgC.shape[0])) 
    result = cv2.addWeighted(ImgC, 1, newwarp, 0.3, 0)
    show_parameters(result,curvature=curvature,offset=offset,min_curverad=min_curverad)
    # Renew coefficients for the next movie frame
    lf_i_minus_1 = left_fit
    lf_i_minus_2 = lf_i_minus_1
    lf_i_minus_3 = lf_i_minus_2
    lf_i_minus_4 = lf_i_minus_3
    rf_i_minus_1 = right_fit
    rf_i_minus_2 = rf_i_minus_1
    rf_i_minus_3 = rf_i_minus_2
    rf_i_minus_4 = rf_i_minus_3
    return result

## sanity check
#result_test = image_pipeline(Img)
#plt.imshow(cv2.cvtColor(result_test,cv2.COLOR_BGR2RGB))

# Create video
from moviepy.editor import VideoFileClip
margin = 50
output = 'project_video_processed02.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(image_pipeline)
output_clip.write_videofile(output,audio=False)