import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import pickle
import io
import os
import glob
#%matplotlib inline
from skimage.feature import hog
from scipy.ndimage import label
from features import *
import time
import json

resize = False
input_file_name= 'project_video.mp4'

#### START - FUNCTION TO READ AN INPUT IMAGE ###################################
def readVideo():

    # Read input video from current working directory
    inpImage = cv2.VideoCapture('test_videos/project_video.mp4')

    return inpImage

def resize_frame(frame, width=1280, height=720):
    return cv2.resize(frame, (width, height))

def camera_calibration():
    images = glob.glob('camera_cal/calibration*.jpg')
    img = mpimg.imread(images[0])
    plt.imshow(img);
    
    # store chessboard coordinates
    chess_points = []
    # store points from transformed img
    image_points = []
    
    # board is 6 rows by 9 columns. each item is one (xyz) point 
    # remember, only care about inside points. that is why board is 9x6, not 10x7
    chess_point = np.zeros((9*6, 3), np.float32)
    # z stays zero. set xy to grid values
    chess_point[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    for image in images:
        
        img = mpimg.imread(image)
        # convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        
        # returns boolean and coordinates
        success, corners = cv.findChessboardCorners(gray, (9,6), None)
        
        if success:
            image_points.append(corners)
            #these will all be the same since it's the same board
            chess_points.append(chess_point)
        else:
            print('corners not found {}'.format(image))
    		
    image = mpimg.imread('./camera_cal/calibration2.jpg')
    
    plt.figure();
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10));
    ax1.imshow(image);
    ax1.set_title('Captured Image', fontsize=30);
    
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
 
    ret , corners = cv.findChessboardCorners(gray,(9,6),None)    
    if ret == False:
        print('corners not found')
    img = cv.imread('./camera_cal/calibration2.jpg')
    img1 = cv.drawChessboardCorners(img,(9,6),corners,ret) 
    
    ax2.imshow(img1);
    ax2.set_title('Corners drawn Image', fontsize=30);
    plt.tight_layout();
    plt.savefig('saved_figures/chess_corners.png');
    plt.show;

# Save everything!
# points_pkl = {}
# points_pkl["chesspoints"] = chess_points
# points_pkl["imagepoints"] = image_points
# points_pkl["imagesize"] = (img.shape[1], img.shape[0])
# pickle.dump(points_pkl,open("object_and_image_points.pkl", "wb" ))

def distort_correct(img,mtx,dist,camera_img_size):
    img_size1 = (img.shape[1],img.shape[0])
    #print(img_size1)
    #print(camera_img_size)
    assert (img_size1 == camera_img_size),'image size is not compatible'
    undist = cv.undistort(img, mtx, dist, None, mtx)
    return undist

def distortion_correction():
    points_pickle = pickle.load( open( "object_and_image_points.pkl", "rb" ) )
    chess_points = points_pickle["chesspoints"]
    image_points = points_pickle["imagepoints"]
    img_size = points_pickle["imagesize"]
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(chess_points, image_points, img_size, None, None)

    # camera = {}
    # camera["mtx"] = mtx
    # camera["dist"] = dist
    # camera["imagesize"] = img_size
    # pickle.dump(camera, open("camera_matrix.pkl", "wb"))
    	
    img = mpimg.imread('./camera_cal/calibration2.jpg')
    img_size1 = (img.shape[1], img.shape[0])
    
    undist = distort_correct(img, mtx, dist, img_size)
    
    ### Visualize the captured 
    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10));
    ax1.imshow(img);
    ax1.set_title('Captured Image', fontsize=30);
    ax2.imshow(undist);
    ax2.set_title('Undistorted Image', fontsize=30);
    plt.tight_layout()
    plt.savefig('saved_figures/undistorted_chess.png')
    
    # load camera matrix and distortion matrix
    camera = pickle.load(open( "camera_matrix.pkl", "rb" ))
    mtx = camera['mtx']
    dist = camera['dist']
    camera_img_size = camera['imagesize']
    
    # get an undistorted dashcam frame
    image = mpimg.imread('test_images/test1.jpg')
    image = distort_correct(image,mtx,dist,camera_img_size)
    plt.imshow(image);

def abs_sobel_thresh(img, orient='x', thresh=(0,255)):
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

def test():	
    # set color map to gray, default is RGB
    plt.imshow(abs_sobel_thresh(image, thresh=(20,110)),  cmap='gray');
    plt.imshow(mag_threshold(image, thresh=(20,100)),  cmap='gray');
    plt.imshow(dir_threshold(image, thresh=(0.8,1.2)),  cmap='gray');
    plt.imshow(hls_select(image, sthresh=(140,255), lthresh=(120, 255)),  cmap='gray');
    plt.imshow(red_select(image, thresh=(200,255)),  cmap='gray');
    result = binary_pipeline(image)

    # Plot the result
    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)
    
    ax2.imshow(result, cmap='gray')
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.tight_layout()
    plt.savefig('saved_figures/combined_filters.png')
    
    # get an undistorted dashcam frame
    image = mpimg.imread('test_images/test5.jpg')
    image = distort_correct(image,mtx,dist,camera_img_size)
    plt.imshow(image);
    
    result = binary_pipeline(image)
    
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)
    
    ax2.imshow(result, cmap='gray')
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	
    birdseye_result, inverse_perspective_transform = warp_image(result)

    # Plot the result
    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    
    
    image_size = (image.shape[1], image.shape[0])
    x = image.shape[1]
    y = image.shape[0]
    source_points = np.int32([
                        [0.117 * x, y],
                        [(0.5 * x) - (x*0.078), (2/3)*y],
                        [(0.5 * x) + (x*0.078), (2/3)*y],
                        [x - (0.117 * x), y]
                        ])
    
    draw_poly = cv.polylines(image,[source_points],True,(255,0,0), 5)
    
    ax1.imshow(draw_poly)
    ax1.set_title('Source', fontsize=40)
    ax2.imshow(birdseye_result, cmap='gray')
    ax2.set_title('Destination', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.tight_layout()
    plt.savefig('saved_figures/perspective_transform.png')
    
    # reload the image for later use
    image = mpimg.imread('test_images/test5.jpg')
    image = distort_correct(image,mtx,dist,camera_img_size)
    
    #look at only lower half of image
    histogram = np.sum(birdseye_result[int(birdseye_result.shape[0]/2):,:], axis=0)
    plt.figure();
    plt.plot(histogram);
    plt.savefig('saved_figures/lane_histogram.png')
	
    left_fit,right_fit = track_lanes_initialize(birdseye_result)

    Image('saved_figures/01_window_search.png')
    
    global frame_count
    frame_count=0
    left_fit,right_fit,leftx,lefty,rightx,righty = track_lanes_update(birdseye_result, left_fit,right_fit)

    Image('saved_figures/02_updated_search_window.png')
    
    measure_curve(birdseye_result,left_fit, right_fit)
    
    colored_lane = lane_fill_poly(birdseye_result, image, left_fit, right_fit)
    plt.figure()
    plt.imshow(colored_lane);
    plt.tight_layout()
    plt.savefig('saved_figures/lane_polygon.png')
    
    offset = vehicle_offset(colored_lane, left_fit, right_fit)
    print(offset)

def mag_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the xy magnitude 
    mag = np.sqrt(x**2 + y**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale = np.max(mag)/255
    eightbit = (mag/scale).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(eightbit)
    binary_output[(eightbit > thresh[0]) & (eightbit < thresh[1])] =1 
    return binary_output
	

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 1) Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    x = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel))
    y = np.absolute(cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel))
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direction = np.arctan2(y, x)
    binary_output = np.zeros_like(direction)
    binary_output[(direction > thresh[0]) & (direction < thresh[1])] = 1
    return binary_output

def hls_select(img, sthresh=(0, 255),lthresh=()):
    # 1) Convert to HLS color space
    hls_img = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    L = hls_img[:,:,1]
    S = hls_img[:,:,2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(S)
    binary_output[(S >= sthresh[0]) & (S <= sthresh[1])
                 & (L > lthresh[0]) & (L <= lthresh[1])] = 1
    return binary_output

def red_select(img, thresh=(0, 255)):
    # Apply a threshold to the R channel
    R = img[:,:,0]
    # Return a binary image of threshold result
    binary_output = np.zeros_like(R)
    binary_output[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary_output

def binary_pipeline(img):
    
    img_copy = cv.GaussianBlur(img, (3, 3), 0)
    #img_copy = np.copy(img)
    
    # color channels
    s_binary = hls_select(img_copy, sthresh=(140, 255), lthresh=(120, 255))
    #red_binary = red_select(img_copy, thresh=(200,255))
    
    # Sobel x
    x_binary = abs_sobel_thresh(img_copy,thresh=(25, 200))
    y_binary = abs_sobel_thresh(img_copy,thresh=(25, 200), orient='y')
    xy = cv.bitwise_and(x_binary, y_binary)
    
    #magnitude & direction
    mag_binary = mag_threshold(img_copy, sobel_kernel=3, thresh=(30,100))
    dir_binary = dir_threshold(img_copy, sobel_kernel=3, thresh=(0.8, 1.2))
    
    # Stack each channel
    gradient = np.zeros_like(s_binary)
    gradient[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    final_binary = cv.bitwise_or(s_binary, gradient)
    
    return final_binary

def warp_image(img):
    
    image_size = (img.shape[1], img.shape[0])
    x = img.shape[1]
    y = img.shape[0]

    #the "order" of points in the polygon you are defining does not matter
    #but they need to match the corresponding points in destination_points!
    source_points = np.float32([
    [0.117 * x, y],
    [(0.5 * x) - (x*0.078), (2/3)*y],
    [(0.5 * x) + (x*0.078), (2/3)*y],
    [x - (0.117 * x), y]
    ])

    file_path='point.json'
    with open(file_path,'r') as file:
        data=json.load(file)

    if input_file_name in data:
        print('test2.mp4')
        source_points=np.float32(data[input_file_name])
    else:
        print(f"{input_file_name} key not found in JSON file")


















    #snap = cv.imread('test_images/a.jpg')
    #snap = cv.resize(snap, (snap.shape[1]//4, snap.shape[0]//4), cv.INTER_AREA)
    #cv.circle(snap, (int(source_points[0, 0]), int(source_points[0, 1])), 5, (0, 0, 255), -1)
    #cv.circle(snap, (int(source_points[1, 0]), int(source_points[1, 1])), 5, (0, 0, 255), -1)
    #cv.circle(snap, (int(source_points[2, 0]), int(source_points[2, 1])), 5, (0, 0, 255), -1)
    #cv.circle(snap, (int(source_points[3, 0]), int(source_points[3, 1])), 5, (0, 0, 255), -1)
    #cv.imshow('Points', snap)

#     #chicago footage
#     source_points = np.float32([
#                 [300, 720],
#                 [500, 600],
#                 [700, 600],
#                 [850, 720]
#                 ])
    
#     destination_points = np.float32([
#                 [200, 720],
#                 [200, 200],
#                 [1000, 200],
#                 [1000, 720]
#                 ])
    
    destination_points = np.float32([
    [0.25 * x, y],
    [0.25 * x, 0],
    [x - (0.25 * x), 0],
    [x - (0.25 * x), y]
    ])
    
    perspective_transform = cv.getPerspectiveTransform(source_points, destination_points)
    inverse_perspective_transform = cv.getPerspectiveTransform( destination_points, source_points)
    
    warped_img = cv.warpPerspective(img, perspective_transform, image_size, flags=cv.INTER_LINEAR)
    
    #print(source_points)
    #print(destination_points)
    
    return warped_img, inverse_perspective_transform

def track_lanes_initialize(binary_warped):
    
    global window_search
    
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    #print(f'binary_warped.shape {binary_warped.shape}')
    #print(f'histogram.shape {histogram.shape}')
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # we need max for each half of the histogram. the example above shows how
    # things could be complicated if didn't split the image in half 
    # before taking the top 2 maxes
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #print(f'leftx_base: {leftx_base}, rightx_base: {rightx_base}')
    
    # Choose the number of sliding windows
    # this will throw an error in the height if it doesn't evenly divide the img height
    nwindows = 9
    # Set height of windows
    window_height = int(binary_warped.shape[0]/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    #print(nonzerox.shape)
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    if resize:
        margin = 100//4
        minpix = 50//4
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    
    t1 = time.time()
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
        win_y_high = int(binary_warped.shape[0] - window*window_height)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3) 
        cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    
    #cv.imshow('Slide', out_img)
    t2 = time.time()
    #print('\tLoop', t2 - t1)
            
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
    #ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    #left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    #right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    #ltx = np.trunc(left_fitx)
    #rtx = np.trunc(right_fitx)
    #plt.plot(right_fitx)
    # plt.show()

    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    #plt.imshow(out_img)
    #plt.plot(left_fitx,  ploty, color = 'yellow')
    #plt.plot(right_fitx, ploty, color = 'yellow')
    #plt.xlim(0, binary_warped.shape[1])
    #plt.ylim(binary_warped.shape[0], 0)
    #plt.show()

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit,right_fit

def track_lanes_update(binary_warped, left_fit,right_fit):

    global window_search
    global frame_count
    
    # repeat window search to maintain stability
    if frame_count % 10 == 0:
        window_search=True
   
        
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
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


    return left_fit,right_fit,leftx,lefty,rightx,righty

# A function to get quadratic polynomial output
def get_val(y,poly_coeff):
    return poly_coeff[0]*y**2+poly_coeff[1]*y+poly_coeff[2]
	
import numpy as np
import cv2 as cv

def get_val(ploty, fit):
    # Function to calculate x values from polynomial fit coefficients
    return fit[0] * ploty**2 + fit[1] * ploty + fit[2]

def lane_fill_poly(binary_warped, undist, left_fit, right_fit, inverse_perspective_transform, offcenter, resize=False):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = get_val(ploty, left_fit)
    right_fitx = get_val(ploty, right_fit)

    # Create an empty image to draw the lane
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast x and y for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane
    threshold = 0.6
    if abs(offcenter) > threshold:  # Car is off-center more than 0.6 m
        # Draw Red lane (note BGR format for OpenCV)
        cv.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))  # Red
    else:  # Draw Green lane
        cv.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))  # Green

    # Warp using inverse perspective transform
    newwarp = cv.warpPerspective(color_warp, inverse_perspective_transform, (binary_warped.shape[1], binary_warped.shape[0]))
    
    # Display the warped image for debugging (optional)
    cv.imshow('warpPerspective', newwarp)
    
    # Overlay the lane on the original image
    if resize:
        newwarp = cv.resize(newwarp, (newwarp.shape[1] * 4, newwarp.shape[0] * 4), cv.INTER_AREA)
    result = cv.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    return result


def measure_curve(binary_warped,left_fit,right_fit):
        
    # generate y values 
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    # measure radius at the maximum y value, or bottom of the image
    # this is closest to the car 
    y_eval = np.max(ploty)
    
    # coversion rates for pixels to metric
    # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
    # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    if resize:
        ym_per_pix = (30/720)*4 # meters per pixel in y dimension
        xm_per_pix = (3.7/700)*4 # meters per pixel in x dimension
   
    # x positions lanes
    leftx = get_val(ploty,left_fit)
    rightx = get_val(ploty,right_fit)

    # fit polynomials in metric 
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # calculate radii in metric from radius of curvature formula
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # averaged radius of curvature of left and right in real world space
    # should represent approximately the center of the road
    curve_rad = round((left_curverad + right_curverad)/2)
    
    return curve_rad

    
def vehicle_offset(img,left_fit,right_fit):
    
    # THIS RATE CAN CHANGE GIVEN THE RESOLUTION OF THE CAMERA!!!!!
    # BE SURE TO CHANGE THIS IF USING DIFFERENT SIZE IMAGES!!!
    xm_per_pix = 3.7/700 
    image_center = img.shape[1]/2
    
    ## find where lines hit the bottom of the image, closest to the car
    left_low = get_val(img.shape[0],left_fit)
    right_low = get_val(img.shape[0],right_fit)
    
    # pixel coordinate for center of lane
    lane_center = (left_low+right_low)/2.0
    
    ## vehicle offset
    distance = image_center - lane_center
    
    ## convert to metric
    return (round(distance*xm_per_pix,5))

camera = pickle.load(open( "camera_matrix.pkl", "rb" ))

def add_information_box(processed_frame, left_fit, right_fit, curve_radius, offset):
    # Define box dimensions
    box_width = 400
    box_height = 450

    # Create a transparent overlay for the box
    box_overlay = np.zeros((box_height, box_width, 3), dtype=np.uint8)

    # Add a red outline to the box
    cv2.rectangle(box_overlay, (0, 0), (box_width - 1, box_height - 1), (0, 0, 255), 2)

    # Load direction images
    straight_img = cv2.imread("straight.png", cv2.IMREAD_COLOR)
    left_img = cv2.imread("image_left.png", cv2.IMREAD_COLOR)
    right_img = cv2.imread("image_right.png", cv2.IMREAD_COLOR)

    # Resize direction images to fit the box
    direction_img_size = (180, 180)
    straight_img = cv2.resize(straight_img, direction_img_size)
    left_img = cv2.resize(left_img, direction_img_size)
    right_img = cv2.resize(right_img, direction_img_size)

    # Determine direction based on the curvature
    if abs(left_fit[0]) > abs(right_fit[0]):
        value = left_fit[0]
    else:
        value = right_fit[0]

    if abs(value) <= 0.00015:
        direction = 'F'
    elif value < 0:
        direction = 'L'
    else:
        direction = 'R'

    W = 400
    H = 500
    widget = np.copy(processed_frame[:H, :W])
    widget //= 2
    widget[0, :] = [0, 0, 255]
    widget[-1, :] = [0, 0, 255]
    widget[:, 0] = [0, 0, 255]
    widget[:, -1] = [0, 0, 255]
    processed_frame[:H, :W] = widget

    if direction == 'L':
        y, x = np.where(left_img[:, :, 0] != 0)
        processed_frame[y, x - 100 + W // 2] = left_img[y, x]
    elif direction == 'R':
        y, x = np.where(right_img[:, :, 0] != 0)
        processed_frame[y, x - 100 + W // 2] = right_img[y, x]
    else:
        y, x = np.where(straight_img[:, :, 0] != 0)
        processed_frame[y, x - 100 + W // 2] = straight_img[y, x]

    msg = "Keep Straight Ahead"
    curvature_msg = "Curvature = {:.0f} m".format(curve_radius)
    if direction == 'L':
        msg = "Left Curve Ahead"
    elif direction == 'R':
        msg = "Right Curve Ahead"

    cv2.putText(processed_frame, msg, org=(10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    if direction in ['L', 'R']:
        cv2.putText(processed_frame, curvature_msg, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.putText(
            processed_frame,
            "Good Lane Keeping",
            org=(10, 400),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.2,
            color=(0, 255, 0),
            thickness=2)

    cv2.putText(
        processed_frame,
        "Vehicle is {:.2f} m away from center".format(offset),
        org=(10, 450),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.63,
        color=(255, 255, 255),
        thickness=2)
    
    return processed_frame

def img_pipeline(img):
    
    global window_search
    global left_fit_prev
    global right_fit_prev
    global frame_count
    global curve_radius
    global offset
    
    img = cv.resize(img, (1280, 720))
    #plt.imshow(img)
    #plt.show()
    # load camera matrix and distortion matrix
    t1 = time.time()
    mtx = camera['mtx']
    dist = camera['dist']
    camera_img_size = camera['imagesize']
    
    #correct lens distortion
    undist = distort_correct(img,mtx,dist,camera_img_size)
    orig_img = undist
    if resize:
        undist = cv.resize(undist, (undist.shape[1]//4, undist.shape[0]//4), cv.INTER_AREA)
    t2 = time.time()
    #print('Camera correct', t2 - t1)

    t1 = time.time()
    # get binary image
    binary_img = binary_pipeline(undist)
    t2 = time.time()
    #print('Binary Pipeline', t2 - t1)

    t1 = time.time()
    #perspective transform
    birdseye, inverse_perspective_transform = warp_image(binary_img)
    t2 = time.time()
    #print('WarpImage', t2 - t1)
    #cv.imshow('Birdseye', birdseye)
    #cv.imshow('B_img', binary_img)
    
    t1 = time.time()
    if window_search:
        #window_search = False
        #window search
        left_fit,right_fit = track_lanes_initialize(birdseye)
        #store values
        left_fit_prev = left_fit
        right_fit_prev = right_fit
        
    else:
        #load values
        left_fit = left_fit_prev
        right_fit = right_fit_prev
        #search in margin of polynomials
        left_fit,right_fit,leftx,lefty,rightx,righty = track_lanes_update(birdseye, left_fit,right_fit)
    t2 = time.time()
    #print('TrackLane', t2 - t1)
    
    #save values
    left_fit_prev = left_fit
    right_fit_prev = right_fit
    
    #update ~twice per second
    if frame_count==0 or frame_count%15==0:
        t1 = time.time()
        #measure radii
        curve_radius = measure_curve(birdseye,left_fit,right_fit)
        t2 = time.time()
        #print('MeasureCurve', t2 - t1)

        t1 = time.time()
        #measure offset
        offset = vehicle_offset(undist, left_fit, right_fit)
        t2 = time.time()
        #print('VehicleOffset', t2 - t1)
    
    t1 = time.time()
    #draw polygon
    processed_frame = lane_fill_poly(birdseye, orig_img, left_fit, right_fit, inverse_perspective_transform, offset)
    t2 = time.time()
    #print('LaneFIllPoly', t2 - t1)
	
    #if resize:
    #    processed_frame = cv.resize(processed_frame, (processed_frame.shape[1]*4, processed_frame.shape[0]*4), cv.INTER_AREA)

    #printing information to frame
    processed_frame = add_information_box(processed_frame, left_fit, right_fit, curve_radius, offset)
   
    frame_count += 1
    return processed_frame

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv.cvtColor(img, cv.COLOR_RGB2LUV)
		
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap
	
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
	
def draw_labeled_bboxes(img, labels):
    box_list = []
    
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        # could be [0,255] or [0,1] depending how the file was read in
        cv.rectangle(img, bbox[0], bbox[1], (255,0,0), 6)
        
        box_list.append(bbox)

    return img, box_list
	
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    #draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    heat = np.zeros_like(img[:,:,0]).astype(float)
    bboxes = []
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv.resize(ctrans_tosearch, (int(imshape[1]/scale), int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = int(xleft*scale)
                ytop_draw = int(ytop*scale)
                win_draw = int(window*scale)
                bboxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
                #if you want to draw hog subsampling boxes
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
    
    #add up heat overlap 
    heatmap = add_heat(heat, bboxes)
                
    return heatmap, bboxes #, draw_img
	
def track_vehicles(img):
    
    global draw_img_prev, bbox_list_prev, labels_prev, heatmap_sum
    global first_frame, frame_count
    
    model_pickle = pickle.load(open('00-training.pkl', 'rb'))
    svc = model_pickle['svc']
    #print(type(svc))
    X_scaler = model_pickle['scaler']
    orient = model_pickle['orient']
    pix_per_cell = model_pickle['pix_per_cell'] 
    cell_per_block = model_pickle['cell_per_block']
    spatial_size = model_pickle['spatial_size']
    hist_bins = model_pickle['hist_bins']
    
    #this could be changed relative to image size
    ystart = 400
    ystop = 420
    
    if first_frame:
        
        #initialize the running average heatmap
        heatmap_sum = np.zeros_like(img[:,:,0]).astype(float)
        
        for scale in (1.0,1.5,2.0):
            
            #as the image scale gets bigger, the ROI needs to extend
            ystop = ystop + 75

            heatmap, bboxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, 
                                        pix_per_cell, cell_per_block, spatial_size, hist_bins)
            #sum up heatmap for all scales
            heatmap_sum = np.add(heatmap_sum,heatmap)
        
        heatmap = apply_threshold(heatmap_sum, 2)
        heatmap = np.clip(heatmap, 0, 1)
        labels = label(heatmap)
        draw_img, bbox_list = draw_labeled_bboxes(np.copy(img), labels)

        draw_img_prev = draw_img
        bbox_list_prev = bbox_list
        labels_prev = labels

        first_frame = False
    
        return draw_img
    
    if frame_count <= 2:
        
        frame_count += 1
        
        #reset ROI
        ystop = 420
        
        for scale in (1.0,1.5,2.0):

            ystop = ystop + 75
            heatmap, bboxes = find_cars(img, ystart, ystop, scale, svc, X_scaler,
                                                            orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
            heatmap_sum = np.add(heatmap_sum,heatmap)
            draw_img = np.copy(img)
        
        #draw old boxes
        for unique_car in range(1, labels_prev[1]+1):
            draw_img = cv.rectangle(draw_img, bbox_list_prev[unique_car-1][0],
                                     bbox_list_prev[unique_car-1][1], (255,0,0), 6)
        return draw_img
    
    heatmap = apply_threshold(heatmap_sum,2)
    heatmap = np.clip(heatmap, 0, 255)
    labels = label(heatmap)   
    draw_img, bbox_list = draw_labeled_bboxes(np.copy(img), labels)   
    
    draw_img_prev = draw_img
    bbox_list_prev = bbox_list
    labels_prev = labels
    
    #reset heatmap sum and frame_count
    heatmap_sum = np.zeros_like(img[:,:,0]).astype(float)
    frame_count = 0
    
    return draw_img

def video_pipeline(image):
    lane_image = img_pipeline(image)
    global first_frame
    frame_count = 0
    first_frame = True
    draw = track_vehicles(lane_image)
    return draw
	
def test_frames():
    filenames = os.listdir("test_images/")
    global window_search
    global frame_count
    for filename in filenames:
        frame_count = 15
        window_search = True
        image = mpimg.imread('test_images/'+filename)
        image = cv.imread('test_images/'+filename)
        t1 = time.time()
        lane_image = img_pipeline(image)
        t2 = time.time()
        print('ImgPipeline', t2 - t1)
        #lane_image = cv.cvtColor(lane_image, cv.COLOR_RGB2BGR)
        #cv.imshow('Img', lane_image)
        global first_frame
        frame_count = 0
        first_frame = True
        #draw = track_vehicles(lane_image)
        #mpimg.imsave('output_images/lane_'+filename,draw)
        #print('output_images/lane_'+filename)
        cv2.imshow('Draw', lane_image)
        break
	
    #Image('output_images/lane_test2.jpg')
    cv.waitKey(0)

def test_video():
    from moviepy.video.fx.all import crop
    global window_search 
    global frame_count
    window_search = True
    frame_count = 0
    
    #chicago footage
    for filename in ['project_video.mp4']:
        clip = VideoFileClip('videos/'+filename)#.subclip((3,25),(3,35))
        #clip_crop = crop(clip, x1=320, y1=0, x2=1600, y2=720)
        out = clip.fl_image(video_pipeline)
        #project_video_clip = clip.fl_image(out)
        #out = clip_crop.fl_image(img_pipeline)
        out.write_videofile('videos/processed_'+filename, audio=False, verbose=False)
        print('Success!')
    	
    video = 'videos/processed_drive.mp4'
    HTML("""
    <video width="960" height="540" controls>
      <source src="{0}">
    </video>
    """.format(video))

def video_process():
    global window_search 
    global frame_count
    window_search = True
    frame_count = 0

    image = readVideo()
    t1 = time.time()
    num_famres = 0

    while True:
        t2 = time.time()
        num_famres += 1
        if t2 - t1 > 1:
            print("FPS: ", num_famres)
            num_famres = 0
            t1 = t2
        has, frame = image.read()
        if has:
            #cv2.imwrite(f'snaps/image{0}.png', frame)
            #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            t3 = time.time()
            frame = img_pipeline(frame)
            t4 = time.time()
            #print('ImgPipeline {}\n'.format(t2 - t1))
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) == 27:
                break
        else:
            break

    # Cleanup
    image.release()
    cv2.destroyAllWindows()
	
def main():
    #camera_calibration()
    global resize
    #resize = True
    #test_frames()
    #test_video()
    video_process()
	
if __name__ == '__main__':
    main()