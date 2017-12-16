import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lessons import *
from scipy.ndimage.measurements import label
from collections import deque
# NOTE: the next import is only valid for scikit-learn version <=0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

history = deque(maxlen=10)
frames = deque(maxlen=10)
img = cv2.imread('test_images/test1.jpg')
video = cv2.VideoCapture('project_video.mp4')
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('output.mp4', fourcc, 20, (1280, 720))
previous = []

### TODO Tweak these parameters and see how the results change
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# color_space = 'YUV' # not as good
orient = 11
#orient = 9 # HOG pixels per cell
# pix_per_cell = 8
pix_per_cell = 16 # HOG pixels per cell
# cell_per_block = 4 # HOG cells per block
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2 or "ALL"
spatial_size = (32, 32)
#spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16 # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than a list of images
def single_img_features(img, color_space=color_space, spatial_size=spatial_size,
						hist_bins=hist_bins, orient=orient,
						pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel='ALL',
						spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat):
	#1) Define an empty list to receive features
	img_features = []
	#2) Apply color conversion if other than 'RGB'
	if color_space != 'RGB':
		if color_space != 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif color_space == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else: feature_image = np.copy(img)
	#3) Compute spatial features if flag is set
	if spatial_feat == True:
		spatial_features = bin_spatial(feature_image, size=spatial_size)
		#4) Append features to list
		img_features.append(spatial_features)
	#5) Compute histogram features if flag is set
	if hist_feat == True:
		hist_features = color_hist(feature_image, nbins=hist_bins)
		#6) Append features to list
		img_features.append(hist_features)
	#7) Compute HOG features if flag is set
	if hog_feat == True:
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.extend(get_hog_features(feature_image[:,:,channel],
														orient, pix_per_cell, cell_per_block,
														vis= False, feature_vec=True))
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
						pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		#8) Append features to list
		img_features.append(hog_features)

	#9) Return concatenated array of features
	return np.concatenate(img_features)

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space=color_space,
					spatial_size=spatial_size, hist_bins=hist_bins,
					pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
					hog_channel=hog_channel, spatial_feat=spatial_feat,
					hist_feat=hist_feat, hog_feat=hog_feat):

	#1) Create an empty list  to receive positive detection windows
	on_windows = []
	#2) Iterate over all windows in the list
	for window in windows:
		#3) Extract the test window from original image
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
		#4) Extract features for that window using single_img_features()
		features = single_img_features(test_img, color_space=color_space,
					spatial_size=spatial_size, hist_bins=hist_bins,
					orient=orient,pix_per_cell=pix_per_cell,
					cell_per_block=cell_per_block,
					hog_channel=hog_channel, spatial_feat=spatial_feat,
					hist_feat=hist_feat, hog_feat=hog_feat)
		#5) Scale extracted features to be fed to classifier
		test_features = scaler.transform(np.array(features).reshape(1, -1))
		#6) Predict using your classifier
		prediction = clf.predict(test_features)
		
		#7) If positive (prediction = 1) then save the window
		if prediction == 1:
			on_windows.append(window)
	#8) Return windows for positive detections
	return on_windows

# Read in cars and notcars
'''
cars_path='vehicles/*/*.png'
#car_testing_data_path = 'vehicle_images_separated/*.png'
cars_images = glob.glob(cars_path)
car_testing_data = glob.glob(car_testing_data_path)
notcars_path='non-vehicles/*/*.png'
cars_images = glob.glob(cars_path)
notcars_images = glob.glob(notcars_path)
cars = []
notcars = []
cars_testing_data = []


for image in cars_images:
	cars.append(image)
for image in car_testing_data:
	cars_testing_data.append(image)
for image in notcars_images:
	notcars.append(image)
'''

# Read in cars and notcars
cars_path='vehicles/*/*.png'
cars_images = glob.glob(cars_path)
notcars_path='non-vehicles/*/*.png'
cars_images = glob.glob(cars_path)
notcars_images = glob.glob(notcars_path)
cars = []
notcars = []

for image in cars_images:
	cars.append(image)
for image in notcars_images:
	notcars.append(image)

# I don't need to reduce the sample size on my own machine
# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
#sample_size = 500
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]



car_features = extract_features(cars, color_space=color_space,
							spatial_size=spatial_size, hist_bins=hist_bins,
							orient=orient, pix_per_cell=pix_per_cell,
							cell_per_block=cell_per_block,
							hog_channel=hog_channel, hog_feat=hog_feat)
'''car_testing_features =  extract_features(cars_testing_data, color_space=color_space,
							spatial_size=spatial_size, hist_bins=hist_bins,
							orient=orient, pix_per_cell=pix_per_cell,
							cell_per_block=cell_per_block,
							hog_channel=hog_channel, hog_feat=hog_feat)
'''
notcar_features = extract_features(notcars, color_space=color_space,
						spatial_size=spatial_size, hist_bins=hist_bins,
						orient=orient, pix_per_cell=pix_per_cell,
						cell_per_block=cell_per_block,
						hog_channel=hog_channel, spatial_feat=spatial_feat,
						hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

# Apply the Scaler to X
scaled_X = X_scaler.transform(X)


# Define the labels vector
y  = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
	scaled_X, y, test_size=0.3, random_state=rand_state)


'''
X_train, y_train = scaled_X, y

X_test, y_test = scaled_X_test, y_test
'''
print('Using:', orient,'orientations', pix_per_cell,
	'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC(tol=.01,C=.8)
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print (round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
'''
image = mpimg.imread('bbox-example-image.jpg')
draw_image = np.copy(image)
# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255
windows = slide_window(image, x_start_stop=[None, None], y_start_stop=(450, 750),
					xy_window=(96, 96), xy_overlap=(0.5, 0.5))
hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
						spatial_space=spatial_size, hist_bins=hist_bins,
						orient=orient, pix_per_cell=pix_per_cell,
						cell_per_block=cell_per_block,
						hog_channel=hog_channel, spatial_feat=spatial_feat,
						hist_feat=hist_feat, hog_feat=hog_feat)
window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
plt.imshow(window_img)
'''

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    # print (img_tosearch.shape) (356, 1280, 3)
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = ((ch1.shape[1]) // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1

    # when set to 1 find the two cars at the end but many more false positives as opposed to 2
    # try changing this numbr of 4 to 8 pixels
    # 4 misses everything
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    

    bbox_list = []

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

            # print ("xleft ", xleft)
            # print ("ytop ", ytop)

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            # confidence of prediction
            
            
            # Filtering out boxes where xleft is less than 200 because that is on the wrong side of the road
            # originally was 200
            if test_prediction == 1 and xleft > 400 and svc.decision_function(test_features) > .5:
                # print (svc.decision_function(test_features))
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)

                bbox_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
                # display image for writeup
                display_img = cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
                # cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                # plt.imshow(display_img)
                # plt.show()

                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
    
    return bbox_list            
    #return draw_img

def add_heat(heatmap, bbox_list):

	# Iterate through list of boxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each box
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

		# display image for writeup
		# plt.imshow(heatmap, cmap='hot')
		# plt.show()
	
	# Return updated heatmap
	return heatmap # Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map

	# display image for writeup
	# plt.imshow(heatmap)
	# plt.show()
	return heatmap

ystart = 300
ystop = 656
xstart = 250
# .75 detects small images but also more false positives, try scale = 1
scale = 1.25
# scale = 1.5
# scale = 2

def draw_labeled_bboxes(img, labels):
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
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
	


		
		# commenting out previous because causing video to rewind
		# global previous
		# previous.append((bbox[0], bbox[1]))
		# print ("bbox[0]", bbox[0], "bbox[1]", bbox[1])

	# Return the image
	return img


def process_video(clip1):
	while (clip1.isOpened()):
		ret, frame = clip1.read()
		
		# cv2.imshow('frame',frame)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		#out_img = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
		heat = np.zeros_like(frame[:,:,0]).astype(np.float)
		bbox_list = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
		
		# Add heat to each box in box list
		heat = add_heat(heat, bbox_list)
		global history
		
		history.append(heat)
		heat_sum  = sum(history)/len(history)
		#heat = heat_sum/len(history)

		# Add threshold to help remove false positives
		heat = apply_threshold(heat_sum, 2)

		# Visualize the heatmap when displaying
		heatmap = np.clip(heat, 0, 255)
		global frames
		# use average frame to smooth boxe
		
		sum_frames = frame
		for f in frames:
			sum_frames += f
		avg_frame = sum_frames/(len(frames) +1)
		# Find final boxes from heatmap using label function
		labels = label(heatmap)
		avg_frame = avg_frame.astype('uint8')
		draw_img = draw_labeled_bboxes(np.copy(avg_frame), labels)
		#draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB )
		#cv2.imshow('draw_image',draw_img)
	
		out.write(draw_img)
		#out.write(out_img)

def process_image(img):
	
	frame = img
		#out_img = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
	heat = np.zeros_like(frame[:,:,0]).astype(np.float)
	bbox_list = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
		
	# Add heat to each box in box list
	heat = add_heat(heat, bbox_list)

	# Add threshold to help remove false positives
	heat = apply_threshold(heat, 1)

	# Visualize the heatmap when displaying
	heatmap = np.clip(heat, 0, 255)

	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(np.copy(frame), labels)
	out.write(draw_img)


# to test out images
#process_image(img)

# to test out video
process_video(video)


# ****  Decide what features to use - some combination of color and gradient

# Chose and train a classifier, LinearSVM probalby best but oculd try others as well

#  *****try with the original data set and not moving any of the images around

# add deque for bounding baxes to make them more stable as well

# video images are rgb but training images with cv2 are bgr so need to conver

# scaling of training images scaled from 0 to 1 or 0 to 255
 

# error when trying to average bboxes


# try to do tracking , eveyr tenth frame search
