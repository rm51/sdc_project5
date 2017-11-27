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
# NOTE: the next import is only valid for scikit-learn version <=0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

img = mpimg.imread('test_images/test1.jpg')

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than a list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
						hist_bins=32, orient=9,
						pix_per_cell=8, cell_per_block=2, hog_channel=0,
						spatial_feat=True, hist_feat=True, hog_feat=True):
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
def search_windows(img, windows, clf, scaler, color_space='RGB',
					spatial_size=(32, 32), hist_bins=32,
					pix_per_cell=8, cell_per_block=2,
					hog_channel=0, spatial_feat=True,
					hist_feat=True, hog_feat=True):

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
cars_path='vehicles_smallset/*/*.jpeg'
cars_images = glob.glob(cars_path)
notcars_path='non-vehicles_smallset/*/*.jpeg'
cars_images = glob.glob(cars_path)
notcars_images = glob.glob(notcars_path)
cars = []
notcars = []

for image in cars_images:
	cars.append(image)
for image in notcars_images:
		notcars.append(image)

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

### TODO Tweak these parameters and see how the results change
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9 # HOG pixels per cell
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 1 # Can be 0, 1, 2 or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16 # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space,
							spatial_size=spatial_size, hist_bins=hist_bins,
							orient=orient, pix_per_cell=pix_per_cell,
							cell_per_block=cell_per_block,
							hog_channel=hog_channel, hog_feat=hog_feat)
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
	scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient,'orientations', pix_per_cell,
	'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
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
	ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

	ch1 = ctrans_tosearch[:,:,0]
	ch2 = ctrans_tosearch[:,:,1]
	ch3 = ctrans_tosearch[:,:,2]

	# Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
	nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
	nfeat_per_block = orient*cell_per_block**2

	# 64 was the original sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 2 # Instead of overlap, define how many cells to step
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
			subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

			# Get color features
			spatial_features = bin_spatial(subimg, size=spatial_size)
			hist_features = color_hist(subimg, nbins=hist_bins)

			# Scale features and make a prediction
			test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
			test_prediction = svc.predict(test_features)

			if test_prediction == 1:
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)
				cv2.rectangle(draw_img, (xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw, ytop_draw+win_draw_ystart),(0,0,255),6)

	return draw_img

ystart = 400
ystop = 656
scale = 1.5

out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

plt.imshow(out_img)
plt.imshow()








































