import numpy as np 
import cv2
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

clip1 = cv2.VideoCapture('project_video.mp4')




def plot_image(img):
	scale = max(img.shape[0], img.shape[1],64) / 64
	img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

	# Convert subsampled image to desired color space(s)
	img_small_HLS = cv2.cvtColor(img_small, cv2.COLOR_BGR2HLS)  # OpenCV uses BGR, matplotlib likes RGB
	img_small_LUV = cv2.cvtColor(img_small, cv2.COLOR_BGR2LUV)
	img_small_hls = img_small_HLS / 255.  # scaled to [0, 1], only for plotting

	# Plot and show
	plot3d(img_small_HLS, img_small_hls)
	plt.show()

	plot3d(img_small_LUV, img_small_hls, axis_labels=list("LUV"))
	plt.show()


def plot3d(pixels, colors_rgb,
        axis_labels=list("HLS"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation

car_array = []
cars_path='vehicle_images/*.png'
car_images = glob.glob(cars_path)
for car_image in car_images:
	img = cv2.imread(car_image)
	print (car_image)
	plot_image(img)
	car_array.append(img)
