import numpy as np 
import cv2
import glob
import os

clip1 = cv2.VideoCapture('project_video.mp4')

car_array = []
cars_path='test_images/*.jpg'
car_images = glob.glob(cars_path)
for car_image in car_images:
	img = cv2.imread(car_image)
	car_array.append(img)
