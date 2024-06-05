#!/usr/bin/env python3

import statistics
import random
import time
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import Image, PointCloud2, LaserScan
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from geometry_msgs.msg import Point
import message_filters

from ultralytics import YOLO
import sys
import os

from sklearn.decomposition import PCA
import pickle as pk

def erode_img(img):
	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	# img = cv2.erode(img, kernel)
	# img = cv2.erode(img, kernel)
	#img = cv2.convertScaleAbs(img, alpha=1.3, beta=40.0)
	#img = cv2.Canny(img, 10,20)

	# low_sigma = cv2.GaussianBlur(img,(1,1),0)
	# high_sigma = cv2.GaussianBlur(img,(3,3),0)
	# img = low_sigma - high_sigma

	# img = cv2.erode(img, kernel)
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img = cv2.Laplacian(img,-1,1,5)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img,(3,3),0)
	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	# img = cv2.dilate(img, kernel)

	return img

def open_img(path, dtype=np.float64, resize=False, n=120, m=80):
	image = cv2.imread(path)
	image = erode_img(image) #Tole pomaga, da lazje najdemo male napake #TODO

	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#print(image.shape)
	if(resize):
		image = cv2.resize(image, (m, n))
	image = np.asarray(image)
	if(dtype == np.float64):
		image = image.astype(dtype) / 255
	return image

def prepare_data(dir, resize=False, n=120, m=80):
	files = os.listdir(dir)
	matrix = []
   
	for file in files:
		im = open_img(os.path.join(dir, file), resize=resize, n=n, m=m)
		#print(im.shape)
		#im = np.reshape(im, -1)
		im = np.reshape(im, -1)
		#print(im.shape)
		matrix.append(im)

	return np.array(matrix)

def fit_pca(matrix, n_components=10):
	pca = PCA(n_components=n_components)
	pca.fit(matrix)
	return pca

def get_error(img, pca):
	original_shape = img.shape
	img = np.reshape(img, -1)
	img2 = pca.inverse_transform(pca.transform([img.copy()]))
	err = abs(img - img2)
	return np.reshape(err, original_shape)

class face_extractor(Node):
	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None

		self.rgb_image_sub = message_filters.Subscriber(self, Image, "/oakd/rgb/preview/image_raw")
		self.ts = message_filters.ApproximateTimeSynchronizer( [self.rgb_image_sub], 10, 0.05, allow_headerless=False) 
		self.ts.registerCallback(self.rgb_callback)

		self.model = YOLO("yolov8n.pt")
		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Mona2", cv2.WINDOW_NORMAL)
		cv2.namedWindow("ErrorMona", cv2.WINDOW_NORMAL)

		self.face_count = 0
		self.prev_img_stats = [0,0,0,0]

		pwd = os.getcwd()
		self.gpath = pwd[0:len(pwd.lower().split("rins")[0])+4]

		self.m = 300
		self.n = int(1.5*self.m)
		
		train_folder = f"{self.gpath}/mona_images"
		self.pca = None
		#matrix = prepare_data(train_folder, resize=True, n=self.n, m=self.m)
		#self.pca = fit_pca(matrix, n_components=220)
		#pk.dump(self.pca, open("pca.pkl","wb"))
		self.pca = pk.load(open("pca.pkl",'rb')) 

		self.min_img = None
		self.min_img_inited = False
		print("OK")

	def rgb_callback(self, rgb_data):
		key = cv2.waitKey(1)
		if key==27: #Esc
			print("exiting")
			exit()

		cv_image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
		# cv2.imshow("Image", cv_image)
		# return

		hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
		edge_img = np.zeros((cv_image.shape[1], cv_image.shape[0], 1), dtype=np.uint8)
		edge_img[hsv_image[:,:,1] < 200] = (255)
		edge_img[(hsv_image[:,:,0] > 30) & (hsv_image[:,:,0] < 220)] = (255)
		cv2.floodFill(edge_img, None, (int(50),int(555)), 0)
		
		# gray = 255 - edge_img	
		gray = edge_img	
		
		#dilate
		
		# dil_mat = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
		# gray = cv2.erode(gray, dil_mat)
		#cv2.imshow("GrayDilated", gray)
		# cv2.imshow("Gray", gray)
		
		contours_tupled, hierarchy = cv2.findContours(image=gray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
		contours = []
		for c in contours_tupled:
			_,_,w,h = cv2.boundingRect(c)	
			if(w < 240 or h < 350):
				continue

			touches_left = any(0 in x for x in c[:,:,0])
			touches_right = any(gray.shape[1]-1 in x for x in c[:,:,0])
			touches_top = any(0 in y for y in c[:,:,1])
			if(touches_left or touches_right or touches_top):
				continue

			contours.append(cv2.convexHull(c))

		mask = np.zeros_like(gray)
		for i in range(len(contours)):
			cv2.drawContours(mask, contours, i, 255, cv2.FILLED)

		#cv2.imshow("Mask", mask)
		#print(f"mask: {mask.shape}, img: {cv_image.shape}")
		
		# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
		# mask = cv2.erode(mask, kernel)

		# #redo contours
		# contours_tupled, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
		# contours = []
		# for c in contours_tupled:
		# 	contours.append(cv2.convexHull(c))

		# #cv2.imshow("Mask1", mask)
		# #return

		masked = cv_image.copy()
		masked[mask[:,:,0] == 0] = (255,255,255)
		# masked[mask == 0] = (255,255,255)
		# cv2.imshow("Masked", masked)
		# return

		res = self.model.predict(masked, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)
		for x in res:
			bbox = x.boxes.xyxy
			if bbox.nelement() == 0: #No element
				continue
			bbox = bbox[0]
			#if(abs(bbox[2] - bbox[0]) < 50 or abs(bbox[3] - bbox[1]) < 80): #Too small
			if(abs(bbox[2] - bbox[0]) < 135 or abs(bbox[3] - bbox[1]) < 200): #Too small
				continue
		 
			if(abs(0.66 - (abs(bbox[2] - bbox[0]) / abs(bbox[3] - bbox[1]))) > 0.1): #Not proper proportions
				continue

			img_stats = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
			pis = self.prev_img_stats
			diff = abs(img_stats[0] - pis[0]) + abs(img_stats[1] - pis[1]) + abs(img_stats[2] - pis[2]) + abs(img_stats[3] - pis[3])
			self.prev_img_stats = img_stats
			print(self.face_count)

			#cv_image = cv2.rectangle(cv_image, (img_stats[0], img_stats[1]), (img_stats[2], img_stats[3]), (0,0,255), 3)
			
			img_surface = abs(img_stats[2] - img_stats[0]) * abs(img_stats[3] - img_stats[1])
	
			min_err = float("inf")
			min_index = 0
			cbox = img_surface
			
			for i,c in enumerate(contours):
				c_box = cv2.boundingRect(c)	
			
				c_box = [c_box[0], c_box[1], c_box[0] + c_box[2], c_box[1] + c_box[3]]
				c_surface = abs(c_box[2] - c_box[0]) * abs(c_box[3] - c_box[1])
				error = 1.0 - (img_surface / c_surface)
				
				if(error < min_err and error > 0):
					min_err = error
					min_index = i
					cbox = [int(c_box[0]), int(c_box[1]), int(c_box[2]), int(c_box[3])]
				# cv_image = cv2.rectangle(cv_image, (c_box[0], c_box[1]), (c_box[2], c_box[3]), (0,255,0), 2)
				# for p in c:
				# 	print(p)
				# 	cv2.circle(cv_image,p[0], 3, (0,0,255), -1)

			# print(min_err)
			if(min_err < 0.6 and min_err >= 0):
				cv_image = cv2.rectangle(cv_image, (cbox[0], cbox[1]), (cbox[2], cbox[3]), (255,0,0), 2)
				mona_img = masked[int(cbox[1]):int(cbox[3]), int(cbox[0]+3):int(cbox[2]-3)]
				mona_mask = mask[int(cbox[1]):int(cbox[3]), int(cbox[0]+3):int(cbox[2]-3)]

				# cv2.imshow("MonaMask", mona_mask)
				# return

				if(diff > 1):
					self.face_count += 1
					#cv2.imwrite(f"{self.gpath}/mona_images/mona_{self.face_count:04}.jpg", mona_img)
					#cv2.imshow("Mona", mona_img)

					#Popravimo perspektivo.
					#print(mona_img.shape)
				
					first_red_left = 0
					first_red_right = 0

					last_red_left  = mona_mask.shape[0]
					last_red_right = mona_mask.shape[0]
				
					fnd = 0
					for j in range(mona_mask.shape[0]):
						col_left = mona_mask[j, 0]
						col_right = mona_mask[j, mona_mask.shape[1]-1]
						if((fnd & 1) == 0 and col_left != 0):
							fnd = fnd | 1
							first_red_left = j
					
						if((fnd & 2) == 0 and col_right != 0):
							fnd = fnd | 2
							first_red_right = j

						if(fnd == 3):
							break

					fnd = 0
					for j in range(mona_mask.shape[0]-1, 0, -1):
						col_left = mona_mask[j, 0]
						col_right = mona_mask[j, mona_mask.shape[1]-1]
						if((fnd & 1) == 0 and col_left != 0):
							fnd = fnd | 1
							last_red_left = j
					
						if((fnd & 2) == 0 and col_right != 0):
							fnd = fnd | 2
							last_red_right = j

						if(fnd == 3):
							break

					p0 = [0, first_red_left]
					p1 = [0, last_red_left]
					p2 = [mona_img.shape[1], last_red_right]
					p3 = [mona_img.shape[1], first_red_right]
					p = np.float32([p0, p1, p2, p3])
					
					t0 = [0, 0]
					t1 = [0, self.n]
					t2 = [self.m, self.n]
					t3 = [self.m, 0]
					t = np.float32([t0, t1, t2, t3])
				
					pm = cv2.getPerspectiveTransform(p, t)
					mona2 = cv2.warpPerspective(mona_img,pm,[self.m,self.n])
				
					if(self.pca == None):
						cv2.imwrite(f"{self.gpath}/mona_images/mona_{self.face_count:04}.jpg", mona2)

					mona2 = erode_img(mona2)
					cv2.imshow("Mona2", mona2)

					if(self.pca != None):
						mona2 = mona2.astype(np.float64) / 255
						err = get_error(mona2, self.pca)
						err *= 255
						err = err.astype(np.uint8)

						if(not self.min_img_inited):
							self.min_img = err
							self.min_img_inited = True
						else:
							#self.min_img = np.minimum(self.min_img, err)
							# self.min_img = self.min_img * 0.5 + 0.5 * err
							#self.min_img = cv2.fastNlMeansDenoising(err)
							self.min_img = err
							# self.min_img = err

				
						ret, err = cv2.threshold(err,60,255,cv2.THRESH_TOZERO)

						##Robove pobarvamo na crno
						err[0:3,:] = 0
						err[err.shape[0]-4:,:] = 0
						err[:,0:3] = 0
						err[:,err.shape[1]-4:] = 0
						
						# img = err
						# kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
						# img = cv2.dilate(img, kernel2)
						# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
						# img = cv2.erode(img, kernel)
						# err = img

						# final_img = np.zeros_like(err)
						# contours_tupled, hierarchy = cv2.findContours(image=err, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
						# for i,c in enumerate(contours_tupled):
						# 	_,_,w,h = cv2.boundingRect(c)	
						# 	if(w < 5 and h < 5):
						# 		continue
						# 	# c = cv2.convexHull(c)
						# 	cv2.drawContours(final_img, [c], 0, 255, cv2.FILLED)

						# cv2.imshow("ErrorMona", final_img) 
						# cv2.imshow("ErrorMona", self.min_img) 
						cv2.imshow("ErrorMona", err) 


		cv2.imshow("Image", cv_image)

def main():
	rclpy.init(args=None)
	node = face_extractor()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
