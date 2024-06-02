import numpy as np 
import cv2
import os
from sklearn.decomposition import PCA

def open(path, dtype=np.float64, resize=False, n=120, m=80):
	image = cv2.imread(path)
	image = erode_img(image)
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
		im = open(os.path.join(dir, file), resize=resize, n=n, m=m)
		#print(im.shape)
		im = np.reshape(im, -1)
		#print(im.shape)
		matrix.append(im)

	return np.array(matrix)

def fit_pca(matrix, n_components=10):
	pca = PCA(n_components=n_components)
	pca.fit(matrix)
	return pca

def get_error(img, pca):
	img = np.reshape(img, -1)
	img2 = pca.inverse_transform(pca.transform([img.copy()]))
	
	return abs(img - img2), img2

def erode_img(img):
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	img = cv2.erode(img, kernel)
	img = cv2.erode(img, kernel)
	return img

if __name__ == '__main__':
	#n = 200
	#m = 130
	m = 300
	n = int(m*1.5)

	pwd = os.getcwd()
	gpath = pwd[0:len(pwd.lower().split("rins")[0])+4]

	train_folder = f"{gpath}/mona_images"
	monas = f"{gpath}/src/dis_tutorial3/worlds/task3_meshes"
	matrix = prepare_data(train_folder, resize=True, n=n, m=m)

	#print(matrix.shape)
	#print(np.any(np.isnan(matrix)))
	pca = fit_pca(matrix, n_components=40)

	monas_fake = ['anomona_0.png', 'anomona_1.png', 'anomona_2.png', 'anomona_3.png', 'anomona_4.png', 
				  'easy_anomona_0.png', 'easy_anomona_1.png', 'easy_anomona_2.png', 'easy_anomona_3.png', 'easy_anomona_4.png', 'mona.png']
	monas_real = ['mona_0001.jpg', 'mona_0020.jpg', 'mona_0099.jpg', 'mona_0150.jpg', 'mona_0243.jpg', 'mona_0324.jpg','mona_0432.jpg', 'mona_0523.jpg', 'mona_0610.jpg', 'mona_0677.jpg', 'mona_0732.jpg', 'mona_0844.jpg', 'mona_0934.jpg']

	# monas_test = ["mona.png", "mona1.png", "mona2.png", "mona3.png", "mona4.png", "mona5.png" ]

	# for mona in monas_test:
	# 	img = open(mona, resize=True, n=n, m=m)
	# 	err, img2 = get_error(img, pca)

	# 	err_val = np.sum(err)
	# 	print(f"Error: {err_val}")
	# 	err = np.reshape(err, (n, m, 3))
	# 	cv2.namedWindow(mona, cv2.WINDOW_NORMAL)
	# 	cv2.imshow(mona, err)
	# 	
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()

	for mona in monas_fake:
		img = open(os.path.join(monas, mona), resize=True, n=n, m=m)
		err, img2 = get_error(img, pca)
		err = np.reshape(err, (n, m, 3))
		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		cv2.imshow('Image', err)
		
		cv2.waitKey(0)
	for mona in monas_real:
		img = open(os.path.join(train_folder, mona), resize=True, n=n, m=m)
		err, img2 = get_error(img, pca)

		err_val = np.sum(err)
		print(f"Error: {err_val}")

		err = np.reshape(err, (n, m, 3))
		cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		cv2.imshow('Image', err)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	cv2.destroyAllWindows()
