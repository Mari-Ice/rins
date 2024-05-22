import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from a6_utils import *
import os

def open(path, mode='L', dtype=np.float64, resize=False, n=120, m=80):
    image = Image.open(path).convert(mode)
    if(resize):
        image = image.resize((m, n))
    image = np.asarray(image)
    if(dtype == np.float64):
        image = image.astype(dtype) / 255
    return image
    
def plot_points(points, color, mark='o'):
    for i in range(points.shape[0]):
        plt.plot(points[i,0], points[i, 1], color=color, marker=mark)
        plt.text(points[i, 0] + 0.05, points[i, 1] + 0.05, str(i))

def PCA(points): # points is a matrix of [point1, point2, ...] like a vector! 
    # therefore transpose the matrix of points to get exactly the X
    X = points.T
    mean = np.sum(X, 1) / points.shape[0]
    mean_matrix = np.ones(X.shape)
    mean_matrix[0] = mean_matrix[0] * mean[0]
    mean_matrix[1] = mean_matrix[1] * mean[1]
    Xd = (X - mean_matrix)
    C = np.dot(Xd, Xd.T) / (points.shape[0]-1)
    U, D, V = np.linalg.svd(C)

    return U, D, V, C, mean, Xd
    


def excercise_1a():
    points = np.array([[3, 4], [3, 6], [7,6], [6, 4]])
    U, D, V, C, mean, Xd = PCA(points)
    plt.plot(0, 0, color='red', marker='o')
    plot_points(points, color='red', mark='x')
    plt.plot([mean[0], mean[0]+U[0, 0]], [mean[1], mean[1]+U[0, 1]], color='red')
    plt.plot([mean[0], mean[0]+U[1, 0]], [mean[1], mean[1]+U[1, 1]], color='green')
    plt.show()

def excercise_1b():
    points = np.loadtxt('./data/points.txt')
    U, D, V, C, mean, Xd = PCA(points)
    plot_points(points, color='red', mark='x')
    plt.plot(0, 0, color='red', marker='o')
    drawEllipse(mean, C)
    plt.show()

def excercise_1cd():
    points = np.loadtxt('./data/points.txt')
    U, D, V, C, mean, Xd = PCA(points)
    plot_points(points, color='red', mark='x')
    plt.plot(0, 0, color='red', marker='o')
    drawEllipse(mean, C)
    v1 = U[:, 0] * np.sqrt(D[0])
    v2 = U[:, 1] * np.sqrt(D[1])
    plt.plot([mean[0], mean[0]+v1[0]], [mean[1], mean[1]+v1[1]], color='red')
    plt.plot([mean[0], mean[0]+v2[0]], [mean[1], mean[1]+v2[1]], color='green')
    plt.show()
    print(D)
    comulative = np.cumsum(D)
    comulative /= np.max(comulative)
    plt.bar(range(comulative.shape[0]), comulative)
    plt.text(-0.3, 1, f'Percentage of lost variance without second eigenvector:\n {D[1]/sum(D)}')
    plt.show()
    
def excercise_1ef():
    points = np.loadtxt('./data/points.txt')
    U, D, V, C, mean, Xd = PCA(points)
    transformed_points = np.dot(U.T, Xd)
    transformed_points[1] = 0
    dim_U = U.copy()
    dim_U[:, 1] = 0
    back_points = np.dot(dim_U, transformed_points)
    means = np.ones(Xd.shape)
    means[0] *= mean[0]
    means[1] *= mean[1]
    back_points += means

    plot_points(back_points.T, color='green', mark='o')
    plot_points(points, color='red', mark='x')
    drawEllipse(mean, C)
    plt.show()

    point6 = np.array([6, 6])
    eucledean = (points - 6)**2
    eucledean = np.sum(eucledean, 1)
    closest_indx = np.argmin(eucledean)

    plot_points(points, color='red', mark='x')
    plt.plot(points[closest_indx, 0], points[closest_indx, 1], color='blue', marker='x', markersize=10)
    plt.plot(6, 6, color='black', marker='x', markersize=10)
    plot_points(back_points.T, color='green', mark='o')
    
    y6 = np.dot(U.T, point6 - mean)
    y6[1] = 0
    back_y6 = np.dot(dim_U, y6) + mean
    plt.plot(back_y6[0], back_y6[1], color='black', marker='o', markersize=8)
    eucl = back_points.T.copy()
    print(eucl)
    eucl[:, 0] = (eucl[:, 0] - back_y6[0])**2
    eucl[:, 1] = (eucl[:, 1] - back_y6[1])**2
    eucl = np.sum(eucl, 1)
    closest = np.argmin(eucl)
    plt.plot(back_points.T[closest, 0], back_points.T[closest, 1], color='blue', marker='o')
    plt.show()


def dual_PCA(points):
    X = points.T
    mean = np.sum(X, 1) / points.shape[0]
    Xd = (X - mean[:, np.newaxis])

    C = np.dot(Xd.T, Xd) / (points.shape[0]-1)
    U, D, V = np.linalg.svd(C)
    D += 10**-15
    
    a = np.diag(1 / np.sqrt(D * (points.shape[0] - 1)))
    
    U1 = np.dot(Xd, np.dot(U, a))
    return U1, D, V, C, mean, Xd

def excercise_2ab():
    points = np.loadtxt('./data/points.txt')
    U, D, V, C, mean, Xd = dual_PCA(points)
    U1, _, _, _, _, _ = PCA(points)
    transformed_points = np.dot(U.T, Xd)
    
    back_points = np.dot(U, transformed_points)
    means = np.ones(Xd.shape)
    means[0] *= mean[0]
    means[1] *= mean[1]
    back_points += means
    print(U)
    print(U1)
    plot_points(back_points.T, color='green', mark='o')
    plot_points(points, color='red', mark='x')
    
    v1 = U[:, 0] * np.sqrt(D[0])
    v2 = U[:, 1] * np.sqrt(D[1])
    plt.plot([mean[0], mean[0]+v1[0]], [mean[1], mean[1]+v1[1]], color='red')
    plt.plot([mean[0], mean[0]+v2[0]], [mean[1], mean[1]+v2[1]], color='green')
    plt.show()

def prepare_data(dir, resize=False, n=120, m=80):
    files = os.listdir(dir)
    matrix = []
    a = 0
    b = 0
    for file in files:
        im = open(os.path.join(dir, file), resize=resize, n=n, m=m)
        if(a == 0):
            a, b = im.shape
            print(im.shape)
        im = np.reshape(im, -1)
        matrix.append(im)

    return np.array(matrix), a, b

def show_eig(U, num, n, m):
    for i in range(num):
        v = U[:, i]
        plt.subplot(1, num, i+1)
        plt.imshow(np.reshape(v, (n,m)))
    plt.show()
def eigenvectors(matrix):
    U, S, V, C, mean, Xd = dual_PCA(matrix)
    return U, mean

def excercise_3ab():
    imgs1, n1, m1 = prepare_data('./data/faces/1')
    imgs2, n2, m2 = prepare_data('./data/faces/2')
    imgs3, n3, m3 = prepare_data('./data/faces/3')
    
    U1, mean1 = eigenvectors(imgs1)
   
    U2, mean2 = eigenvectors(imgs2)
    U3, mean3 = eigenvectors(imgs3)
    show_eig(U1, 5, n1, m1)
    show_eig(U2, 5, n2, m2)
    show_eig(U3, 5, n3, m3)

    image = np.dot(U1.T, imgs1[0, :].copy() - mean1)
    image = np.dot(U1, image) + mean1
    plt.subplot(1, 3, 1)
    plt.imshow(imgs1[0, :].reshape((n1, m1)))
    plt.subplot(1, 3, 2)
    plt.imshow(image.reshape(n1, m1))
    plt.subplot(1, 3, 3)
    plt.imshow(imgs1[0, :].reshape((n1, m1)) - image.reshape(n1, m1), vmin=0, vmax=1)
    plt.show()

    image2 = imgs1[1, :].copy()
    image2[1223] = 0
    image = np.dot(U1.T, image2 - mean1)
    image = np.dot(U1, image) + mean1
    plt.subplot(1, 3, 1)
    plt.imshow(image2.reshape((n1, m1)))
    plt.subplot(1, 3, 2)
    plt.imshow(image.reshape(n1, m1))
    plt.subplot(1, 3, 3)
    plt.imshow(image2.reshape((n1, m1)) - image.reshape(n1, m1))
    plt.show()

    image2 = imgs1[2, :].copy()
    image = np.dot(U1.T, image2 - mean1)
    image[3] = 0
    image = np.dot(U1, image) + mean1
    plt.subplot(1, 3, 1)
    plt.imshow(image2.reshape((n1, m1)))
    plt.subplot(1, 3, 2)
    plt.imshow(image.reshape(n1, m1))
    plt.subplot(1, 3, 3)
    plt.imshow(image2.reshape((n1, m1)) - image.reshape(n1, m1))
    plt.show()

def excercise_3c():
    imgs1, n1, m1 = prepare_data('./data/faces/1')
    U1, mean1 = eigenvectors(imgs1)

    orig_img = imgs1[25, :].copy()
    
    image = np.dot(U1.T, orig_img - mean1)
    image[32:] = 0
    image = np.dot(U1, image) + mean1
    plt.subplot(1, 6, 1)
    plt.imshow(image.reshape((n1, m1)))
    plt.title('32')
    image = np.dot(U1.T, orig_img - mean1)
    image[16:] = 0
    image = np.dot(U1, image) + mean1
    plt.subplot(1, 6, 2)
    plt.imshow(image.reshape((n1, m1)))
    plt.title('16')
    image = np.dot(U1.T, orig_img - mean1)
    image[8:] = 0
    image = np.dot(U1, image) + mean1
    plt.subplot(1, 6, 3)
    plt.imshow(image.reshape((n1, m1)))
    plt.title('8')
    image = np.dot(U1.T, orig_img - mean1)
    image[4:] = 0
    image = np.dot(U1, image) + mean1
    plt.subplot(1, 6, 4)
    plt.imshow(image.reshape((n1, m1)))
    plt.title('4')
    image = np.dot(U1.T, orig_img - mean1)
    image[2:] = 0
    image = np.dot(U1, image) + mean1
    plt.subplot(1, 6, 5)
    plt.imshow(image.reshape((n1, m1)))
    plt.title('2')
    image = np.dot(U1.T, orig_img - mean1)
    image[1:] = 0
    image = np.dot(U1, image) + mean1
    plt.subplot(1, 6, 6)
    plt.imshow(image.reshape((n1, m1)))
    plt.title('1')
    
    plt.show()

def excercise_3d():
    imgs1, n1, m1 = prepare_data('./data/faces/2')
    U1, mean1 = eigenvectors(imgs1)
    avg = np.sum(imgs1, axis=0) / imgs1.shape[0]
    image = np.dot(U1.T, avg - mean1)

    # changing eigenvector 4
    ch_eig = 4
    num_frames = 100
    x_values = np.linspace(0, 2 * np.pi, num_frames)
    x = 10
    val = image[ch_eig]

    for i in x_values:
        image[ch_eig] = val + np.sin(i) * x
        image1 = np.dot(U1, image) + mean1
        image1 = image1.reshape((n1, m1))
        plt.clf()
        plt.imshow(image1, cmap='gray')
        plt.draw()
        plt.pause(0.1)
        
    plt.show()


    ch_eig2 = 3
    val2 = image[ch_eig2]
    image[ch_eig2] = val
    for i in x_values:
        image[ch_eig] = val + np.sin(i) * x
        image[ch_eig2] = val2 + np.cos(i)*x
        image1 = np.dot(U1, image) + mean1
        image1 = image1.reshape((n1, m1))
        plt.clf()
        plt.imshow(image1, cmap='gray')
        plt.draw()
        plt.pause(0.1)
        
    plt.show()

def excercise_3e():
    imgs1, n1, m1 = prepare_data('./data/faces/1')
    U1, mean1 = eigenvectors(imgs1)

    image = open('./data/elephant.jpg')
    n, m = image.shape
    image = image.reshape(-1)
    pca = np.dot(U1.T, image - mean1)
    back_image = np.dot(U1, pca) + mean1
    back_image = back_image.reshape((n, m))
    plt.subplot(1, 2, 1)
    plt.imshow(image.reshape((n, m)))
    plt.subplot(1, 2, 2)
    plt.imshow(back_image)
    plt.show()

def excercise_3f():
    thresh = 5200
    imgs, n, m = prepare_data('./data/myface', resize=True, n=510, m=340)
    print(n, m)
    U, mean = eigenvectors(imgs)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        # Draw rectangles around the detected faces and extract the regions
        myface = False
        for (x, y, w, h) in faces:
            d = 100
            if(y-(d//2) < 0 or y+h+(d//2) >= frame.shape[0] or x-d < 0 or x+w+d >= frame.shape[1]):
                continue
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            color = (0, 0, 255)
            face_region = cv2.resize(gray[y-(d//2):y+h+(d//2), x-d:x+w+d], (n, n)).astype(np.float64) / 255
            face_region = face_region[:, int((n-m)/2):int(n-(n-m)/2)]
            
            # pca transform
            pca = np.dot(U.T, face_region.reshape(-1) - mean)
            back = np.dot(U, pca) + mean
            error = np.sum((face_region.reshape(-1) - back)**2)
            if(error < thresh):
                myface = True
            if(myface):
                color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Display the frame with rectangles around faces
        cv2.imshow('Face Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

    


    
    


    

#########################################################################
################################## MAIN #################################
#########################################################################
    
#excercise_1a()
#excercise_1b()
#excercise_1cd()
#excercise_1ef()
#excercise_2ab()    
#excercise_3ab()
#excercise_3c()
#excercise_3d()
#excercise_3e()
#excercise_3f()