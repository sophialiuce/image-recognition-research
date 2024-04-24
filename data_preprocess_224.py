import numpy as np
import os
from PIL import Image
import cv2
import pandas as pd

size = 2184
IMG_SIZE = 224

x_matrix = np.empty((size, IMG_SIZE, IMG_SIZE, 3))
y_brand = np.empty(size)
y_product = np.empty(size)
y_productBrand = np.empty(size)
cnt = 0
data_folder = "./data/"

dictionary = dict()
df = pd.DataFrame()
        
def resizeImg():
    dirpath = data_folder + "images"
    output_dir = data_folder + "images_resized224"
    for filename in os.listdir(dirpath):
        print("start resizing: " + filename)
        if filename != '.DS_Store':
            image = Image.open(dirpath + '/' +filename) # 读图片
            image = image.resize((IMG_SIZE,IMG_SIZE),Image.BILINEAR)
            img_nparray = np.asarray(image,dtype='float32')
            
            img_nparray = cv2.cvtColor(img_nparray, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_dir+'/'+filename, img_nparray)
              
def vectorizeImg():
    dirpath = data_folder + "images_resized224"
    global  cnt,x_train,y_label
    for filename in os.listdir(dirpath):
        print("start vectorizing: " + filename)
        if filename != '.DS_Store':
            image = cv2.imread(dirpath + '/' +filename)
            x_matrix[cnt] = image
            y_brand[cnt] = float(filename[0])
            y_product[cnt] = float(filename[2])
            y_productBrand[cnt] = float(filename[0])*10 + float(filename[2])
        cnt += 1
        
def saveNpy():
    np.save("./train_data_x_224.npy",x_matrix)
    print("saved x_matrix")
    np.save("./train_data_y_brand_224.npy",y_brand)
    np.save("./train_data_y_product_224.npy",y_product)
    np.save("./train_data_y_productBrand_224.npy",y_productBrand)
        
if __name__ == '__main__':
    resizeImg()
    vectorizeImg()
    saveNpy()