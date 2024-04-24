#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:42:57 2018

@author: sophialiu
"""

import numpy as np
import os
from PIL import Image
import random
import cv2
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import pandas as pd
import pickle

class DataPreprocess:
    
    def __init__(self, size = 2184, cnt = 0, IMG_SIZE = 128):
        self.size = size
        self.IMG_SIZE = IMG_SIZE
        self.x_matrix = np.empty((size, 224, 224, 3))
        self.y_brand = np.empty(size)
        self.y_productBrand = np.empty(size)
        self.cnt = cnt
        self.dictionary = dict()
        self.df = pd.DataFrame()
        self.datagen = ImageDataGenerator(
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')
        self.data_folder = './data/'
        
    def relight(self, img, light=1, bias=0):
        w = img.shape[1]
        h = img.shape[0]
        #image = []
        for i in range(0,w):
            for j in range(0,h):
                for c in range(3):
                    tmp = int(img[j,i,c]*light + bias)
                    if tmp > 255:
                        tmp = 255
                    elif tmp < 0:
                        tmp = 0
                    img[j,i,c] = tmp
        return img
    
    def relightImg(self):
        dirpath = self.data_folder + "images"
        output_dir = self.data_folder + "input_images"
        for filename in os.listdir(dirpath):
            print('Start Relight: '+dirpath + '/' +filename)
            image = cv2.imread(dirpath + '/' +filename)
            cv2.imwrite(output_dir+'/'+filename[0:7]+'.jpg', image) #save the original image
            
            image = self.relight(image, random.uniform(0.5, 1.5), 
                        random.randint(-50, 50)) # randomly change the light and contrast
            cv2.imwrite(output_dir+'/'+filename[0:7]+'_relight.jpg', image)
    
    def augmentationImg(self): # augmentation with keras
        dirpath = self.data_folder + "images"
        output_dir = self.data_folder + "input_images"
        for filename in os.listdir(dirpath):
            print('Start Augmentation: '+ filename)
            img = load_img(dirpath + '/' +filename)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0
            for batch in self.datagen.flow(x,
                                      batch_size=1,
                                      save_to_dir=output_dir,
                                      save_prefix= filename[0:7],
                                      save_format='jpg'):
                i += 1
                if i > 2: # the number here controls the number of images generated
                    break  # otherwise the generator would loop indefinitely
                    
    def grayscaleImg(self):
        dirpath = self.data_folder + "input_images"
        output_dir = self.data_folder + "input_images_gray"
        for filename in os.listdir(dirpath):
            print("start grayscale: " + filename)
            if filename != '.DS_Store':
                image = Image.open(dirpath + '/' +filename)
                image = image.resize((224,224),Image.BILINEAR)
                image = image.convert("L") # 灰度
                img_nparray = np.asarray(image,dtype='float32')
                cv2.imwrite(output_dir+'/'+filename, img_nparray)
            
    def resizeImg(self):
        dirpath = self.data_folder + "images"
        output_dir = self.data_folder + "images_resized"
        for filename in os.listdir(dirpath):
            print("start resizing: " + filename)
            if filename != '.DS_Store':
                image = Image.open(dirpath + '/' +filename)
                image = image.resize((self.IMG_SIZE,self.IMG_SIZE),Image.BILINEAR)
                img_nparray = np.asarray(image,dtype='float32')
            
                img_nparray = cv2.cvtColor(img_nparray, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_dir+'/'+filename, img_nparray)
                
    def vectorizeImg(self):
        dirpath = self.data_folder + "Images_resized"
        global  cnt,x_train,y_label
        for filename in os.listdir(dirpath):
            print("start vectorizing: " + filename)
            if filename != '.DS_Store':
                image = cv2.imread(dirpath + '/' +filename)
                self.x_matrix[cnt] = image
                self.y_brand[cnt] = float(filename[0])
                self.y_product[cnt] = float(filename[2])
                self.y_productBrand[cnt] = float(filename[0])*10 + float(filename[2])
            cnt += 1
            
    # def readCSV(self):
    #     path = self.data_folder + "Label.csv"
    #     df = pd.read_csv(path)
    #     return df

    # def readTXT(self):
    #     path = self.data_folder + "Product-Brand.txt"
    #     file_object = open(path,'r')
    #     try: 
    #         for line in file_object:
    #             if line == '\n':
    #                 continue;
    #             else:
    #                 l = line.split(',')
    #                 self.dictionary.update({l[0]:int(l[1])})
                
    #     finally:
    #         file_object.close()
    #     return self.dictionary
    
    # def storeX(self):
    #     #pickle_dump(x_matrix, self.data_folder + "x_matrix_4.pkl")
    #     file_path = self.data_folder + "x_matrix_4.pkl"
    #     n_bytes = 2**31
    #     max_bytes = 2**31 - 1
    #     data = bytearray(self.x_matrix)
    
    #     bytes_out = pickle.dumps(data)
    #     with open(file_path, 'wb') as f_out:
    #         for idx in range(0, n_bytes, max_bytes):
    #             f_out.write(bytes_out[idx:idx+max_bytes])

    # def storeDictionary(self):
    #     with open(self.data_folder + 'dictionary.pkl', 'wb') as f:
    #         pickle.dump(self.dictionary,f)
        
    # def storeLabels(self):
    #     with open(self.data_folder + 'labels.pkl', 'wb') as f:
    #         pickle.dump(self.y_brand,f)
    #         pickle.dump(self.y_product,f)
    #         pickle.dump(self.y_productBrand,f)
        
    # def saveNpy(self):
    #     np.save("./train_data_x.npy",self.x_matrix)
    #     print("saved x_matrix")
    #     np.save("./train_data_y_brand.npy",self.y_brand)
    #     np.save("./train_data_y_product.npy",self.y_product)
    #     np.save("./train_data_y_productBrand.npy",self.y_productBrand)
        
if __name__ == '__main__':
    d = DataPreprocess()
    d.augmentationImg()
    d.relightImg()
    d.grayscaleImg()
    d.resizeImg()
    d.vectorizeImg()
    #df = d.readCSV()
    #txt = d.readTXT()
    #d.storeLabels()
    #d.storeDictionary()
    #d.storeX()
    #d.saveNpy()