#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:07:17 2018

@author: sophialiu
"""

import numpy as np

brand = np.load('train_data_y_brand_224.npy')
product = np.load('train_data_y_product_224.npy')
productBrand = np.load('train_data_y_productBrand_224.npy')

brand = brand.astype(int)
product = product.astype(int)
productBrand = productBrand.astype(int)

n_values_b = np.max(brand) + 1
n_values_p = np.max(product) + 1
n_values_pb = np.max(productBrand) + 1

y_brand_one_hot = np.eye(n_values_b)[brand]
y_product_one_hot = np.eye(n_values_p)[product]
y_productBrand_one_hot = np.eye(n_values_pb)[productBrand]

np.save("./train_data_y_brand_one_hot_224.npy",y_brand_one_hot)
np.save("./train_data_y_product_one_hot_224.npy",y_product_one_hot)
np.save("./train_data_y_productBrand_one_hot_224.npy",y_productBrand_one_hot)