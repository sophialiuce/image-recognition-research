#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 07:40:26 2018

@author: sophialiu
"""

import pandas as pd
import numpy as np

dictionary = dict()

def save_structure():
    df = pd.read_csv("./InterviewExam/Label.csv")
    for i in range(0, len(df)):
        f = df.iloc[i]['file']
        index = int(f[0]) * 10000 + int(f[2]) * 1000 + int(f[4:7])
        info = (df.iloc[i]['product_name'], df.iloc[i]['brand_name'], df.iloc[i]['Price'])
        dictionary.update({index:info})
    
def interpolation_search(lis, f):
    key = int(f[0]) * 10000 + int(f[2]) * 1000 + int(f[4:7])
    print(key)
    low = 0
    high = len(lis) - 1
    time = 0
    while low < high:
        time += 1
        # The core of the interpolation search is the "mid"
        mid = low + int((high - low) * (key - lis[low])/(lis[high] - lis[low]))
        print("mid=%s, low=%s, high=%s" % (mid, low, high))
        if key < lis[mid]:
            high = mid - 1
        elif key > lis[mid]:
            low = mid + 1
        else:
            # 打印查找的次数
            print("Search times: %s" % time)
            return mid
        
    print("Search times: %s" % time)
    return False
 
if __name__ == '__main__':
    save_structure()
    filename = "6_9_009.png"
    li = np.array(list(dictionary.keys()))
    result_index = interpolation_search(li, filename)
    result = dictionary[li[result_index]]
    print("For file ", filename, " the result is: ")
    print(result)
