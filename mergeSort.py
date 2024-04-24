#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 11:41:28 2018

@author: sophialiu
"""

import pandas as pd
import numpy as np

def getPrice():
    df = pd.read_csv("./InterviewExam/Label.csv")
    return np.array(df['Price'])

def merge(a, b):
    c = []
    h = j = 0
    while j < len(a) and h < len(b):
        if a[j] > b[h]:
            c.append(a[j])
            j += 1
        else:
            c.append(b[h])
            h += 1

    if j == len(a):
        for i in b[h:]:
            c.append(i)
    else:
        for i in a[j:]:
            c.append(i)

    return c


def merge_sort(lists):
    if len(lists) <= 1:
        return lists
    middle = len(lists)//2
    left = merge_sort(lists[:middle])
    right = merge_sort(lists[middle:])
    return merge(left, right)

def spendMoney(li, lower, upper):
    sum_ = 0
    count = 0
    while True:
        if sum_ >= lower and sum_ <= upper:
            break
        sum_ += li[count]
        count += 1
        
    return sum_, count

if __name__ == '__main__':
    a = getPrice()
    res = merge_sort(a)
    sum_, count = spendMoney(res, 2980, 3020)
    print("The sum is: ", sum_, ". The number of items is: ", count)