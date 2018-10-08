import os
import sys
import math
import tensorflow as tf
train_data_path='.\\data\\train'
test_data_path='.\\data\\test'
#read data and return a tuple list ,the tuple is (file_path,label)
def read_train_data(data_path):
    list=[(os.path.join(file_path,file),0 if file.split('.')=='dog' else 1) for file in os.listdir(data_path)]
    return list
#create TFrecord data by train_data list


