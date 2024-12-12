#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :create_data_lists.py
@说明        :创建训练集和测试集列表文件
@时间        :2020/02/11 11:32:59
@作者        :钱彬
@版本        :1.0
'''


from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['D:/Cfbl/RE-SRGAN/data/PCB_picture/images/train',],
                      test_folders=['D:/Cfbl/RE-SRGAN/data/PCB_picture/images/test'],
                      min_size=100,
                      output_folder='D:/Cfbl/RE-SRGAN/data')
