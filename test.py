# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:09:51 2020

@author: 10844
"""

import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from PIL import Image
from model import Vgg16, Res18, WRN


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


"""
data_root = os.path.abspath(os.path.join(os.getcwd(), "./tiger.jpg"))
img = Image.open(data_root)  #读取.jpg文件
print(img.size)
arr = np.array(img)
print (arr[8][6])
"""

"""
arr = np.array([[1,2,3],[1,2,3]])
print(arr.shape)
print(arr)
arr.resize(6)
print(arr.shape)
print(arr)
"""


"""
a = np.array([1], dtype=int)
print(sys.getsizeof(a))
a = np.array([1], dtype=float)
print(sys.getsizeof(a))
np.array([1], dtype=np.float32)
print(sys.getsizeof(a))
np.array([1], dtype=np.double)
print(sys.getsizeof(a))
print('\n')
a = np.array([1, 2], dtype=int)
print(sys.getsizeof(a))
a = np.array([1, 2], dtype=float)
print(sys.getsizeof(a))
np.array([1, 2], dtype=np.float32)
print(sys.getsizeof(a))
np.array([1], dtype=np.double)
print(sys.getsizeof(a))
"""

"""
arr = [''] * 100
for i in range(0, 100):
    arr[i] = str(i+1);
print(arr)
"""

"""
class Feature():
    def __init__(self, index=0, score=0.0):
        self.index = index
        self.score = score
feature_info = [Feature()] * 100
for i in range(0, 100):
    print(feature_info[99].index)
    print(feature_info[99].score)
    print('\n')
"""

"""
class Feature():
    def __init__(self, index=0, score=0.0):
        self.index = index
        self.score = score
        
feature_info = [Feature() for i in range(0, 5)]
feature_info[0].score = 4.0
feature_info[1].score = 1.0
feature_info[2].score = 5.0
feature_info[3].score = 3.0
feature_info[4].score = 2.0

for i in range(0, 5):
    feature_info[i].index = i
feature_info.sort(key=lambda x : x.score, reverse=True)
for i in range(0, 5):
    print(feature_info[i].index)
"""


"""
feature = {'index': 0,'score':0.0}
feature_info = [feature for i in range(0, 5)]
feature_info[0]['score'] = 4.0
feature_info[1]['score'] = 3.0
feature_info[2]['score'] = 5.0
feature_info[3]['score'] = 1.0
feature_info[4]['score'] = 2.0
for i in range(0, 5):
    print(feature_info[i]['score'])
"""

"""
a = np.array([[1, 3, 5, 7],
     [2, 4, 6, 8],
     [3, 6, 9, 12]
    ])
b = np.sum(a[:, :3], axis=1)
print(b)
temp = np.array(a)
print(temp)
a.T[0] = 0
print(a)
a = temp
print(a)
"""

"""
a = 1
b = 1.0
print(a == b)
"""

"""
def weight_variable(shape):
    initial=tf.compat.v1.truncated_normal(shape,stddev=0.1)
    return initial

if __name__=='__main__':

    W_conv1 = weight_variable([5, 5, 1, 32])
    a = W_conv1.numpy()
    print (a[0][0])
   # b_conv1 = bias_variable([32])
"""

"""
sess = tf.compat.v1.Session()
with sess.as_default():
    print(type(tf.compat.v1.constant([1, 2, 3]).eval()))
    
"""

"""
a = np.array([[1, 2, 3],
              [1, 2, 3]])
b = np.array([[4, 5, 6],
             [4, 5, 6]])
print(np.vstack((a, b)))
print(np.hstack((a, b)))
"""

"""
from scipy.stats import norm
 
q = norm.cdf(1.96)  #累计密度函数
print(q)
print(norm.ppf(q))  #累计密度函数的反函数
"""

"""
os.chdir('C:/Users/10844/Desktop/data/original_data')
all_files = os.listdir()
print(all_files)
"""

"""
i = 1
j = 1
print('(', i, ',', j, ')\n')
"""

"""
x = torch.empty(5,3)
print(x)
print(torch.sum(x, dim=1, keepdim=False))
print(torch.mean(x, dim=1, keepdim=True))
print(x.repeat(3,2))
y = torch.unsqueeze(x, dim=1).repeat(1, 5, 1)
print(y)
print(y.size())
"""

"""
x = np.empty((5,3))
y = np.empty((5,2))
print(x)
print(y)
training_data = np.hstack((x, y))
print(training_data)
np.random.shuffle(training_data)
x = training_data[:, :-2]
y = training_data[:, -2:]
print(x)
print(y)
"""

"""
x = np.ones((10,5))
print(x)
print(np.sum(x[:, :3], axis=1, keepdims=False))
print(np.sum(x[0, :3]))
"""

"""
x = np.ones((10, 5))
print(np.sum(x[1, :0]))
"""

"""
data = pd.read_csv('data_lgbm_test.csv')
print(np.array(data.index))
print(np.array(data['Survival time']))
"""

"""
data = pd.read_csv('data_mean5_delete_200_correlation_0.05.csv')
print(data.shape)
print(data.iloc[5:10,0:3])
result = np.zeros((1200, 10))
np.savetxt('level1_result.csv', result, delimiter=',')
"""

"""
data_all = pd.read_csv('data_mean5_delete_200_correlation_0.05.csv').drop(['Survival time'], axis=1)
data_level1 = pd.read_csv('level1_character.csv').drop(['Survival time'], axis=1)
data_level2 = pd.read_csv('level2_character.csv').drop(['Survival time'], axis=1)
data_level3 = pd.read_csv('level3_character.csv').drop(['Survival time'], axis=1)
data_level4 = pd.read_csv('level4_character.csv').drop(['Survival time'], axis=1)
data_all = pd.concat([data_all, data_level1], axis=1, join='inner')
data_all = pd.concat([data_all, data_level2], axis=1, join='inner')
data_all = pd.concat([data_all, data_level3], axis=1, join='inner')
data_all = pd.concat([data_all, data_level4], axis=1, join='inner')
data_all.to_csv('all_character.csv', index=False)
"""

"""
arrx = np.array([[1, 2, -3], [4, 5, 6], [7, 8, 9]])
arry = np.array([1, 1, 1])
arrz = arrx - arry
result = (arrx - arry).sum(axis=1).argmax()
print(arrz.shape)
print(arrz.sum(axis=1))
print(arrz)
print(result)
"""

"""
class Node():
    def __init__(self, index=0, value=0.0):
        self.index = index
        self.value = value

kernel= np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6]])
weight_importance = [Node(i+1, kernel[int(i / 3)][int(i % 3)]) for i in range(0, 9)]
weight_importance.sort(key=lambda f: f.value, reverse=True)
weight_importance = weight_importance[0:4]
weight_importance.sort(key=lambda f: f.index, reverse=True)
pattern_shape_id = 1000 * weight_importance[0].index + 100 * weight_importance[1].index + 10 * weight_importance[2].index + weight_importance[3].index
print(pattern_shape_id)
"""

"""
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count()) # 返回gpu数量
device = torch.device('cuda:0')
# net = Net().to(device)
# x = x.cuda()
# out = net(x).cpu()
# out = out.detach().numpy()
"""

"""
arr = torch.zeros(3, 3)
arr[0][2] = -3.2
print(arr[0][2].abs())
"""

"""
def test():
    a = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[2, 3, 4], [2, 3, 4], [2, 3, 4]], [[3, 4, 5], [3, 4, 5], [3, 4, 5]]])
    print(a)
    # a = a.data.cpu().numpy()
    np.save('s_VGG16_.npy', a)
    b = np.load('s_VGG16_.npy')
    print(b)
"""

"""
def test():
    a = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[2, 3, 4], [2, 3, 4], [2, 3, 4]], [[3, 4, 5], [3, 4, 5], [3, 4, 5]]])
    b = np.array([[[7, 8, 9], [7, 8, 9], [4, 5, 6]], [[1, 2, 3], [7, 8, 9], [1, 2, 3]], [[4, 5, 6], [1, 2, 3], [7, 8, 9]]])
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    c = b * a[0]
    print(c)
    min_number = (b * a[0]).abs().sum(axis=2).sum(axis=1).argmin().item()
    print(min_number)
    c = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    d = np.array([[7, 8, 9], [7, 8, 9], [4, 5, 6]])
    c = torch.from_numpy(c)
    d = torch.from_numpy(d)
    e = torch.pow(c-d, 2)
    print(e)
"""

"""
def test():
    a = [100, 120, 140, 160, 180, 190, 195, 200]
    for i in range(0, 200):
        if i + 1 in a:
            print(i)
    print(len(a))
"""


"""
def test():
    a = [100, 120, 140, 160, 180, 190, 195, 200]
    print(a)
    a = np.array(a)
    print(a)
    print(a.shape)
    a = a.tolist()
    print(a)

    b = torch.zeros(10)
    print(b)
    b = b.numpy()
    print(b.shape)
    b = b.tolist()
    print(b)
"""

"""
def test():
    a = 'shape&normalized'
    b = 'shape&original'
    c = 'shape'
    d = 'normalized'
    e = 'normalized'
    print(c in a)
    print(d in a)
    print(d in b)
    print(e in d)
"""


"""
def test():
    for i in range(10, 0, -1):
        if i == 5:
            continue
        print(i)
    print(0.57 < 0.2)
"""


"""
def test():
    w = torch.empty(3, 5)

    nn.init.uniform_(w)
    print(w)
    print(w.std().item())
    nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    print(w)
    print(w.std().item())
    nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    print(w)
    print(w.std().item())

    nn.init.normal_(w)
    print(w)
    print(w.std().item())
    nn.init.xavier_normal_(w)
    print(w)
    print(w.std().item())
    nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    print(w)
    print(w.std().item())
"""

"""
def test():
    w1 = torch.empty(3, 5)
    w2 = torch.empty(24, 40)
    a = 1.0 / math.sqrt(15)
    b = 1.0 / math.sqrt(960)
    nn.init.uniform_(w1, -a, a)
    print(w1)
    print(w1.std().item())
    nn.init.uniform_(w2, -b, b)
    print(w2)
    print(w2.std().item())

    nn.init.kaiming_normal_(w1, mode='fan_out', nonlinearity='relu')
    print(w1)
    print(w1.std().item())
    nn.init.kaiming_normal_(w2, mode='fan_out', nonlinearity='relu')
    print(w2)
    print(w2.std().item())
"""


"""
def test():
    x = np.arange(0.1, 10, 0.1)
    y1 = []
    y2 = []
    for i in x:
        result = i - 1
        y1.append(result)
        result = math.log(i, 2)
        y2.append(result)
    plt.plot(x, y1, label="kx")
    plt.plot(x, y2, label="logx")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
"""


"""
def test(model_name):
    key_list = ['conv1.weight', 'conv2.weight']
    in_channel = [3, 5]
    out_channel = [6, 4]
    value_list = [torch.ones((in_channel[i], out_channel[i])) for i in range(0, len(key_list))]
    z_dict = dict(zip(key_list, value_list))
    print(z_dict)
"""


"""
def test():
    # 创建剪枝矩阵
    layer_in_channel = [3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]
    layer_out_channel = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight', 'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight']
    value_list = [torch.zeros((layer_out_channel[i], layer_in_channel[i], 3, 3)) for i in range(0, len(weight_name))]
    mask = dict(zip(weight_name, value_list))

    print(mask.keys())

    # save
    with open('best.pkl', 'wb') as f:
        pkl.dump(mask, f, pkl.HIGHEST_PROTOCOL)

    # load
    with open('best.pkl', 'rb') as f:
        mask_info = pkl.load(f)

    print(mask_info)
"""


"""
def test():
    a = np.array([[[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[2, 3, 4], [2, 3, 4], [2, 3, 4]], [[3, 4, 5], [3, 4, 5], [3, 4, 5]]], [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[2, 3, 4], [2, 3, 4], [2, 3, 4]], [[3, 4, 5], [3, 4, 5], [3, 4, 5]]]])
    b = np.array([[[7, 8, 9], [7, 8, 9], [4, 5, 6]], [[1, 2, 3], [7, 8, 9], [1, 2, 3]], [[4, 5, 6], [1, 2, 3], [7, 8, 9]]])
    c = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    d = np.array([1, 2, 3])
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    c = torch.from_numpy(c)
    d = torch.from_numpy(d)
    print(a - b)
    print(a - c)
    print(a - d)
"""


"""
def test():
    model_original = Vgg16(10)
    print(model_original.state_dict()['conv1.weight'])
    model_original.state_dict()['conv1.weight'] = torch.zeros((64, 3, 3, 3))
    print(model_original.state_dict()['conv1.weight'])
    for i in range(0, 64):
        model_original.state_dict()['conv1.weight'][i] = torch.zeros((3, 3, 3))
    print(model_original.state_dict()['conv1.weight'])
"""


"""
def test():
    # 将模型参数以txt形式存储
    model = Vgg16(10)
    model.load_state_dict(torch.load('model_Vgg16_weight_pattern_shape_translate_after_translate_parameters.pth'))
    str_parameters = ''
    for parameters in model.parameters():
        str_parameters = str_parameters + str(parameters) + str('\n')
    f_parameters = open('parameters_Vgg16_weight_pattern_shape_translate.txt', 'w', encoding='utf-8')
    f_parameters.write(str_parameters)
    f_parameters.close()
"""


"""
def test():
    a = torch.rand(3, 3, 3)
    print(a)
    print(torch.max(a))
    b = [1.00, 1.10, 1.20]
    c = [2.00, 2.10, 2.30]
    for i in range(0, 3):
        b[i] = b[i] + c[i]
    print(b)
"""

"""
def test():
    a = torch.rand(3, 3, 3, 3, 8)
    a = a.numpy()
    print(a.shape)
    print(a)
    a.resize((3, 3, 9, 8))
    print(a.shape)
    # print(a)
    print(a[0][1:3].shape)
"""

"""
def test():
    a = torch.rand(3, 8, 3, 3)
    print(a)
    print(a.shape[0])
    a[0][0] = torch.zeros(3, 3)
    print(a.shape)
    b = torch.zeros(3, 6)
    a = b
    print(a.shape)
"""


def value_test():
    """
    model = Res18(10)
    # model.load_state_dict(torch.load('model_Res18_weight_pattern_value_normalized_translate_after_translate_parameters.pth'))  # 加载训练好的原始模型
    model.load_state_dict(torch.load('model_Res18_weight_pattern_shape_and_value_normalized_translate_after_translate_parameters.pth'))  # 加载训练好的原始模型
    # with open('model_Res18_value_normalized_map_information' + '.pkl', 'rb') as f:
    with open('model_Res18_shape_and_value_normalized_map_information' + '.pkl', 'rb') as f:
        map_information = pkl.load(f)
        f.close()
    # with open('model_Res18_value_multiple_relationship_information' + '.pkl', 'rb') as f:
    with open('model_Res18_shape_and_value_multiple_relationship_information' + '.pkl', 'rb') as f:
        multiple_relationship_information = pkl.load(f)
        f.close()
    weight_name = 'conv5.weight'
    """

    """
    model = Vgg16(10)
    model.load_state_dict(torch.load('model_Vgg16_weight_pattern_value_normalized_translate_after_translate_parameters.pth'))  # 加载训练好的原始模型
    # model.load_state_dict(torch.load('model_Vgg16_weight_pattern_shape_and_value_normalized_translate_after_translate_parameters.pth'))  # 加载训练好的原始模型
    with open('model_Vgg16_value_normalized_map_information' + '.pkl', 'rb') as f:
    # with open('model_Vgg16_shape_and_value_normalized_map_information' + '.pkl', 'rb') as f:
        map_information = pkl.load(f)
        f.close()
    with open('model_Vgg16_value_multiple_relationship_information' + '.pkl', 'rb') as f:
    # with open('model_Vgg16_shape_and_value_multiple_relationship_information' + '.pkl', 'rb') as f:
        multiple_relationship_information = pkl.load(f)
        f.close()
    weight_name = 'conv5.weight'
    """

    model = WRN(10)
    # model.load_state_dict(torch.load('model_WRN_weight_pattern_value_normalized_translate_after_translate_parameters.pth'))  # 加载训练好的原始模型
    model.load_state_dict(torch.load('model_WRN_weight_pattern_shape_and_value_normalized_translate_after_translate_parameters.pth'))  # 加载训练好的原始模型
    # with open('model_WRN_value_normalized_map_information' + '.pkl', 'rb') as f:
    with open('model_WRN_shape_and_value_normalized_map_information' + '.pkl', 'rb') as f:
        map_information = pkl.load(f)
        f.close()
    # with open('model_WRN_value_multiple_relationship_information' + '.pkl', 'rb') as f:
    with open('model_WRN_shape_and_value_multiple_relationship_information' + '.pkl', 'rb') as f:
        multiple_relationship_information = pkl.load(f)
        f.close()
    weight_name = 'conv13.weight'

    total = 0
    wrong = 0
    correct = 0
    for i in range(0, map_information[weight_name].shape[0]):
        for j in range(0, map_information[weight_name].shape[1]):
            if map_information[weight_name][i][j][0] == -1:
                break
            similarity_x = model.state_dict()[weight_name][map_information[weight_name][i][j][0]][i].abs().sum() / model.state_dict()[weight_name][map_information[weight_name][i][j][1]][i].abs().sum()
            similarity_y = multiple_relationship_information[weight_name][map_information[weight_name][i][j][0]][i][0][0].item()
            if similarity_x != similarity_y and similarity_y != 0:
                # print('wrong')
                print(str(similarity_x) + '  ' + str(similarity_y))
                print(str(model.state_dict()[weight_name][map_information[weight_name][i][j][0]][i].abs().sum()) + '  ' + str(model.state_dict()[weight_name][map_information[weight_name][i][j][1]][i].abs().sum()))
                wrong = wrong + 1
            if similarity_x != similarity_y and similarity_y == 0:
                # print('correct')
                # print(str(similarity_x) + '  ' + str(similarity_y))
                # print(str(model.state_dict()[weight_name][map_information[weight_name][i][j][0]][i].abs().sum()) + '  ' + str(model.state_dict()[weight_name][map_information[weight_name][i][j][1]][i].abs().sum()))
                correct = correct + 1
            if similarity_x == similarity_y:
                # print('correct')
                # print(str(similarity_x) + '  ' + str(similarity_y))
                # print(str(model.state_dict()[weight_name][map_information[weight_name][i][j][0]][i].abs().sum()) + '  ' + str(model.state_dict()[weight_name][map_information[weight_name][i][j][1]][i].abs().sum()))
                correct = correct + 1
            total = total + 1
    print('wrong_percentage:')
    print(wrong / total)
    print('correct_percentage:')
    print(correct / total)
    print('finish analyse')


def shape_test(model_name):
    weight_name = []
    if model_name == 'AlexNet':
        weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight',
                       'fc1.weight', 'fc2.weight', 'fc3.weight']
    if model_name == 'Vgg16':
        weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight',
                       'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight',
                       'fc1.weight', 'fc2.weight', 'fc3.weight']
    if model_name == 'Res18':
        weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight',
                       'conv7.weight', 'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight',
                       'conv13.weight', 'conv14.weight', 'conv15.weight', 'conv16.weight', 'conv17.weight',
                       'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight',
                       'fc.weight']
    if model_name == 'WRN':
        weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight',
                       'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight',
                       'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight',
                       'fc.weight']
    if model_name == 'Res50':
        weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight', 'conv8.weight', 'conv9.weight', 'conv10.weight',
                       'conv11.weight', 'conv12.weight', 'conv13.weight', 'conv14.weight', 'conv15.weight', 'conv16.weight', 'conv17.weight', 'conv18.weight', 'conv19.weight',
                       'conv20.weight', 'conv21.weight', 'conv22.weight', 'conv23.weight', 'conv24.weight', 'conv25.weight', 'conv26.weight', 'conv27.weight', 'conv28.weight',
                       'conv29.weight', 'conv30.weight', 'conv31.weight', 'conv32.weight', 'conv33.weight', 'conv34.weight', 'conv35.weight', 'conv36.weight', 'conv37.weight',
                       'conv38.weight', 'conv39.weight', 'conv40.weight', 'conv41.weight', 'conv42.weight', 'conv43.weight', 'conv44.weight', 'conv45.weight', 'conv46.weight',
                       'conv47.weight', 'conv48.weight', 'conv49.weight',
                       'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight', 'shortcut4.weight',
                       'fc.weight']
    with open('model_' + model_name + '_pattern_mask' + '.pkl', 'rb') as f:
        mask = pkl.load(f)
        f.close()
    for i in range(0, len(weight_name)):
        if 'fc' in weight_name[i]:
            print(mask[weight_name[i]][0].abs().sum())
        else:
            print(mask[weight_name[i]][0][0].abs().sum())
    print(mask[weight_name[1]])


def value_original_test():
    """
    model = Res18(10)
    model.load_state_dict(torch.load('model_Res18_weight_pattern_value_original_translate_after_translate_parameters.pth'))  # 加载训练好的原始模型
    with open('model_Res18_original_map_information' + '.pkl', 'rb') as f:
        map_information = pkl.load(f)
        f.close()
    weight_name = 'conv6.weight'
    """

    model = WRN(10)
    model.load_state_dict(torch.load('model_WRN_weight_pattern_value_original_translate_after_translate_parameters.pth'))  # 加载训练好的原始模型
    with open('model_WRN_original_map_information' + '.pkl', 'rb') as f:
        map_information = pkl.load(f)
        f.close()
    weight_name = 'conv8.weight'

    total = 0
    wrong = 0
    correct = 0
    for i in range(0, map_information[weight_name].shape[0]):
        for j in range(0, map_information[weight_name].shape[1]):
            if map_information[weight_name][i][j][0] == -1:
                break
            weight_x = model.state_dict()[weight_name][map_information[weight_name][i][j][0]][i].abs().sum()
            weight_y = model.state_dict()[weight_name][map_information[weight_name][i][j][1]][i].abs().sum()
            if weight_x != weight_y:
                if map_information[weight_name][i][j][1] == -1 and weight_x == 0:
                    correct = correct + 1
                else:
                    print(str(weight_x) + '  ' + str(weight_y))
                    print(str(map_information[weight_name][i][j][0]) + ' ' + str(map_information[weight_name][i][j][1]))
                    wrong = wrong + 1
            if weight_x == weight_y:
                correct = correct + 1
            total = total + 1
    print('wrong_percentage:')
    print(wrong / total)
    print('correct_percentage:')
    print(correct / total)
    print('finish analyse')


"""
def test():
    a = torch.rand(3, 3)
    print(a)
    b = 1.0 / a
    print(b)
    c = a * b
    print(c)
"""


"""
def test():
    model = Res18(10)
    model.load_state_dict(torch.load('model_Res18_weight_pattern_value_normalized_translate_after_translate_parameters.pth'))  # 加载训练好的原始模型
    for name, par in model.named_parameters():
        print(name)
"""

"""
def test():
    print(round(-127.5))
    print(math.ceil(-127.5))
    print(math.floor(-127.5))

    print(round(127.5))
    print(math.ceil(127.5))
    print(math.floor(127.5))
"""

"""
def test():
    a = torch.rand(5, 3, 3)
    print(a.shape)
    print(a)
    b = a[0:1]
    print(b.shape)
    print(b)
    b = a[0:1][0:2]
    print(b.shape)
    print(b)
"""


if __name__ == '__main__':
    # value_test()
    # shape_test('Res50')

    value_original_test()

    """
    a = torch.FloatTensor([[9.2, 1.1], [9.2, 1.1], [9.2, 1.1]])
    print(a[1, 1])
    print(a[1][1])
    """


    
    
    
    
    
    
    
    
    
    
    