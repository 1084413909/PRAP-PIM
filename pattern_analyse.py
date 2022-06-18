import copy
import time
import torch
import torch.utils.data
import pandas as pd
import torch.nn.functional as F
from train_model import test


# 创建用于排序的数据结构
class Node:
    def __init__(self, index=0, value=0.0):
        self.index = index
        self.value = value


def pattern_shape_translate(model, device, in_channel, out_channel, weight_name, pattern_shape_number):
    pattern_importance = [Node(i, 0.0) for i in range(0, 9877)]
    # 遍历所有kernel统计出现次数最多的形状id
    for c_out in range(0, out_channel):
        for c_in in range(0, in_channel):
            # 获取模式形状编号
            weight_importance = [Node(i + 1, model.state_dict()[weight_name][c_out][c_in][int(i / 3)][int(i % 3)].abs()) for i in range(0, 9)]
            weight_importance.sort(key=lambda f: f.value, reverse=True)
            weight_importance = weight_importance[0:4]
            weight_importance.sort(key=lambda f: f.index, reverse=True)
            pattern_id = 1000 * weight_importance[0].index + 100 * weight_importance[1].index + 10 * weight_importance[2].index + weight_importance[3].index  # 获得每个kernel的形状id
            pattern_importance[pattern_id].value = pattern_importance[pattern_id].value + 1  # 统计每个模式出现的次数
    pattern_importance.sort(key=lambda f: f.value, reverse=True)  # 对各个模式按出现的次数排序
    # 构造出现次数最多模式的形状
    important_pattern_shape = torch.zeros(pattern_shape_number, 3, 3)
    for i in range(0, pattern_shape_number):
        print(str(pattern_importance[i].index) + ' ' + str(pattern_importance[i].value))
        pattern_id = pattern_importance[i].index
        flag = 1000
        for j in range(0, 4):
            location = pattern_id / flag - 1
            pattern_id = pattern_id % flag
            flag = flag / 10
            important_pattern_shape[i][int(location / 3)][int(location % 3)] = 1.0
    important_pattern_shape = important_pattern_shape.to(device)
    # 遍历所有kernel，对每个kernel进行模式剪枝
    for c_out in range(0, out_channel):
        for c_in in range(0, in_channel):
            # 找到最相近的模式
            select_number = (important_pattern_shape * model.state_dict()[weight_name][c_out][c_in]).abs().sum(axis=2).sum(axis=1).argmax().item()
            # 模式形状剪枝
            model.state_dict()[weight_name][c_out][c_in] = model.state_dict()[weight_name][c_out][c_in] * important_pattern_shape[select_number]


def calculate_pattern_difference_original(model, device, in_channel, out_channel, threshold, weight_name):
    # 创建相关变量
    threshold_value = int(out_channel * threshold)  # 最终保留的模式数
    translate_value = out_channel - threshold_value  # 最终转换的模式数
    translate_number = [0] * translate_value  # 转换模式的索引
    keep_number = [0] * threshold_value  # 保留模式的索引
    weightx_value = torch.zeros(translate_value, 3, 3)  # 记录转换模式中的权重数值
    weighty_value = torch.zeros(threshold_value + 1, 3, 3)  # 记录保留模式中的权重数值
    weightx_value = weightx_value.to(device)
    weighty_value = weighty_value.to(device)
    pattern_difference = [0.0] * 512

    # 进行kernel级模式匹配
    for c_in in range(0, in_channel):
        # 找到参数绝对值最小的模式
        weight_importance = [Node(i, 0.0) for i in range(0, out_channel)]  # 统计模式的绝对值大小
        for i in range(0, out_channel):
            weight_importance[i].value = model.state_dict()[weight_name][i][c_in].abs().sum().item()
        weight_importance.sort(key=lambda f: f.value, reverse=False)
        # 将绝对值最小的模式加入转换模式集合
        for i in range(0, translate_value):
            translate_number[i] = weight_importance[i].index
            weightx_value[i] = model.state_dict()[weight_name][translate_number[i]][c_in]
        # 统计保留模式的索引
        for i in range(0, threshold_value):
            keep_number[i] = weight_importance[i + translate_value].index
            weighty_value[i] = model.state_dict()[weight_name][keep_number[i]][c_in]
        # 为每个要剪枝的模式匹配剩余最相似的模式
        for i in range(0, translate_value):
            # 找到最相近的模式
            select_number = (weighty_value - weightx_value[i]).abs().sum(axis=2).sum(axis=1).argmin().item()
            # 计算最相近两个模式之间的差异
            pattern_difference[i] = torch.pow(model.state_dict()[weight_name][translate_number[i]][c_in] - weighty_value[select_number], 2).sum().item()
            # 根据差异的大小转换为和保留模式最相似的模式
            model.state_dict()[weight_name][translate_number[i]][c_in] = weighty_value[select_number]

    return pattern_difference


def calculate_pattern_difference_normalized(model, device, in_channel, out_channel, threshold, weight_name):
    # 创建相关变量
    threshold_value = int(out_channel * threshold)  # 最终保留的模式数
    translate_value = out_channel - threshold_value  # 最终转换的模式数
    translate_number = [0] * translate_value  # 转换模式的索引
    keep_number = [0] * threshold_value  # 保留模式的索引
    weightx_value = torch.zeros(translate_value, 3, 3)  # 记录转换模式中的权重数值
    weighty_value = torch.zeros(threshold_value, 3, 3)  # 记录保留模式中的权重数值
    x_importance = [0.0] * translate_value  # 记录转换模式中的权重标准值
    y_importance = [0.0] * threshold_value  # 记录保留模式中的权重标准值
    weightx_value = weightx_value.to(device)
    weighty_value = weighty_value.to(device)
    pattern_difference = [0.0] * 512

    # 进行kernel级模式匹配
    for c_in in range(0, in_channel):
        # 找到参数绝对值最小的模式
        weight_importance = [Node(i, 0.0) for i in range(0, out_channel)]  # 统计模式的绝对值大小
        for i in range(0, out_channel):
            weight_importance[i].value = model.state_dict()[weight_name][i][c_in].abs().sum().item()
        weight_importance.sort(key=lambda f: f.value, reverse=False)
        # 将绝对值最小的模式加入转换模式集合
        for i in range(0, translate_value):
            translate_number[i] = weight_importance[i].index
            x_importance[i] = model.state_dict()[weight_name][translate_number[i]][c_in].abs().sum().item()
            weightx_value[i] = model.state_dict()[weight_name][translate_number[i]][c_in] / x_importance[i]
        # 统计保留模式的索引
        for i in range(0, threshold_value):
            keep_number[i] = weight_importance[i + translate_value].index
            y_importance[i] = model.state_dict()[weight_name][keep_number[i]][c_in].abs().sum().item()
            weighty_value[i] = model.state_dict()[weight_name][keep_number[i]][c_in] / y_importance[i]
        # 为每个要剪枝的模式匹配剩余最相似的模式
        for i in range(0, translate_value):
            # 找到最相近的模式
            select_number = (weighty_value - weightx_value[i]).abs().sum(axis=2).sum(axis=1).argmin().item()
            # 计算最相近两个模式之间的差异
            pattern_difference[i] = torch.pow(model.state_dict()[weight_name][translate_number[i]][c_in] - weighty_value[select_number] * x_importance[i], 2).sum().item()
            # 根据差异的大小转换为和保留模式最相似的模式
            model.state_dict()[weight_name][translate_number[i]][c_in] = weighty_value[select_number] * x_importance[i]

    return pattern_difference


def get_pattern_difference(model, model_name, translate_name, device, optimizer, scheduler, train_loader, test_loader, epoches, pattern_shape_number, translate_epoch):
    result_all = pd.DataFrame()  # 记录全部结果存储到csv文件
    after_translate_accuracy_best = 0.0  # 记录转换后模型的最高准确率
    before_translate_accuracy = [0.0] * len(translate_epoch)  # 记录转换前模型准确率
    before_translate_loss = [0.0] * len(translate_epoch)  # 记录转换前模型损失值
    after_translate_accuracy = [0.0] * len(translate_epoch)  # 记录转换后模型准确率
    after_translate_loss = [0.0] * len(translate_epoch)  # 记录转换后模型损失值
    model_accuracy_difference = [0.0] * len(translate_epoch)  # 记录模型转换前后的误差

    current_iteration = 0  # 记录当前训练epoch数
    accuracy_max = 0  # 记录模型最高准确率
    start_time = time.time()  # 统计训练时间
    best_model = copy.deepcopy(model.state_dict())  # 记录最优模型

    result_accuracy = pd.DataFrame()  # 记录训练过程数据并存储到csv文件
    train_accuracy_record = [0.0] * epoches  # 记录每个epoch训练集的准确率
    train_loss_record = [0.0] * epoches  # 记录每个epoch训练集的损失值
    test_accuracy_record = [0.0] * epoches  # 记录每个epoch测试集的准确率
    test_loss_record = [0.0] * epoches  # 记录每个epoch测试集的损失值

    for epoch in range(epoches):
        model.train()  # 启用batch normalization和drop out
        total = 0  # 记录样本数
        correct = 0  # 记录总正确数
        train_loss = 0.0  # 记录总损失值

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据加载到gpu
            optimizer.zero_grad()  # 计算梯度
            outputs = model(inputs)  # 前向传播
            loss = F.cross_entropy(outputs, targets)  # 计算损失
            loss.backward()  # 后向传播
            optimizer.step()  # 更新优化器

            # 可视化训练过程
            train_loss = train_loss + loss.item()  # 计算当前损失值
            _, predicted = outputs.max(1)
            total = total + targets.size(0)
            correct = correct + predicted.eq(targets).sum().item()  # 计算当前准确率

        scheduler.step()  # 余弦退火调整学习率

        # 记录最优模型
        train_accuracy_record[epoch] = correct / total
        train_loss_record[epoch] = train_loss
        print('epoch: ' + str(epoch + 1) + '  train_loss: ' + str(train_loss_record[epoch]) + ';  train_accuracy: ' + str(train_accuracy_record[epoch] * 100) + '%')
        test_accuracy_record[epoch], test_loss_record[epoch] = test(model, device, test_loader)
        print('epoch: ' + str(epoch + 1) + '  test_loss: ' + str(test_loss_record[epoch]) + ';  test_accuracy: ' + str(test_accuracy_record[epoch] * 100) + '%')

        if test_accuracy_record[epoch] > accuracy_max:
            accuracy_max = test_accuracy_record[epoch]
            best_model = copy.deepcopy(model.state_dict())

        # 差异分析
        if epoch + 1 in translate_epoch:
            model.load_state_dict(best_model)
            before_translate_accuracy[current_iteration], before_translate_loss[current_iteration] = test(model, device, test_loader)  # 测试转换前模型准确率
            print('Before_translate_accuracy: ' + str(before_translate_accuracy[current_iteration]) + ' Before_translate_loss: ' + str(before_translate_loss[current_iteration]))
            print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

            if 'shape' in translate_name:
                if model_name == 'Vgg16':
                    pattern_shape_translate(model, device, 64, 64, 'conv2.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 64, 128, 'conv3.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 128, 128, 'conv4.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 128, 256, 'conv5.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 256, 256, 'conv6.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 256, 256, 'conv7.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 256, 512, 'conv8.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 512, 512, 'conv9.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 512, 512, 'conv10.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 512, 512, 'conv11.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 512, 512, 'conv12.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 512, 512, 'conv13.weight', pattern_shape_number)
                if model_name == 'Res18':
                    pattern_shape_translate(model, device, 64, 64, 'conv2.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 64, 64, 'conv3.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 64, 64, 'conv4.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 64, 64, 'conv5.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 64, 128, 'conv6.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 128, 128, 'conv7.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 128, 128, 'conv8.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 128, 128, 'conv9.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 128, 256, 'conv10.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 256, 256, 'conv11.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 256, 256, 'conv12.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 256, 256, 'conv13.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 256, 512, 'conv14.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 512, 512, 'conv15.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 512, 512, 'conv16.weight', pattern_shape_number)
                    pattern_shape_translate(model, device, 512, 512, 'conv17.weight', pattern_shape_number)

            if 'original' in translate_name:
                if model_name == 'Vgg16':
                    pattern_difference_conv2 = calculate_pattern_difference_original(model, device, 64, 64, 0.5, 'conv2.weight')
                    pattern_difference_conv3 = calculate_pattern_difference_original(model, device, 64, 128, 0.5, 'conv3.weight')
                    pattern_difference_conv4 = calculate_pattern_difference_original(model, device, 128, 128, 0.5, 'conv4.weight')
                    pattern_difference_conv5 = calculate_pattern_difference_original(model, device, 128, 256, 0.25, 'conv5.weight')
                    pattern_difference_conv6 = calculate_pattern_difference_original(model, device, 256, 256, 0.25, 'conv6.weight')
                    pattern_difference_conv7 = calculate_pattern_difference_original(model, device, 256, 256, 0.25, 'conv7.weight')
                    pattern_difference_conv8 = calculate_pattern_difference_original(model, device, 256, 512, 0.125, 'conv8.weight')
                    pattern_difference_conv9 = calculate_pattern_difference_original(model, device, 512, 512, 0.125, 'conv9.weight')
                    pattern_difference_conv10 = calculate_pattern_difference_original(model, device, 512, 512, 0.125, 'conv10.weight')
                    pattern_difference_conv11 = calculate_pattern_difference_original(model, device, 512, 512, 0.125, 'conv11.weight')
                    pattern_difference_conv12 = calculate_pattern_difference_original(model, device, 512, 512, 0.125, 'conv12.weight')
                    pattern_difference_conv13 = calculate_pattern_difference_original(model, device, 512, 512, 0.125, 'conv13.weight')
                    result_difference = pd.DataFrame()  # 将模式差异结果存储到csv文件
                    result_difference['pattern_difference_conv2'] = pattern_difference_conv2
                    result_difference['pattern_difference_conv3'] = pattern_difference_conv3
                    result_difference['pattern_difference_conv4'] = pattern_difference_conv4
                    result_difference['pattern_difference_conv5'] = pattern_difference_conv5
                    result_difference['pattern_difference_conv6'] = pattern_difference_conv6
                    result_difference['pattern_difference_conv7'] = pattern_difference_conv7
                    result_difference['pattern_difference_conv8'] = pattern_difference_conv8
                    result_difference['pattern_difference_conv9'] = pattern_difference_conv9
                    result_difference['pattern_difference_conv10'] = pattern_difference_conv10
                    result_difference['pattern_difference_conv11'] = pattern_difference_conv11
                    result_difference['pattern_difference_conv12'] = pattern_difference_conv12
                    result_difference['pattern_difference_conv13'] = pattern_difference_conv13
                    result_difference.to_csv('model_' + model_name + '_' + translate_name + '_pattern_difference_' + str(current_iteration) + '.csv')
                if model_name == 'Res18':
                    pattern_difference_conv2 = calculate_pattern_difference_original(model, device, 64, 64, 0.6, 'conv2.weight')
                    pattern_difference_conv3 = calculate_pattern_difference_original(model, device, 64, 64, 0.6, 'conv3.weight')
                    pattern_difference_conv4 = calculate_pattern_difference_original(model, device, 64, 64, 0.6, 'conv4.weight')
                    pattern_difference_conv5 = calculate_pattern_difference_original(model, device, 64, 64, 0.6, 'conv5.weight')
                    pattern_difference_conv6 = calculate_pattern_difference_original(model, device, 64, 128, 0.5, 'conv6.weight')
                    pattern_difference_conv7 = calculate_pattern_difference_original(model, device, 128, 128, 0.5, 'conv7.weight')
                    pattern_difference_conv8 = calculate_pattern_difference_original(model, device, 128, 128, 0.5, 'conv8.weight')
                    pattern_difference_conv9 = calculate_pattern_difference_original(model, device, 128, 128, 0.5, 'conv9.weight')
                    pattern_difference_conv10 = calculate_pattern_difference_original(model, device, 128, 256, 0.4, 'conv10.weight')
                    pattern_difference_conv11 = calculate_pattern_difference_original(model, device, 256, 256, 0.4, 'conv11.weight')
                    pattern_difference_conv12 = calculate_pattern_difference_original(model, device, 256, 256, 0.4, 'conv12.weight')
                    pattern_difference_conv13 = calculate_pattern_difference_original(model, device, 256, 256, 0.4, 'conv13.weight')
                    pattern_difference_conv14 = calculate_pattern_difference_original(model, device, 256, 512, 0.3, 'conv14.weight')
                    pattern_difference_conv15 = calculate_pattern_difference_original(model, device, 512, 512, 0.3, 'conv15.weight')
                    pattern_difference_conv16 = calculate_pattern_difference_original(model, device, 512, 512, 0.3, 'conv16.weight')
                    pattern_difference_conv17 = calculate_pattern_difference_original(model, device, 512, 512, 0.3, 'conv17.weight')
                    result_difference = pd.DataFrame()  # 将模式差异结果存储到csv文件
                    result_difference['pattern_difference_conv2'] = pattern_difference_conv2
                    result_difference['pattern_difference_conv3'] = pattern_difference_conv3
                    result_difference['pattern_difference_conv4'] = pattern_difference_conv4
                    result_difference['pattern_difference_conv5'] = pattern_difference_conv5
                    result_difference['pattern_difference_conv6'] = pattern_difference_conv6
                    result_difference['pattern_difference_conv7'] = pattern_difference_conv7
                    result_difference['pattern_difference_conv8'] = pattern_difference_conv8
                    result_difference['pattern_difference_conv9'] = pattern_difference_conv9
                    result_difference['pattern_difference_conv10'] = pattern_difference_conv10
                    result_difference['pattern_difference_conv11'] = pattern_difference_conv11
                    result_difference['pattern_difference_conv12'] = pattern_difference_conv12
                    result_difference['pattern_difference_conv13'] = pattern_difference_conv13
                    result_difference['pattern_difference_conv14'] = pattern_difference_conv14
                    result_difference['pattern_difference_conv15'] = pattern_difference_conv15
                    result_difference['pattern_difference_conv16'] = pattern_difference_conv16
                    result_difference['pattern_difference_conv17'] = pattern_difference_conv17
                    result_difference.to_csv('model_' + model_name + '_' + translate_name + '_pattern_difference_' + str(current_iteration) + '.csv')

            if 'normalized' in translate_name:
                if model_name == 'Vgg16':
                    pattern_difference_conv2 = calculate_pattern_difference_normalized(model, device, 64, 64, 0.5, 'conv2.weight')
                    pattern_difference_conv3 = calculate_pattern_difference_normalized(model, device, 64, 128, 0.5, 'conv3.weight')
                    pattern_difference_conv4 = calculate_pattern_difference_normalized(model, device, 128, 128, 0.5, 'conv4.weight')
                    pattern_difference_conv5 = calculate_pattern_difference_normalized(model, device, 128, 256, 0.25, 'conv5.weight')
                    pattern_difference_conv6 = calculate_pattern_difference_normalized(model, device, 256, 256, 0.25, 'conv6.weight')
                    pattern_difference_conv7 = calculate_pattern_difference_normalized(model, device, 256, 256, 0.25, 'conv7.weight')
                    pattern_difference_conv8 = calculate_pattern_difference_normalized(model, device, 256, 512, 0.125, 'conv8.weight')
                    pattern_difference_conv9 = calculate_pattern_difference_normalized(model, device, 512, 512, 0.125, 'conv9.weight')
                    pattern_difference_conv10 = calculate_pattern_difference_normalized(model, device, 512, 512, 0.125, 'conv10.weight')
                    pattern_difference_conv11 = calculate_pattern_difference_normalized(model, device, 512, 512, 0.125, 'conv11.weight')
                    pattern_difference_conv12 = calculate_pattern_difference_normalized(model, device, 512, 512, 0.125, 'conv12.weight')
                    pattern_difference_conv13 = calculate_pattern_difference_normalized(model, device, 512, 512, 0.125, 'conv13.weight')
                    result_difference = pd.DataFrame()  # 将模式差异结果存储到csv文件
                    result_difference['pattern_difference_conv2'] = pattern_difference_conv2
                    result_difference['pattern_difference_conv3'] = pattern_difference_conv3
                    result_difference['pattern_difference_conv4'] = pattern_difference_conv4
                    result_difference['pattern_difference_conv5'] = pattern_difference_conv5
                    result_difference['pattern_difference_conv6'] = pattern_difference_conv6
                    result_difference['pattern_difference_conv7'] = pattern_difference_conv7
                    result_difference['pattern_difference_conv8'] = pattern_difference_conv8
                    result_difference['pattern_difference_conv9'] = pattern_difference_conv9
                    result_difference['pattern_difference_conv10'] = pattern_difference_conv10
                    result_difference['pattern_difference_conv11'] = pattern_difference_conv11
                    result_difference['pattern_difference_conv12'] = pattern_difference_conv12
                    result_difference['pattern_difference_conv13'] = pattern_difference_conv13
                    result_difference.to_csv('model_' + model_name + '_' + translate_name + '_pattern_difference_' + str(current_iteration) + '.csv')
                if model_name == 'Res18':
                    pattern_difference_conv2 = calculate_pattern_difference_normalized(model, device, 64, 64, 0.6, 'conv2.weight')
                    pattern_difference_conv3 = calculate_pattern_difference_normalized(model, device, 64, 64, 0.6, 'conv3.weight')
                    pattern_difference_conv4 = calculate_pattern_difference_normalized(model, device, 64, 64, 0.6, 'conv4.weight')
                    pattern_difference_conv5 = calculate_pattern_difference_normalized(model, device, 64, 64, 0.6, 'conv5.weight')
                    pattern_difference_conv6 = calculate_pattern_difference_normalized(model, device, 64, 128, 0.5, 'conv6.weight')
                    pattern_difference_conv7 = calculate_pattern_difference_normalized(model, device, 128, 128, 0.5, 'conv7.weight')
                    pattern_difference_conv8 = calculate_pattern_difference_normalized(model, device, 128, 128, 0.5, 'conv8.weight')
                    pattern_difference_conv9 = calculate_pattern_difference_normalized(model, device, 128, 128, 0.5, 'conv9.weight')
                    pattern_difference_conv10 = calculate_pattern_difference_normalized(model, device, 128, 256, 0.4, 'conv10.weight')
                    pattern_difference_conv11 = calculate_pattern_difference_normalized(model, device, 256, 256, 0.4, 'conv11.weight')
                    pattern_difference_conv12 = calculate_pattern_difference_normalized(model, device, 256, 256, 0.4, 'conv12.weight')
                    pattern_difference_conv13 = calculate_pattern_difference_normalized(model, device, 256, 256, 0.4, 'conv13.weight')
                    pattern_difference_conv14 = calculate_pattern_difference_normalized(model, device, 256, 512, 0.3, 'conv14.weight')
                    pattern_difference_conv15 = calculate_pattern_difference_normalized(model, device, 512, 512, 0.3, 'conv15.weight')
                    pattern_difference_conv16 = calculate_pattern_difference_normalized(model, device, 512, 512, 0.3, 'conv16.weight')
                    pattern_difference_conv17 = calculate_pattern_difference_normalized(model, device, 512, 512, 0.3, 'conv17.weight')
                    result_difference = pd.DataFrame()  # 将模式差异结果存储到csv文件
                    result_difference['pattern_difference_conv2'] = pattern_difference_conv2
                    result_difference['pattern_difference_conv3'] = pattern_difference_conv3
                    result_difference['pattern_difference_conv4'] = pattern_difference_conv4
                    result_difference['pattern_difference_conv5'] = pattern_difference_conv5
                    result_difference['pattern_difference_conv6'] = pattern_difference_conv6
                    result_difference['pattern_difference_conv7'] = pattern_difference_conv7
                    result_difference['pattern_difference_conv8'] = pattern_difference_conv8
                    result_difference['pattern_difference_conv9'] = pattern_difference_conv9
                    result_difference['pattern_difference_conv10'] = pattern_difference_conv10
                    result_difference['pattern_difference_conv11'] = pattern_difference_conv11
                    result_difference['pattern_difference_conv12'] = pattern_difference_conv12
                    result_difference['pattern_difference_conv13'] = pattern_difference_conv13
                    result_difference['pattern_difference_conv14'] = pattern_difference_conv14
                    result_difference['pattern_difference_conv15'] = pattern_difference_conv15
                    result_difference['pattern_difference_conv16'] = pattern_difference_conv16
                    result_difference['pattern_difference_conv17'] = pattern_difference_conv17
                    result_difference.to_csv('model_' + model_name + '_' + translate_name + '_pattern_difference_' + str(current_iteration) + '.csv')

            print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
            after_translate_accuracy[current_iteration], after_translate_loss[current_iteration] = test(model, device, test_loader)  # 测试转换后模型准确率
            print('After_translate_accuracy: ' + str(after_translate_accuracy[current_iteration]) + ' After_translate_loss: ' + str(after_translate_loss[current_iteration]))
            model_accuracy_difference[current_iteration] = before_translate_accuracy[current_iteration] - after_translate_accuracy[current_iteration]  # 计算模型转换损失
            print('Model_accuracy_difference: ' + str(model_accuracy_difference[current_iteration]))
            if after_translate_accuracy[current_iteration] > after_translate_accuracy_best:
                torch.save(model.state_dict(), 'model_' + model_name + '_' + translate_name + '_parameters.pth')  # 保存最优模型参数
                after_translate_accuracy_best = after_translate_accuracy[current_iteration]  # 更新最佳转换后模型的准确率
                print('After_translate_accuracy_best: ' + str(after_translate_accuracy_best))

            current_iteration = current_iteration + 1

    time_now = time.time() - start_time
    print('Finished Training')
    print('Training complete in {:.0f}m {:.0f}s'.format(time_now // 60, time_now % 60))

    # 将训练过程数据保存到csv文件
    result_accuracy['Train_Accuracy'] = train_accuracy_record
    result_accuracy['Train_Loss'] = train_loss_record
    result_accuracy['Test_Accuracy'] = test_accuracy_record
    result_accuracy['Test_Loss'] = test_loss_record
    result_accuracy.to_csv('model_' + model_name + '_train_info' + '.csv')

    # 将剪枝过程数据保存到csv文件
    result_all['Before_Translate_Accuracy'] = before_translate_accuracy
    result_all['Before_Translate_Loss'] = before_translate_loss
    result_all['After_Translate_Accuracy'] = after_translate_accuracy
    result_all['After_Translate_Loss'] = after_translate_loss
    result_all['Model_Difference'] = model_accuracy_difference
    result_all.to_csv('model_' + model_name + '_' + translate_name + '.csv')


def calculate_mean(difference_list):
    pattern_number = 0
    total_difference = 0
    for i in range(0, len(difference_list)):
        if difference_list[i] != 0:
            pattern_number = pattern_number + 1
            total_difference = total_difference + difference_list[i]
    difference_mean = total_difference / pattern_number
    return  difference_mean


def pattern_analyse(model_name, translate_name, translate_epoch):
    if model_name == 'Vgg16':
        result_difference_mean = pd.DataFrame()  # 将模式差异结果存储到csv文件
        # for i in range(0, len(translate_epoch)):
        for i in range(0, 1):
            difference_info = pd.read_csv('model_' + model_name + '_' + translate_name + '_pattern_difference_' + str(i) + '.csv')
            pattern_difference_conv2 = difference_info['pattern_difference_conv2']
            pattern_difference_conv3 = difference_info['pattern_difference_conv3']
            pattern_difference_conv4 = difference_info['pattern_difference_conv4']
            pattern_difference_conv5 = difference_info['pattern_difference_conv5']
            pattern_difference_conv6 = difference_info['pattern_difference_conv6']
            pattern_difference_conv7 = difference_info['pattern_difference_conv7']
            pattern_difference_conv8 = difference_info['pattern_difference_conv8']
            pattern_difference_conv9 = difference_info['pattern_difference_conv9']
            pattern_difference_conv10 = difference_info['pattern_difference_conv10']
            pattern_difference_conv11 = difference_info['pattern_difference_conv11']
            pattern_difference_conv12 = difference_info['pattern_difference_conv12']
            pattern_difference_conv13 = difference_info['pattern_difference_conv13']

            pattern_difference_mean = [0.0] * 12
            pattern_difference_mean[0] = calculate_mean(pattern_difference_conv2)
            pattern_difference_mean[1] = calculate_mean(pattern_difference_conv3)
            pattern_difference_mean[2] = calculate_mean(pattern_difference_conv4)
            pattern_difference_mean[3] = calculate_mean(pattern_difference_conv5)
            pattern_difference_mean[4] = calculate_mean(pattern_difference_conv6)
            pattern_difference_mean[5] = calculate_mean(pattern_difference_conv7)
            pattern_difference_mean[6] = calculate_mean(pattern_difference_conv8)
            pattern_difference_mean[7] = calculate_mean(pattern_difference_conv9)
            pattern_difference_mean[8] = calculate_mean(pattern_difference_conv10)
            pattern_difference_mean[9] = calculate_mean(pattern_difference_conv11)
            pattern_difference_mean[10] = calculate_mean(pattern_difference_conv12)
            pattern_difference_mean[11] = calculate_mean(pattern_difference_conv13)
            result_difference_mean['pattern_difference_mean_iteration_' + str(i + 1)] = pattern_difference_mean
            result_difference_mean.to_csv('model_' + model_name + '_' + translate_name + '_pattern_difference_mean.csv')

    if model_name == 'Vgg18':
        result_difference_mean = pd.DataFrame()  # 将模式差异结果存储到csv文件
        for i in range(0, len(translate_epoch)):
            difference_info = pd.read_csv('model_' + model_name + '_' + translate_name + '_pattern_difference_' + str(i) + '.csv')
            pattern_difference_conv2 = difference_info['pattern_difference_conv2']
            pattern_difference_conv3 = difference_info['pattern_difference_conv3']
            pattern_difference_conv4 = difference_info['pattern_difference_conv4']
            pattern_difference_conv5 = difference_info['pattern_difference_conv5']
            pattern_difference_conv6 = difference_info['pattern_difference_conv6']
            pattern_difference_conv7 = difference_info['pattern_difference_conv7']
            pattern_difference_conv8 = difference_info['pattern_difference_conv8']
            pattern_difference_conv9 = difference_info['pattern_difference_conv9']
            pattern_difference_conv10 = difference_info['pattern_difference_conv10']
            pattern_difference_conv11 = difference_info['pattern_difference_conv11']
            pattern_difference_conv12 = difference_info['pattern_difference_conv12']
            pattern_difference_conv13 = difference_info['pattern_difference_conv13']
            pattern_difference_conv14 = difference_info['pattern_difference_conv14']
            pattern_difference_conv15 = difference_info['pattern_difference_conv15']
            pattern_difference_conv16 = difference_info['pattern_difference_conv16']
            pattern_difference_conv17 = difference_info['pattern_difference_conv17']

            pattern_difference_mean = [0.0] * 12
            pattern_difference_mean[0] = calculate_mean(pattern_difference_conv2)
            pattern_difference_mean[1] = calculate_mean(pattern_difference_conv3)
            pattern_difference_mean[2] = calculate_mean(pattern_difference_conv4)
            pattern_difference_mean[3] = calculate_mean(pattern_difference_conv5)
            pattern_difference_mean[4] = calculate_mean(pattern_difference_conv6)
            pattern_difference_mean[5] = calculate_mean(pattern_difference_conv7)
            pattern_difference_mean[6] = calculate_mean(pattern_difference_conv8)
            pattern_difference_mean[7] = calculate_mean(pattern_difference_conv9)
            pattern_difference_mean[8] = calculate_mean(pattern_difference_conv10)
            pattern_difference_mean[9] = calculate_mean(pattern_difference_conv11)
            pattern_difference_mean[10] = calculate_mean(pattern_difference_conv12)
            pattern_difference_mean[11] = calculate_mean(pattern_difference_conv13)
            pattern_difference_mean[12] = calculate_mean(pattern_difference_conv14)
            pattern_difference_mean[13] = calculate_mean(pattern_difference_conv15)
            pattern_difference_mean[14] = calculate_mean(pattern_difference_conv16)
            pattern_difference_mean[15] = calculate_mean(pattern_difference_conv17)
            result_difference_mean['pattern_difference_mean_iteration_' + str(i + 1)] = pattern_difference_mean
            result_difference_mean.to_csv('model_' + model_name + '_' + translate_name + '_pattern_difference_mean.csv')


if __name__ == '__main__':
    translate_epoch = [100, 120, 140, 160, 180, 190, 200]
    pattern_analyse('Vgg16', 'shape&original', translate_epoch)
