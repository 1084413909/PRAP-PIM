import copy
import time
import torch
import torch.utils.data
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from model import Vgg16
from train_model import test
from cut import pattern_value_original_translate


# 模型参数
model_name = 'Vgg16'
layer_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight', 'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight']
layer_in_channel = [3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]
layer_out_channel = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

# 超参数
lr = 0.1
epoches = 200
batch_size = 128

# 剪枝参数
kernel_keep_ratios = [0.5, 0.25, 0.125, 0.0625]
channel_keep_ratios = [0.75, 0.5, 0.25, 0.125]
translate_epoch = [100, 120, 140, 160, 180, 190, 200]

# 打印数组时不再显示省略号改为全部显示
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


# 创建用于排序的数据结构
class Node:
    def __init__(self, index=0, value=0.0):
        self.index = index
        self.value = value


def get_dataloader(data_name):
    print('...Preparing data...')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if data_name == 'cifar10':
        cifar10_train_size, cifar10_test_size, cifar10_classes, cifar10_input_size = 50000, 10000, 10, (3, 32, 32)
        cifar10_train_dataset = datasets.CIFAR10('./cifar10_data', train=True, transform=transform_train, download=False)
        cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train_dataset, batch_size=batch_size, shuffle=False)
        cifar10_test_dataset = datasets.CIFAR10('./cifar10_data', train=False, transform=transform_test, download=False)
        cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test_dataset, batch_size=batch_size, shuffle=False)
        return cifar10_train_loader, cifar10_test_loader, cifar10_train_size, cifar10_test_size, cifar10_classes, cifar10_input_size


def pre_train(model, model_name, device, optimizer, scheduler, train_loader, test_loader, epoches, translate_epoch):
    accuracy_max = 0  # 记录模型最高准确率
    start_time = time.time()  # 统计训练时间
    best_model = copy.deepcopy(model.state_dict())  # 记录最优模型

    result = pd.DataFrame()  # 记录训练过程数据并存储到csv文件
    train_accuracy_record = [0.0] * epoches  # 记录每个epoch训练集的准确率
    train_loss_record = [0.0] * epoches  # 记录每个epoch训练集的损失值
    test_accuracy_record = [0.0] * epoches  # 记录每个epoch测试集的准确率
    test_loss_record = [0.0] * epoches  # 记录每个epoch测试集的损失值

    for epoch in range(0, epoches):
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

        # 记录最优模型
        train_accuracy_record[epoch] = correct / total
        train_loss_record[epoch] = train_loss
        print('epoch: ' + str(epoch + 1) + '  train_loss: ' + str(train_loss_record[epoch]) + ';  train_accuracy: ' + str(train_accuracy_record[epoch] * 100) + '%')
        test_accuracy_record[epoch], test_loss_record[epoch] = test(model, device, test_loader)
        print('epoch: ' + str(epoch + 1) + '  test_loss: ' + str(test_loss_record[epoch]) + ';  test_accuracy: ' + str(test_accuracy_record[epoch] * 100) + '%')
        if test_accuracy_record[epoch] > accuracy_max:
            accuracy_max = test_accuracy_record[epoch]
            best_model = copy.deepcopy(model.state_dict())

        scheduler.step()  # 余弦退火调整学习率

        if epoch + 1 == translate_epoch[0]:
            model.load_state_dict(best_model)
            checkpoint = {
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, 'model_Vgg16_original_parameter_epoch100_ckpt.pth')

    time_now = time.time() - start_time
    print('Finished Training')
    print('Training complete in {:.0f}m {:.0f}s'.format(time_now // 60, time_now % 60))

    # 将训练过程数据保存到csv文件
    result['Train_Accuracy'] = train_accuracy_record
    result['Train_Loss'] = train_loss_record
    result['Test_Accuracy'] = test_accuracy_record
    result['Test_Loss'] = test_loss_record
    result.to_csv('model_' + model_name + '_train_info' + '.csv')

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), 'model_' + model_name + '_original_parameters.pth')  # 保存最优模型参数
    best_test_accuracy, best_test_loss = test(model, device, test_loader)
    print('Best Result: test_loss: ' + str(best_test_loss) + ';  test_accuracy: ' + str(best_test_accuracy * 100) + '%')

    return best_test_accuracy


def layer_reuse_ratio_and_model_accuracy(model, translate_name, weight_name, in_channel, out_channel, device, optimizer, scheduler, train_loader, test_loader, epoches, threshold, translate_epoch):
    current_iteration = 0  # 记录当前训练epoch数
    accuracy_max = 0  # 记录模型最高准确率
    after_translate_accuracy_best = 0  # 记录转化后模型的最高准确率
    start_time = time.time()  # 统计训练时间
    best_model = copy.deepcopy(model.state_dict())  # 记录最优模型
    checkpoint = torch.load('model_Vgg16_original_parameter_epoch100_ckpt.pth')  # 加载断点

    for epoch in range(checkpoint['epoch'], epoches):
        if epoch == checkpoint['epoch']:
            model.load_state_dict(checkpoint['model'])  # 加载模型参数
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            scheduler.load_state_dict(checkpoint['lr_schedule'])  # 加载学习率优化器
            best_model = copy.deepcopy(model.state_dict())  # 设置最优模型

        else:
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
            train_accuracy_record = correct / total
            print('epoch: ' + str(epoch + 1) + '  train_loss: ' + str(train_loss) + ';  train_accuracy: ' + str(train_accuracy_record * 100) + '%')
            test_accuracy_record, test_loss_record = test(model, device, test_loader)
            print('epoch: ' + str(epoch + 1) + '  test_loss: ' + str(test_loss_record) + ';  test_accuracy: ' + str(test_accuracy_record * 100) + '%')

            if test_accuracy_record > accuracy_max:
                accuracy_max = test_accuracy_record
                best_model = copy.deepcopy(model.state_dict())

        # 模式转换
        if epoch + 1 in translate_epoch:
            model.load_state_dict(best_model)
            before_translate_accuracy, before_translate_loss = test(model, device, test_loader)  # 测试转换前模型准确率
            print('Before_translate_accuracy: ' + str(before_translate_accuracy) + ' Before_translate_loss: ' + str(before_translate_loss))
            print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

            if 'original' in translate_name:
                pattern_value_original_translate(model, in_channel, out_channel, threshold, weight_name)

            print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
            after_translate_accuracy, after_translate_loss = test(model, device, test_loader)  # 测试转换后模型准确率
            print('After_translate_accuracy: ' + str(after_translate_accuracy) + ' After_translate_loss: ' + str(after_translate_loss))
            model_accuracy_difference = before_translate_accuracy - after_translate_accuracy  # 计算模型转换损失
            print('Model_accuracy_difference: ' + str(model_accuracy_difference))
            if after_translate_accuracy > after_translate_accuracy_best:
                torch.save(model.state_dict(), 'model_Vgg16_translate_layer_' + weight_name + '_' + granularity + '_threshold_' + str(threshold) + '_parameters.pth')  # 保存最优模型参数
                after_translate_accuracy_best = after_translate_accuracy  # 更新最佳转换后模型的准确率
                print('After_translate_accuracy_best: ' + str(after_translate_accuracy_best))

            current_iteration = current_iteration + 1

    time_now = time.time() - start_time
    print('Finished Training')
    print('Training complete in {:.0f}m {:.0f}s'.format(time_now // 60, time_now % 60))
    print(weight_name + '_' + str(threshold) + '_' + str(after_translate_accuracy_best))

    return after_translate_accuracy_best


device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 在gpu上训练
if device == 'cuda':
    cudnn.deterministic = True
    cudnn.benchmark = True  # 不改变给定的神经网络结构的情况下，大大提升其训练和预测的速度
train_loader, test_loader, train_size, test_size, num_classes, input_size = get_dataloader('cifar10')  # 构建训练集、测试集
model_original = Vgg16(num_classes).to(device)  # 创建原始模型
optimizer = optim.SGD(model_original.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 创建优化器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)  # 动态学习率
# accuracy_original = pre_train(model_original, model_name, device, optimizer, scheduler, train_loader, test_loader, epoches, translate_epoch)  # 训练原始模型
accuracy_record_kernel = np.zeros((4, 13))  # 记录经过模式转换后的模型准确率
accuracy_loss_kernel = np.zeros((4, 13))  # 记录模式转换造成的准确率损失
accuracy_original = 0.9380


result_all_kernel = pd.DataFrame()
result_all_channel = pd.DataFrame()
for i in range(0, 4):
    for j in range(0, 13):
        # 探索kernel稀疏率
        accuracy_record_kernel[i][j] = layer_reuse_ratio_and_model_accuracy(model_original, 'kernel', layer_name[j], layer_in_channel[j], layer_out_channel[j], device, optimizer, scheduler, train_loader, test_loader, epoches, kernel_keep_ratios[i], translate_epoch)
        accuracy_loss_kernel[i][j] = accuracy_original - accuracy_record_kernel[i][j]
    result_all_kernel['keep_ratio_' + str(kernel_keep_ratios[i])] = accuracy_record_kernel[i].tolist()
    result_all_kernel['accuracy_loss_' + str(kernel_keep_ratios[i])] = accuracy_loss_kernel[i].tolist()
# 将结果存储到csv文件
result_all_kernel.to_csv('kernel_reuse_ratio_and_model_accuracy.csv')
