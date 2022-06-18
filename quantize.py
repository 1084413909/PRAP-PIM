import os
import copy
import time
import torch
import torch.quantization
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model import Vgg16


def test(model, device, test_loader):
    model.eval()  # 不启用Batch Normalization 和 Dropout
    total = 0
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            test_loss = test_loss + loss.item()
            _, predicted = outputs.max(1)
            total = total + targets.size(0)
            correct = correct + predicted.eq(targets).sum().item()

    test_accuracy = correct / total
    return test_accuracy, test_loss


# 训练模型
def train(model, model_name, device, optimizer, scheduler, train_loader, test_loader, epoches):
    accuracy_max = 0  # 记录模型最高准确率
    start_time = time.time()  # 统计训练时间
    best_model = copy.deepcopy(model.state_dict())  # 记录最优模型

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

        # 记录最优模型
        train_accuracy = correct / total
        train_loss = train_loss
        print('epoch: ' + str(epoch + 1) + '  train_loss: ' + str(train_loss) + ';  train_accuracy: ' + str(train_accuracy * 100) + '%')
        test_accuracy, test_loss = test(model, device, test_loader)
        print('epoch: ' + str(epoch + 1) + '  test_loss: ' + str(test_loss) + ';  test_accuracy: ' + str(test_accuracy * 100) + '%')
        if test_accuracy > accuracy_max:
            accuracy_max = test_accuracy
            best_model = copy.deepcopy(model.state_dict())

        scheduler.step()  # 余弦退火调整学习率

    time_now = time.time() - start_time
    print('Finished Training')
    print('Training complete in {:.0f}m {:.0f}s'.format(time_now // 60, time_now % 60))

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), 'model_' + model_name + '_before_quantize_parameters.pth')  # 保存模型参数


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end='')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

    return top1, top5


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


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
        cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train_dataset, batch_size=100, shuffle=True)
        cifar10_test_dataset = datasets.CIFAR10('./cifar10_data', train=False, transform=transform_test, download=False)
        cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test_dataset, batch_size=100, shuffle=False)
        return cifar10_train_loader, cifar10_test_loader, cifar10_train_size, cifar10_test_size, cifar10_classes, cifar10_input_size


train_loader, test_loader, train_size, test_size, num_classes, input_size = get_dataloader('cifar10')  # 构建训练集、测试集
model_fp32 = Vgg16(num_classes)
model_fp32.load_state_dict(torch.load('model_Vgg16_original_parameters.pth'))
model_fp32 = model_fp32.to('cpu')

print("Size of model before optimal quantization")
print_size_of_model(model_fp32)
model_fp32.eval()
top1, top5 = evaluate(model_fp32, test_loader)  # test acc
print('Evaluation accuracy on %d images, %2.2f' % (test_size, top1.avg))

model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # set the quantize config
print(model_fp32.qconfig)
model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv1', 'batch1', 'relu1'], ['conv2', 'batch2', 'relu2'], ['conv3', 'batch3', 'relu3'], ['conv4', 'batch4', 'relu4'], ['conv5', 'batch5', 'relu5'],
                                                                ['conv6', 'batch6', 'relu6'], ['conv7', 'batch7', 'relu7'], ['conv8', 'batch8', 'relu8'], ['conv9', 'batch9', 'relu9'], ['conv10', 'batch10', 'relu10'],
                                                                ['conv11', 'batch11', 'relu11'], ['conv12', 'batch12', 'relu12'], ['conv13', 'batch13', 'relu13']])
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused, inplace=True)
print("Start calibrate")
with torch.no_grad():
    for image, target in train_loader:
        model_fp32_prepared(image)
print("Calibrate done")
model_int8 = torch.quantization.convert(model_fp32_prepared, inplace=True)  # convert to quantize model

print("Size of model after optimal quantization")
print_size_of_model(model_int8)
top1, top5 = evaluate(model_int8, test_loader)  # test acc
print('Evaluation accuracy on %d images, %2.2f' % (test_size, top1.avg))

torch.jit.save(torch.jit.script(model_int8), 'model_Vgg16_quantization.pth')  # save quantized model
print(model_int8)
