import os
import math
import torch
import torch.utils.data
import numpy as np
import pickle as pkl
import torch.nn as nn
from model import AlexNet, Vgg16, Res18, Res50, WRN
from torchvision import transforms, datasets


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# model_name = 'Res18'
model_name = 'Res50'
# model_name = 'AlexNet'
# model_name = 'Vgg16'
# model_name = 'WRN'
# translate_name = 'original'
# translate_name = 'weight_pattern_shape_translate'
translate_name = 'weight_pattern_value_original_translate'
# translate_name = 'weight_pattern_value_normalized_translate'
# translate_name = 'weight_pattern_shape_and_value_normalized_translate'
OU_size = 8
batch_size = 128
batch_number = 391


# 中间特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extract_layers, device):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule.to(device)
        self.extract_layers = extract_layers

    def forward(self, x):
        if model_name == 'Vgg16' or model_name == 'AlexNet':
            for name, module in self.submodule._modules.items():
                if 'fc' in name:
                    x = x.view(x.size(0), -1)
                if 'batch' in name:
                    continue
                if name in self.extract_layers:
                    max_value = torch.max(x).item()
                    min_value = torch.min(x).item()
                    if max_value < -min_value:
                        max_value = -min_value
                    scale = (max_value - 0) / 127
                    x = torch.round(x / scale) * scale

                x = module(x)

            return x

        if model_name == 'Res18':
            shortcut_feature_map = torch.zeros(3, 64, 3, 3)
            for name, module in self.submodule._modules.items():
                if 'fc' in name:
                    x = x.view(x.size(0), -1)
                if 'batch' in name or 'shortcut' in name:
                    continue
                if name in self.extract_layers:
                    max_value = torch.max(x).item()
                    min_value = torch.min(x).item()
                    if max_value < -min_value:
                        max_value = -min_value
                    scale = (max_value - 0) / 127
                    x = torch.round(x / scale) * scale

                    if name == 'conv2' or name == 'conv4' or name == 'conv6' or name == 'conv8' or name == 'conv10' or name == 'conv12' or name == 'conv14' or name == 'conv16':
                        shortcut_feature_map = x
                    if name == 'conv3' or name == 'conv5' or name == 'conv9' or name == 'conv13' or name == 'conv17':
                        x = shortcut_feature_map + module(x)
                        continue
                    if name == 'conv7':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut1':
                                x = x + module(shortcut_feature_map)
                                break
                        continue
                    if name == 'conv11':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut2':
                                x = x + module(shortcut_feature_map)
                                break
                        continue
                    if name == 'conv15':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut3':
                                x = x + module(shortcut_feature_map)
                                break
                        continue

                x = module(x)

            return x

        if model_name == 'Res50':
            shortcut_feature_map = torch.zeros(3, 64, 3, 3)
            for name, module in self.submodule._modules.items():
                if 'fc' in name:
                    x = x.view(x.size(0), -1)
                if 'batch' in name or 'shortcut' in name:
                    continue
                if name in self.extract_layers:
                    max_value = torch.max(x).item()
                    min_value = torch.min(x).item()
                    if max_value < -min_value:
                        max_value = -min_value
                    scale = (max_value - 0) / 127
                    x = torch.round(x / scale) * scale

                    if name == 'conv2' or name == 'conv5' or name == 'conv8' or name == 'conv11' or name == 'conv14' or name == 'conv17' or name == 'conv20' or name == 'conv23' or name == 'conv26' or name == 'conv29' or name == 'conv32' or name == 'conv35' or name == 'conv38' or name == 'conv41' or name == 'conv44' or name == 'conv47':
                        shortcut_feature_map = x
                    if name == 'conv7' or name == 'conv10' or name == 'conv16' or name == 'conv19' or name == 'conv22' or name == 'conv28' or name == 'conv31' or name == 'conv34' or name == 'conv37' or name == 'conv40' or name == 'conv46' or name == 'conv49':
                        x = shortcut_feature_map + module(x)
                        continue
                    if name == 'conv4':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut1':
                                x = x + module(shortcut_feature_map)
                                break
                        continue
                    if name == 'conv13':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut2':
                                x = x + module(shortcut_feature_map)
                                break
                        continue
                    if name == 'conv25':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut3':
                                x = x + module(shortcut_feature_map)
                                break
                        continue
                    if name == 'conv43':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut4':
                                x = x + module(shortcut_feature_map)
                                break
                        continue

                x = module(x)

            return x

        if model_name == 'WRN':
            shortcut_feature_map = torch.zeros(3, 64, 3, 3)
            for name, module in self.submodule._modules.items():
                if 'fc' in name:
                    x = x.view(x.size(0), -1)
                if 'batch' in name or 'shortcut' in name:
                    continue
                if name in self.extract_layers:
                    max_value = torch.max(x).item()
                    min_value = torch.min(x).item()
                    if max_value < -min_value:
                        max_value = -min_value
                    scale = (max_value - 0) / 127
                    x = torch.round(x / scale) * scale

                    if name == 'conv2' or name == 'conv4' or name == 'conv6' or name == 'conv8' or name == 'conv10' or name == 'conv12':
                        shortcut_feature_map = x
                    if name == 'conv5' or name == 'conv9' or name == 'conv13':
                        x = shortcut_feature_map + module(x)
                        continue
                    if name == 'conv3':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut1':
                                x = x + module(shortcut_feature_map)
                                break
                        continue
                    if name == 'conv7':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut2':
                                x = x + module(shortcut_feature_map)
                                break
                        continue
                    if name == 'conv11':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut3':
                                x = x + module(shortcut_feature_map)
                                break
                        continue

                x = module(x)

            return x

    def get_layer_feature_map(self, x):
        feature_map_outputs = []
        if model_name == 'Vgg16' or model_name == 'AlexNet':
            for name, module in self.submodule._modules.items():
                if 'fc' in name:
                    x = x.view(x.size(0), -1)
                if 'batch' in name:
                    continue
                if name in self.extract_layers:
                    max_value = torch.max(x).item()
                    min_value = torch.min(x).item()
                    if max_value < -min_value:
                        max_value = -min_value
                    scale = (max_value - 0) / 127
                    x = torch.round(x / scale)
                    feature_map_outputs.append(x)
                    x = x * scale

                x = module(x)

            return feature_map_outputs

        if model_name == 'Res18':
            shortcut_feature_map = torch.zeros(3, 64, 3, 3)
            for name, module in self.submodule._modules.items():
                if 'fc' in name:
                    x = x.view(x.size(0), -1)
                if 'batch' in name or 'shortcut' in name:
                    continue
                if name in self.extract_layers:
                    max_value = torch.max(x).item()
                    min_value = torch.min(x).item()
                    if max_value < -min_value:
                        max_value = -min_value
                    scale = (max_value - 0) / 127
                    x = torch.round(x / scale)
                    feature_map_outputs.append(x)
                    x = x * scale

                    if name == 'conv2' or name == 'conv4' or name == 'conv6' or name == 'conv8' or name == 'conv10' or name == 'conv12' or name == 'conv14' or name == 'conv16':
                        shortcut_feature_map = x
                    if name == 'conv3' or name == 'conv5' or name == 'conv9' or name == 'conv13' or name == 'conv17':
                        x = shortcut_feature_map + module(x)
                        continue
                    if name == 'conv7':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut1':
                                x = x + module(shortcut_feature_map)
                                break
                        continue
                    if name == 'conv11':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut2':
                                x = x + module(shortcut_feature_map)
                                break
                        continue
                    if name == 'conv15':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut3':
                                x = x + module(shortcut_feature_map)
                                break
                        continue

                x = module(x)

            return feature_map_outputs

        if model_name == 'Res50':
            shortcut_feature_map = torch.zeros(3, 64, 3, 3)
            for name, module in self.submodule._modules.items():
                if 'fc' in name:
                    x = x.view(x.size(0), -1)
                if 'batch' in name or 'shortcut' in name:
                    continue
                if name in self.extract_layers:
                    max_value = torch.max(x).item()
                    min_value = torch.min(x).item()
                    if max_value < -min_value:
                        max_value = -min_value
                    scale = (max_value - 0) / 127
                    x = torch.round(x / scale)
                    feature_map_outputs.append(x)
                    x = x * scale

                    if name == 'conv2' or name == 'conv5' or name == 'conv8' or name == 'conv11' or name == 'conv14' or name == 'conv17' or name == 'conv20' or name == 'conv23' or name == 'conv26' or name == 'conv29' or name == 'conv32' or name == 'conv35' or name == 'conv38' or name == 'conv41' or name == 'conv44' or name == 'conv47':
                        shortcut_feature_map = x
                    if name == 'conv7' or name == 'conv10' or name == 'conv16' or name == 'conv19' or name == 'conv22' or name == 'conv28' or name == 'conv31' or name == 'conv34' or name == 'conv37' or name == 'conv40' or name == 'conv46' or name == 'conv49':
                        x = shortcut_feature_map + module(x)
                        continue
                    if name == 'conv4':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut1':
                                x = x + module(shortcut_feature_map)
                                break
                        continue
                    if name == 'conv13':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut2':
                                x = x + module(shortcut_feature_map)
                                break
                        continue
                    if name == 'conv25':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut3':
                                x = x + module(shortcut_feature_map)
                                break
                        continue
                    if name == 'conv43':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut4':
                                x = x + module(shortcut_feature_map)
                                break
                        continue

                x = module(x)

            return feature_map_outputs

        if model_name == 'WRN':
            shortcut_feature_map = torch.zeros(3, 64, 3, 3)
            for name, module in self.submodule._modules.items():
                if 'fc' in name:
                    x = x.view(x.size(0), -1)
                if 'batch' in name or 'shortcut' in name:
                    continue
                if name in self.extract_layers:
                    max_value = torch.max(x).item()
                    min_value = torch.min(x).item()
                    if max_value < -min_value:
                        max_value = -min_value
                    scale = (max_value - 0) / 127
                    x = torch.round(x / scale)
                    feature_map_outputs.append(x)
                    x = x * scale

                    if name == 'conv2' or name == 'conv4' or name == 'conv6' or name == 'conv8' or name == 'conv10' or name == 'conv12':
                        shortcut_feature_map = x
                    if name == 'conv5' or name == 'conv9' or name == 'conv13':
                        x = shortcut_feature_map + module(x)
                        continue
                    if name == 'conv3':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut1':
                                x = x + module(shortcut_feature_map)
                                break
                        continue
                    if name == 'conv7':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut2':
                                x = x + module(shortcut_feature_map)
                                break
                        continue
                    if name == 'conv11':
                        x = module(x)
                        for name, module in self.submodule._modules.items():
                            if name == 'shortcut3':
                                x = x + module(shortcut_feature_map)
                                break
                        continue

                x = module(x)

            return feature_map_outputs


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
        cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train_dataset, batch_size=batch_size, shuffle=True)
        cifar10_test_dataset = datasets.CIFAR10('./cifar10_data', train=False, transform=transform_test, download=False)
        cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test_dataset, batch_size=batch_size, shuffle=False)
        return cifar10_train_loader, cifar10_test_loader, cifar10_train_size, cifar10_test_size, cifar10_classes, cifar10_input_size


def quantize(model, weight_name, out_channnel, translate_name):
    print('start quantize')
    for i in range(0, len(weight_name)):
        max_value = torch.max(model.state_dict()[weight_name[i]]).item()
        min_value = torch.min(model.state_dict()[weight_name[i]]).item()
        if max_value < -min_value:
            max_value = -min_value
        if 'normalized' in translate_name:
            scale = (max_value - 0) / (127 * 256)
        else:
            scale = (max_value - 0) / 127
        for j in range(0, out_channnel[i]):
            model.state_dict()[weight_name[i]][j] = torch.round(model.state_dict()[weight_name[i]][j] / scale) * scale
    print('finish quantize')


def fuse(conv, bn):
    # setting weights
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    conv.weight.copy_(torch.mm(w_bn, w_conv).view(conv.weight.size()))

    # setting bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0))
    b_conv = torch.mm(w_bn, b_conv.view(-1, 1)).view(-1)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    conv.bias.copy_(b_conv + b_bn)


# 测试模型
def test(model, device, test_loader):
    model.eval()  # 不启用Batch Normalization 和 Dropout
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total = total + targets.size(0)
            correct = correct + predicted.eq(targets).sum().item()

    test_accuracy = correct / total
    return test_accuracy


def test_quantize(model, my_extractor, device, test_loader):
    model.eval()  # 不启用Batch Normalization 和 Dropout
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = my_extractor(inputs)

            _, predicted = outputs.max(1)
            total = total + targets.size(0)
            correct = correct + predicted.eq(targets).sum().item()

    test_accuracy = correct / total
    return test_accuracy


def get_bit_weights(model_name, model, weight_name, translate_name):
    weight_matrix_bit = []
    power_value = [64, 32, 16, 8, 4, 2, 1]

    for l in range(0, len(weight_name)):
        if os.path.exists('Model_Weights/' + model_name + '/' + translate_name + '/' + weight_name[l] + '.npy'):
            layer_weights_bit = np.load('Model_Weights/' + model_name + '/' + translate_name + '/' + weight_name[l] + '.npy')
            weight_matrix_bit.append(layer_weights_bit)
        else:
            break
        if l == len(weight_name) - 1:
            return weight_matrix_bit

    for l in range(0, len(weight_name)):
        print('get ' + weight_name[l] + ' weights')
        layer_weights = torch.round(model.state_dict()[weight_name[l]] * 127)
        layer_weights = layer_weights.cpu().numpy()

        if layer_weights.ndim == 1:
            layer_weights_bit = np.zeros((layer_weights.shape[0], 8))
            for i in range(0, layer_weights_bit.shape[0]):
                value = layer_weights[i]
                if value >= 0:
                    layer_weights_bit[i][0] = 0
                else:
                    layer_weights_bit[i][0] = 1
                for k in range(1, layer_weights_bit.shape[1]):
                    layer_weights_bit[i][k] = int(value / power_value[k - 1])
                    value = value % power_value[k - 1]
            layer_weights_bit = np.resize(layer_weights_bit, (layer_weights_bit.shape[0] * layer_weights_bit.shape[1]))
            weight_matrix_bit.append(layer_weights_bit)
            np.save('Model_Weights/' + model_name + '/' + translate_name + '/' + weight_name[l], layer_weights_bit)

        if layer_weights.ndim == 2:
            layer_weights_bit = np.zeros((layer_weights.shape[0], layer_weights.shape[1], 8))
            for i in range(0, layer_weights_bit.shape[0]):
                for j in range(0, layer_weights_bit.shape[1]):
                    value = layer_weights[i][j]
                    if value >= 0:
                        layer_weights_bit[i][j][0] = 0
                    else:
                        layer_weights_bit[i][j][0] = 1
                    for k in range(1, layer_weights_bit.shape[2]):
                        layer_weights_bit[i][j][k] = int(value / power_value[k - 1])
                        value = value % power_value[k - 1]
            layer_weights_bit = layer_weights_bit.swapaxes(0, 1)
            layer_weights_bit = np.resize(layer_weights_bit, (layer_weights_bit.shape[0], layer_weights_bit.shape[1] * layer_weights_bit.shape[2]))
            weight_matrix_bit.append(layer_weights_bit)
            np.save('Model_Weights/' + model_name + '/' + translate_name + '/' + weight_name[l], layer_weights_bit)

        if layer_weights.ndim == 4:
            layer_weights_bit = np.zeros((layer_weights.shape[0], layer_weights.shape[1], layer_weights.shape[2], layer_weights.shape[3], 8))
            for i in range(0, layer_weights_bit.shape[0]):
                for j in range(0, layer_weights_bit.shape[1]):
                    for h in range(0, layer_weights_bit.shape[2]):
                        for w in range(0, layer_weights_bit.shape[3]):
                            value = layer_weights[i][j][w][h]
                            if value >= 0:
                                layer_weights_bit[i][j][h][w][0] = 0
                            else:
                                layer_weights_bit[i][j][h][w][0] = 1
                            for k in range(1, layer_weights_bit.shape[4]):
                                layer_weights_bit[i][j][h][w][k] = int(value / power_value[k - 1])
                                value = value % power_value[k - 1]
            layer_weights_bit = np.resize(layer_weights_bit, (layer_weights_bit.shape[0], layer_weights_bit.shape[1] * layer_weights_bit.shape[2] * layer_weights_bit.shape[3], layer_weights_bit.shape[4]))
            layer_weights_bit = layer_weights_bit.swapaxes(0, 1)
            layer_weights_bit = np.resize(layer_weights_bit, (layer_weights_bit.shape[0], layer_weights_bit.shape[1] * layer_weights_bit.shape[2]))
            weight_matrix_bit.append(layer_weights_bit)
            np.save('Model_Weights/' + model_name + '/' + translate_name + '/' + weight_name[l], layer_weights_bit)

    return weight_matrix_bit


def weight_matrix_analyse(model, weight_name):
    total_zeros_count = 0
    total_weights_count = 0

    for i in range(0, len(weight_name)):
        max_value = torch.max(model.state_dict()[weight_name[i]]).item()
        min_value = torch.min(model.state_dict()[weight_name[i]]).item()
        if max_value < -min_value:
            max_value = -min_value
        scale = (max_value - 0) / 127
        for j in range(0, model.state_dict()[weight_name[i]].shape[0]):
            model.state_dict()[weight_name[i]][j] = torch.round(model.state_dict()[weight_name[i]][j] / scale)

    for l in range(0, len(weight_name)):
        zeros_count = 0
        weights_count = 0
        print('analyse ' + weight_name[l] + '  weight matrix')
        layer_weights = model.state_dict()[weight_name[l]]

        if layer_weights.ndim == 1:
            for i in range(0, layer_weights.shape[0]):
                if layer_weights[i] == 0:
                    zeros_count = zeros_count + 1
                weights_count = weights_count + 1

        if layer_weights.ndim == 2:
            for i in range(0, layer_weights.shape[0]):
                for j in range(0, layer_weights.shape[1]):
                    if layer_weights[i][j] == 0:
                        zeros_count = zeros_count + 1
                    weights_count = weights_count + 1

        if layer_weights.ndim == 4:
            for i in range(0, layer_weights.shape[0]):
                for j in range(0, layer_weights.shape[1]):
                    for w in range(0, layer_weights.shape[2]):
                        for h in range(0, layer_weights.shape[3]):
                            if layer_weights[i][j][w][h] == 0:
                                zeros_count = zeros_count + 1
                            weights_count = weights_count + 1

        print(weight_name[l] + ': ' + str(zeros_count / weights_count))
        total_zeros_count = total_zeros_count + zeros_count
        total_weights_count = total_weights_count + weights_count

    print(str(total_zeros_count / total_weights_count))


def get_bit_feature_maps(model_name, feature_maps, layer_number):
    feature_maps_bit = []
    power_value = [64, 32, 16, 8, 4, 2, 1]

    for l in range(0, layer_number):
        if os.path.exists('Feature_Map/' + model_name + '/layer_' + str(l + 1) + '.npy'):
            layer_feature_map_bit = np.load('Feature_Map/' + model_name + '/layer_' + str(l + 1) + '.npy')
            feature_maps_bit.append(layer_feature_map_bit)
        else:
            break
        if l == layer_number - 1:
            return feature_maps_bit

    for l in range(0, layer_number):
        print('get layer ' + str(l) + ' feature map')
        layer_feature_map = feature_maps[l].cpu().numpy()

        if layer_feature_map.ndim == 2:
            layer_feature_map_bit = np.zeros((layer_feature_map.shape[0], layer_feature_map.shape[1], 8))
            for i in range(0, layer_feature_map_bit.shape[0]):
                for j in range(0, layer_feature_map_bit.shape[1]):
                    value = layer_feature_map[i][j]
                    if value >= 0:
                        layer_feature_map_bit[i][j][0] = 0
                    else:
                        layer_feature_map_bit[i][j][0] = 1
                    for k in range(1, layer_feature_map_bit.shape[2]):
                        layer_feature_map_bit[i][j][k] = int(value / power_value[k - 1])
                        value = value % power_value[k - 1]
            feature_maps_bit.append(layer_feature_map_bit)
            np.save('Feature_Map/' + model_name + '/layer_' + str(l + 1), layer_feature_map_bit)

        if layer_feature_map.ndim == 4:
            layer_feature_map_bit = np.zeros((layer_feature_map.shape[0], layer_feature_map.shape[1], layer_feature_map.shape[2], layer_feature_map.shape[3], 8))
            for i in range(0, layer_feature_map_bit.shape[0]):
                for j in range(0, layer_feature_map_bit.shape[1]):
                    for h in range(0, layer_feature_map_bit.shape[2]):
                        for w in range(0, layer_feature_map_bit.shape[3]):
                            value = layer_feature_map[i][j][w][h]
                            if value >= 0:
                                layer_feature_map_bit[i][j][h][w][0] = 0
                            else:
                                layer_feature_map_bit[i][j][h][w][0] = 1
                            for k in range(1, layer_feature_map_bit.shape[4]):
                                layer_feature_map_bit[i][j][h][w][k] = int(value / power_value[k - 1])
                                value = value % power_value[k - 1]
            feature_maps_bit.append(layer_feature_map_bit)
            np.save('Feature_Map/' + model_name + '/layer_' + str(l + 1), layer_feature_map_bit)

    return feature_maps_bit


def feature_map_analyse(feature_maps_bit, layer_number):
    zeros_count = 0
    total_count = 0
    for l in range(0, layer_number):
        print('analyse layer ' + str(l) + ' feature map')
        layer_feature_map_bit = feature_maps_bit[l]
        if layer_feature_map_bit.ndim == 3:
            for i in range(0, layer_feature_map_bit.shape[0]):
                for j in range(0, layer_feature_map_bit.shape[1]):
                    for k in range(0, layer_feature_map_bit.shape[2]):
                        if layer_feature_map_bit[i][j][k] == 0:
                            zeros_count = zeros_count + 1
                        total_count = total_count + 1
        if layer_feature_map_bit.ndim == 5:
            for i in range(0, layer_feature_map_bit.shape[0]):
                for j in range(0, layer_feature_map_bit.shape[1]):
                    for w in range(0, layer_feature_map_bit.shape[2]):
                        for h in range(0, layer_feature_map_bit.shape[3]):
                            for k in range(0, layer_feature_map_bit.shape[4]):
                                if layer_feature_map_bit[i][j][w][h][k] == 0:
                                    zeros_count = zeros_count + 1
                                total_count = total_count + 1
    print(str(zeros_count / total_count))


def logical_improvement_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, feature_map_size, OU_size):
    total_calculation_number = 0
    actual_calculation_number = 0

    for i in range(0, len(layer_in_channel)):
        layer_total_calculation_number = (layer_in_channel[i] * kernel_size[i] * kernel_size[i] / OU_size) * layer_out_channel[i] * (feature_map_size[i] * feature_map_size[i])
        layer_actual_calculation_number = (layer_in_channel[i] * pattern_value_number[i] / OU_size) * (layer_out_channel[i] * best_keep_ratio[i]) * (feature_map_size[i] * feature_map_size[i])
        print(weight_name[i] + ': ' + str(layer_actual_calculation_number / layer_total_calculation_number))
        print(layer_total_calculation_number)

        total_calculation_number = total_calculation_number + layer_total_calculation_number
        actual_calculation_number = actual_calculation_number + layer_actual_calculation_number

    print('logical_calculation_percentage: ' + str(actual_calculation_number / total_calculation_number))
    print('logical_improvement: ' + str(1 / (actual_calculation_number / total_calculation_number)))



device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader, test_loader, train_size, test_size, num_classes, input_size = get_dataloader('cifar10')  # 构建训练集、测试集


if model_name == 'AlexNet':
    model = AlexNet(num_classes).to(device)
    extract_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',
                      'fc1', 'fc2', 'fc3']
    weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight',
                   'fc1.weight', 'fc2.weight', 'fc3.weight',
                   'conv1.bias', 'conv2.bias', 'conv3.bias', 'conv4.bias', 'conv5.bias',
                   'fc1.bias', 'fc2.bias', 'fc3.bias']
    feature_map_size = [32, 16, 8, 8, 8,
                        1, 1, 1]
    kernel_size = [3, 3, 3, 3, 3,
                   1, 1, 1]
    layer_in_channel = [3, 96, 256, 384, 384,
                        4096, 4096, 4096]
    layer_out_channel = [96, 256, 384, 384, 256,
                         4096, 4096, num_classes,
                         96, 256, 384, 384, 256,
                         4096, 4096, num_classes]
    pattern_value_number = [9, 9, 9, 9, 9,
                            1, 1, 1]
    best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0]

    if translate_name == 'original':
        model.load_state_dict(torch.load('model_' + model_name + '_original_parameters.pth'))
    elif translate_name == 'weight_pattern_shape_translate':
        pattern_value_number = [8, 4, 4, 4, 4,
                                0.25, 0.25, 1]
        model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_shape_translate_after_translate_parameters.pth'))
    elif translate_name == 'weight_pattern_value_original_translate':
        with open('model_' + model_name + 'original_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
        # model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_value_original_translate_after_translate_parameters.pth'))
    elif translate_name == 'weight_pattern_value_normalized_translate':
        with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
        model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_value_normalized_translate_after_translate_parameters.pth'))
    else:
        pattern_value_number = [8, 4, 4, 4, 4,
                                0.25, 0.25, 1]
        with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
        model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_shape_and_value_normalized_translate_after_translate_parameters.pth'))
    logical_improvement_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, feature_map_size, OU_size)

    model.eval()
    torch.set_grad_enabled(False)
    original_accuracy = test(model, device, test_loader)
    print(original_accuracy)

    fuse(model.conv1, model.batch1)
    fuse(model.conv2, model.batch2)
    fuse(model.conv3, model.batch3)
    fuse(model.conv4, model.batch4)
    fuse(model.conv5, model.batch5)

    quantize(model, weight_name, layer_out_channel, translate_name)
    my_extractor = FeatureExtractor(model, extract_layers, device)  # 输出是一个网络
    test_quantize_accuracy = test_quantize(model, my_extractor, device, test_loader)
    print(test_quantize_accuracy)

    """
    weight_matrix_bit = get_bit_weights(model_name, model, weight_name, translate_name)
    weight_matrix_analyse(model, weight_name)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            feature_maps = my_extractor.get_layer_feature_map(inputs)
            feature_maps_bit = get_bit_feature_maps(model_name, feature_maps, len(layer_in_channel))
            feature_map_analyse(feature_maps_bit, len(layer_in_channel))
            weight_matrix_analyse(weight_matrix_bit, weight_name)
            break
    """


if model_name == 'Vgg16':
    model = Vgg16(num_classes).to(device)
    extract_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7',
                      'conv8', 'conv9', 'conv10', 'conv11', 'conv12', 'conv13',
                      'fc1', 'fc2', 'fc3']
    weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight',
                   'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight',
                   'fc1.weight', 'fc2.weight', 'fc3.weight',
                   'conv1.bias', 'conv2.bias', 'conv3.bias', 'conv4.bias', 'conv5.bias', 'conv6.bias', 'conv7.bias',
                   'conv8.bias', 'conv9.bias', 'conv10.bias', 'conv11.bias', 'conv12.bias', 'conv13.bias',
                   'fc1.bias', 'fc2.bias', 'fc3.bias']
    feature_map_size = [32, 32, 16, 16, 8, 8, 8,
                        4, 4, 4, 2, 2, 2,
                        1, 1, 1]
    kernel_size = [3, 3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3, 3,
                   1, 1, 1]
    layer_in_channel = [3, 64, 64, 128, 128, 256, 256,
                        256, 512, 512, 512, 512, 512,
                        512, 4096, 4096]
    layer_out_channel = [64, 64, 128, 128, 256, 256, 256,
                         512, 512, 512, 512, 512, 512,
                         4096, 4096, num_classes,
                         64, 64, 128, 128, 256, 256, 256,
                         512, 512, 512, 512, 512, 512,
                         4096, 4096, num_classes]
    pattern_value_number = [9, 9, 9, 9, 9, 9, 9,
                            9, 9, 9, 9, 9, 9,
                            1, 1, 1]
    best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0]

    if translate_name == 'original':
        model.load_state_dict(torch.load('model_' + model_name + '_original_parameters.pth'))
    elif translate_name == 'weight_pattern_shape_translate':
        pattern_value_number = [8, 8, 8, 4, 4, 4, 4,
                                4, 2, 2, 2, 2, 2,
                                0.25, 0.25, 1]
        model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_shape_translate_after_translate_parameters.pth'))
    elif translate_name == 'weight_pattern_value_original_translate':
        with open('model_' + model_name + 'original_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
            print(best_keep_ratio[i])
        # model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_value_original_translate_after_translate_parameters.pth'))
    elif translate_name == 'weight_pattern_value_normalized_translate':
        with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
            print(best_keep_ratio[i])
        model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_value_normalized_translate_after_translate_parameters.pth'))
    else:
        pattern_value_number = [8, 8, 8, 4, 4, 4, 4,
                                4, 2, 2, 2, 2, 2,
                                0.25, 0.25, 1]
        with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
            print(best_keep_ratio[i])
        model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_shape_and_value_normalized_translate_after_translate_parameters.pth'))
    logical_improvement_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, feature_map_size, OU_size)

    """
    model.eval()
    torch.set_grad_enabled(False)
    original_accuracy = test(model, device, test_loader)
    print(original_accuracy)

    fuse(model.conv1, model.batch1)
    fuse(model.conv2, model.batch2)
    fuse(model.conv3, model.batch3)
    fuse(model.conv4, model.batch4)
    fuse(model.conv5, model.batch5)
    fuse(model.conv6, model.batch6)
    fuse(model.conv7, model.batch7)
    fuse(model.conv8, model.batch8)
    fuse(model.conv9, model.batch9)
    fuse(model.conv10, model.batch10)
    fuse(model.conv11, model.batch11)
    fuse(model.conv12, model.batch12)
    fuse(model.conv13, model.batch13)

    quantize(model, weight_name, layer_out_channel, translate_name)
    my_extractor = FeatureExtractor(model, extract_layers, device)  # 输出是一个网络
    test_quantize_accuracy = test_quantize(model, my_extractor, device, test_loader)
    print(test_quantize_accuracy)
    """

    """
    weight_matrix_bit = get_bit_weights(model_name, model, weight_name, translate_name)
    weight_matrix_analyse(model, weight_name)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            feature_maps = my_extractor.get_layer_feature_map(inputs)
            feature_maps_bit = get_bit_feature_maps(model_name, feature_maps, len(layer_in_channel))
            feature_map_analyse(feature_maps_bit, len(layer_in_channel))
            weight_matrix_analyse(weight_matrix_bit, weight_name)
            break
    """


if model_name == 'Res18':
    model = Res18(num_classes).to(device)
    extract_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6',
                      'conv7', 'conv8', 'conv9', 'conv10', 'conv11', 'conv12',
                      'conv13', 'conv14', 'conv15', 'conv16', 'conv17',
                      'shortcut1', 'shortcut2', 'shortcut3',
                      'fc']
    weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight',
                   'conv7.weight', 'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight',
                   'conv13.weight', 'conv14.weight', 'conv15.weight', 'conv16.weight', 'conv17.weight',
                   'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight',
                   'fc.weight',
                   'conv1.bias', 'conv2.bias', 'conv3.bias', 'conv4.bias', 'conv5.bias', 'conv6.bias',
                   'conv7.bias', 'conv8.bias', 'conv9.bias', 'conv10.bias', 'conv11.bias', 'conv12.bias',
                   'conv13.bias', 'conv14.bias', 'conv15.bias', 'conv16.bias', 'conv17.bias',
                   'shortcut1.bias', 'shortcut2.bias', 'shortcut3.bias',
                   'fc.bias']
    feature_map_size = [32, 32, 32, 32, 32, 16,
                        16, 16, 16, 8, 8, 8,
                        8, 4, 4, 4, 4,
                        16, 8, 4,
                        1]
    kernel_size = [3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3,
                   1, 1, 1,
                   1]
    layer_in_channel = [3, 64, 64, 64, 64, 64,
                        128, 128, 128, 128, 256, 256,
                        256, 256, 512, 512, 512,
                        64, 128, 256,
                        512]
    layer_out_channel = [64, 64, 64, 64, 64, 128,
                         128, 128, 128, 256, 256, 256,
                         256, 512, 512, 512, 512,
                         128, 256, 512,
                         num_classes,
                         64, 64, 64, 64, 64, 128,
                         128, 128, 128, 256, 256, 256,
                         256, 512, 512, 512, 512,
                         128, 256, 512,
                         num_classes
                         ]
    pattern_value_number = [9, 9, 9, 9, 9, 9,
                            9, 9, 9, 9, 9, 9,
                            9, 9, 9, 9, 9,
                            1, 1, 1,
                            1]
    best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0,
                       1.0]

    if translate_name == 'original':
        model.load_state_dict(torch.load('model_' + model_name + '_original_parameters.pth'))
    elif translate_name == 'weight_pattern_shape_translate':
        pattern_value_number = [8, 4, 4, 4, 4, 4,
                                4, 4, 4, 4, 4, 4,
                                4, 4, 2, 2, 2,
                                1, 1, 1,
                                1]
        model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_shape_translate_after_translate_parameters.pth'))
    elif translate_name == 'weight_pattern_value_original_translate':
        with open('model_' + model_name + 'original_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
            print(best_keep_ratio[i])
        # model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_value_original_translate_after_translate_parameters.pth'))
    elif translate_name == 'weight_pattern_value_normalized_translate':
        with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
            print(best_keep_ratio[i])
        model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_value_normalized_translate_after_translate_parameters.pth'))
    else:
        pattern_value_number = [8, 4, 4, 4, 4, 4,
                                4, 4, 4, 4, 4, 4,
                                4, 4, 2, 2, 2,
                                1, 1, 1,
                                1]
        with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
            print(best_keep_ratio[i])
        model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_shape_and_value_normalized_translate_after_translate_parameters.pth'))
    logical_improvement_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, feature_map_size, OU_size)

    """
    model.eval()
    torch.set_grad_enabled(False)
    original_accuracy = test(model, device, test_loader)
    print(original_accuracy)

    fuse(model.conv1, model.batch1)
    fuse(model.conv2, model.batch2)
    fuse(model.conv3, model.batch3)
    fuse(model.conv4, model.batch4)
    fuse(model.conv5, model.batch5)
    fuse(model.conv6, model.batch6)
    fuse(model.conv7, model.batch7)
    fuse(model.conv8, model.batch8)
    fuse(model.conv9, model.batch9)
    fuse(model.conv10, model.batch10)
    fuse(model.conv11, model.batch11)
    fuse(model.conv12, model.batch12)
    fuse(model.conv13, model.batch13)
    fuse(model.conv14, model.batch14)
    fuse(model.conv15, model.batch15)
    fuse(model.conv16, model.batch16)
    fuse(model.conv17, model.batch17)
    fuse(model.shortcut1, model.shortcut_batch1)
    fuse(model.shortcut2, model.shortcut_batch2)
    fuse(model.shortcut3, model.shortcut_batch3)

    quantize(model, weight_name, layer_out_channel, translate_name)
    my_extractor = FeatureExtractor(model, extract_layers, device)  # 输出是一个网络
    test_quantize_accuracy = test_quantize(model, my_extractor, device, test_loader)
    print(test_quantize_accuracy)
    """

    """
    weight_matrix_bit = get_bit_weights(model_name, model, weight_name, translate_name)
    weight_matrix_analyse(model, weight_name)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            feature_maps = my_extractor.get_layer_feature_map(inputs)
            feature_maps_bit = get_bit_feature_maps(model_name, feature_maps, len(layer_in_channel))
            feature_map_analyse(feature_maps_bit, len(layer_in_channel))
            weight_matrix_analyse(weight_matrix_bit, weight_name)
            break
    """


if model_name == 'Res50':
    model = Res50(num_classes).to(device)
    extract_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10',
                      'conv11', 'conv12', 'conv13', 'conv14', 'conv15', 'conv16', 'conv17', 'conv18', 'conv19',
                      'conv20', 'conv21', 'conv22', 'conv23', 'conv24', 'conv25', 'conv26', 'conv27', 'conv28',
                      'conv29', 'conv30', 'conv31', 'conv32', 'conv33', 'conv34', 'conv35', 'conv36', 'conv37',
                      'conv38', 'conv39', 'conv40', 'conv41', 'conv42', 'conv43', 'conv44', 'conv45', 'conv46',
                      'conv47', 'conv48', 'conv49',
                      'shortcut1', 'shortcut2', 'shortcut3', 'shortcut4',
                      'fc']
    weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight', 'conv8.weight', 'conv9.weight', 'conv10.weight',
                   'conv11.weight', 'conv12.weight', 'conv13.weight', 'conv14.weight', 'conv15.weight', 'conv16.weight', 'conv17.weight', 'conv18.weight', 'conv19.weight',
                   'conv20.weight', 'conv21.weight', 'conv22.weight', 'conv23.weight', 'conv24.weight', 'conv25.weight', 'conv26.weight', 'conv27.weight', 'conv28.weight',
                   'conv29.weight', 'conv30.weight', 'conv31.weight', 'conv32.weight', 'conv33.weight', 'conv34.weight', 'conv35.weight', 'conv36.weight', 'conv37.weight',
                   'conv38.weight', 'conv39.weight', 'conv40.weight', 'conv41.weight', 'conv42.weight', 'conv43.weight', 'conv44.weight', 'conv45.weight', 'conv46.weight',
                   'conv47.weight', 'conv48.weight', 'conv49.weight',
                   'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight', 'shortcut4.weight',
                   'fc.weight',
                   'conv1.bias', 'conv2.bias', 'conv3.bias', 'conv4.bias', 'conv5.bias', 'conv6.bias', 'conv7.bias', 'conv8.bias', 'conv9.bias', 'conv10.bias',
                   'conv11.bias', 'conv12.bias', 'conv13.bias', 'conv14.bias', 'conv15.bias', 'conv16.bias', 'conv17.bias', 'conv18.bias', 'conv19.bias',
                   'conv20.bias', 'conv21.bias', 'conv22.bias', 'conv23.bias', 'conv24.bias', 'conv25.bias', 'conv26.bias', 'conv27.bias', 'conv28.bias',
                   'conv29.bias', 'conv30.bias', 'conv31.bias', 'conv32.bias', 'conv33.bias', 'conv34.bias', 'conv35.bias', 'conv36.bias', 'conv37.bias',
                   'conv38.bias', 'conv39.bias', 'conv40.bias', 'conv41.bias', 'conv42.bias', 'conv43.bias', 'conv44.bias', 'conv45.bias', 'conv46.bias',
                   'conv47.bias', 'conv48.bias', 'conv49.bias',
                   'shortcut1.bias', 'shortcut2.bias', 'shortcut3.bias', 'shortcut4.bias',
                   'fc.bias']
    feature_map_size = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                        32, 16, 16, 16, 16, 16, 16, 16, 16,
                        16, 16, 16, 16, 8, 8, 8, 8, 8,
                        8, 8, 8, 8, 8, 8, 8, 8, 8,
                        8, 8, 8, 8, 4, 4, 4, 4, 4,
                        4, 4, 4,
                        32, 16, 8, 4,
                        1]
    kernel_size = [3, 1, 3, 1, 1, 3, 1, 1, 3, 1,
                   1, 3, 1, 1, 3, 1, 1, 3, 1,
                   1, 3, 1, 1, 3, 1, 1, 3, 1,
                   1, 3, 1, 1, 3, 1, 1, 3, 1,
                   1, 3, 1, 1, 3, 1, 1, 3, 1,
                   1, 3, 1,
                   1, 1, 1, 1,
                   1]
    layer_in_channel = [3, 64, 64, 64, 256, 64, 64, 256, 64, 64,
                        256, 128, 128, 512, 128, 128, 512, 128, 128,
                        512, 128, 128, 512, 256, 256, 1024, 256, 256,
                        1024, 256, 256, 1024, 256, 256, 1024, 256, 256,
                        1024, 256, 256, 1024, 512, 512, 2048, 512, 512,
                        2048, 512, 512,
                        64, 256, 512, 1024,
                        2048]
    layer_out_channel = [64, 64, 64, 256, 64, 64, 256, 64, 64, 256,
                         128, 128, 512, 128, 128, 512, 128, 128, 512,
                         128, 128, 512, 256, 256, 1024, 256, 256, 1024,
                         256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
                         256, 256, 1024, 512, 512, 2048, 512, 512, 2048,
                         512, 512, 2048,
                         256, 512, 1024, 2048,
                         num_classes,
                         64, 64, 64, 256, 64, 64, 256, 64, 64, 256,
                         128, 128, 512, 128, 128, 512, 128, 128, 512,
                         128, 128, 512, 256, 256, 1024, 256, 256, 1024,
                         256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
                         256, 256, 1024, 512, 512, 2048, 512, 512, 2048,
                         512, 512, 2048,
                         256, 512, 1024, 2048,
                         num_classes]
    pattern_value_number = [9, 1, 9, 1, 1, 9, 1, 1, 9, 1,
                            1, 9, 1, 1, 9, 1, 1, 9, 1,
                            1, 9, 1, 1, 9, 1, 1, 9, 1,
                            1, 9, 1, 1, 9, 1, 1, 9, 1,
                            1, 9, 1, 1, 9, 1, 1, 9, 1,
                            1, 9, 1,
                            1, 1, 1, 1,
                            1]
    best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0,
                       1.0]

    if translate_name == 'original':
        model.load_state_dict(torch.load('model_' + model_name + '_original_parameters.pth'))
    elif translate_name == 'weight_pattern_shape_translate':
        pattern_value_number = [8, 1, 8, 1, 1, 8, 1, 1, 8, 1,
                                1, 4, 1, 0.5, 4, 1, 0.5, 4, 1,
                                0.5, 4, 1, 0.5, 4, 1, 0.5, 4, 1,
                                0.5, 4, 1, 0.5, 4, 1, 0.5, 4, 1,
                                0.5, 4, 1, 0.5, 2, 0.5, 0.5, 2, 0.5,
                                0.5, 2, 0.5,
                                1, 1, 1, 1,
                                1]
        # model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_shape_translate_after_translate_parameters.pth'))
    elif translate_name == 'weight_pattern_value_original_translate':
        with open('model_' + model_name + 'original_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
            print(best_keep_ratio[i])
        # model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_value_original_translate_after_translate_parameters.pth'))
    elif translate_name == 'weight_pattern_value_normalized_translate':
        with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
            print(best_keep_ratio[i])
        # model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_value_normalized_translate_after_translate_parameters.pth'))
    else:
        pattern_value_number = [8, 1, 8, 1, 1, 8, 1, 1, 8, 1,
                                1, 4, 1, 0.5, 4, 1, 0.5, 4, 1,
                                0.5, 4, 1, 0.5, 4, 1, 0.5, 4, 1,
                                0.5, 4, 1, 0.5, 4, 1, 0.5, 4, 1,
                                0.5, 4, 1, 0.5, 2, 0.5, 0.5, 2, 0.5,
                                0.5, 2, 0.5,
                                1, 1, 1, 1,
                                1]
        with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
            print(best_keep_ratio[i])
        # model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_shape_and_value_normalized_translate_after_translate_parameters.pth'))
    logical_improvement_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, feature_map_size, OU_size)

    """
    model.eval()
    torch.set_grad_enabled(False)
    original_accuracy = test(model, device, test_loader)
    print(original_accuracy)

    fuse(model.conv1, model.batch1)
    fuse(model.conv2, model.batch2)
    fuse(model.conv3, model.batch3)
    fuse(model.conv4, model.batch4)
    fuse(model.conv5, model.batch5)
    fuse(model.conv6, model.batch6)
    fuse(model.conv7, model.batch7)
    fuse(model.conv8, model.batch8)
    fuse(model.conv9, model.batch9)
    fuse(model.conv10, model.batch10)
    fuse(model.conv11, model.batch11)
    fuse(model.conv12, model.batch12)
    fuse(model.conv13, model.batch13)
    fuse(model.conv14, model.batch14)
    fuse(model.conv15, model.batch15)
    fuse(model.conv16, model.batch16)
    fuse(model.conv17, model.batch17)
    fuse(model.conv18, model.batch18)
    fuse(model.conv19, model.batch19)
    fuse(model.conv20, model.batch20)
    fuse(model.conv21, model.batch21)
    fuse(model.conv22, model.batch22)
    fuse(model.conv23, model.batch23)
    fuse(model.conv24, model.batch24)
    fuse(model.conv25, model.batch25)
    fuse(model.conv26, model.batch26)
    fuse(model.conv27, model.batch27)
    fuse(model.conv28, model.batch28)
    fuse(model.conv29, model.batch29)
    fuse(model.conv30, model.batch30)
    fuse(model.conv31, model.batch31)
    fuse(model.conv32, model.batch32)
    fuse(model.conv33, model.batch33)
    fuse(model.conv34, model.batch34)
    fuse(model.conv35, model.batch35)
    fuse(model.conv36, model.batch36)
    fuse(model.conv37, model.batch37)
    fuse(model.conv38, model.batch38)
    fuse(model.conv39, model.batch39)
    fuse(model.conv40, model.batch40)
    fuse(model.conv41, model.batch41)
    fuse(model.conv42, model.batch42)
    fuse(model.conv43, model.batch43)
    fuse(model.conv44, model.batch44)
    fuse(model.conv45, model.batch45)
    fuse(model.conv46, model.batch46)
    fuse(model.conv47, model.batch47)
    fuse(model.conv48, model.batch48)
    fuse(model.conv49, model.batch49)
    fuse(model.shortcut1, model.shortcut_batch1)
    fuse(model.shortcut2, model.shortcut_batch2)
    fuse(model.shortcut3, model.shortcut_batch3)
    fuse(model.shortcut4, model.shortcut_batch4)

    quantize(model, weight_name, layer_out_channel, translate_name)
    my_extractor = FeatureExtractor(model, extract_layers, device)  # 输出是一个网络
    test_quantize_accuracy = test_quantize(model, my_extractor, device, test_loader)
    print(test_quantize_accuracy)
    """
    """
    weight_matrix_bit = get_bit_weights(model_name, model, weight_name, translate_name)
    weight_matrix_analyse(model, weight_name)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            feature_maps = my_extractor.get_layer_feature_map(inputs)
            feature_maps_bit = get_bit_feature_maps(model_name, feature_maps, len(layer_in_channel))
            feature_map_analyse(feature_maps_bit, len(layer_in_channel))
            weight_matrix_analyse(weight_matrix_bit, weight_name)
            break
    """

if model_name == 'WRN':
    model = WRN(num_classes).to(device)
    extract_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7',
                      'conv8', 'conv9', 'conv10', 'conv11', 'conv12', 'conv13',
                      'shortcut1', 'shortcut2', 'shortcut3',
                      'fc']
    weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight',
                   'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight',
                   'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight',
                   'fc.weight',
                   'conv1.bias', 'conv2.bias', 'conv3.bias', 'conv4.bias', 'conv5.bias', 'conv6.bias', 'conv7.bias',
                   'conv8.bias', 'conv9.bias', 'conv10.bias', 'conv11.bias', 'conv12.bias', 'conv13.bias',
                   'shortcut1.bias', 'shortcut2.bias', 'shortcut3.bias',
                   'fc.bias']
    feature_map_size = [32, 32, 32, 32, 32, 16, 16,
                        16, 16, 8, 8, 8, 8,
                        32, 16, 8,
                        1]
    kernel_size = [3, 3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3, 3,
                   1, 1, 1,
                   1]
    layer_in_channel = [3, 16, 128, 128, 128, 128, 256,
                        256, 256, 256, 512, 512, 512,
                        16, 128, 256,
                        512]
    layer_out_channel = [16, 128, 128, 128, 128, 256, 256,
                         256, 256, 512, 512, 512, 512,
                         128, 256, 512,
                         num_classes,
                         16, 128, 128, 128, 128, 256, 256,
                         256, 256, 512, 512, 512, 512,
                         128, 256, 512,
                         num_classes
                         ]
    pattern_value_number = [9, 9, 9, 9, 9, 9, 9,
                            9, 9, 9, 9, 9, 9,
                            1, 1, 1,
                            1]
    best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0,
                       1.0]

    if translate_name == 'original':
        model.load_state_dict(torch.load('model_' + model_name + '_original_parameters.pth'))
    elif translate_name == 'weight_pattern_shape_translate':
        pattern_value_number = [8, 4, 4, 4, 4, 4, 4,
                                4, 4, 4, 2, 2, 2,
                                1, 1, 1,
                                1]
        model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_shape_translate_after_translate_parameters.pth'))
    elif translate_name == 'weight_pattern_value_original_translate':
        with open('model_' + model_name + 'original_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
            print(best_keep_ratio[i])
        # model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_value_original_translate_after_translate_parameters.pth'))
    elif translate_name == 'weight_pattern_value_normalized_translate':
        with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
            print(best_keep_ratio[i])
        model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_value_normalized_translate_after_translate_parameters.pth'))
    else:
        pattern_value_number = [8, 4, 4, 4, 4, 4, 4,
                                4, 4, 4, 2, 2, 2,
                                1, 1, 1,
                                1]
        with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'rb') as f:
            reuse_ratio_information = pkl.load(f)
            f.close()
        for i in range(0, len(extract_layers)):
            best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]
            print(best_keep_ratio[i])
        model.load_state_dict(torch.load('model_' + model_name + '_weight_pattern_shape_and_value_normalized_translate_after_translate_parameters.pth'))
    logical_improvement_analyse(weight_name, layer_in_channel, layer_out_channel, best_keep_ratio, pattern_value_number, kernel_size, feature_map_size, OU_size)

    """
    model.eval()
    torch.set_grad_enabled(False)
    original_accuracy = test(model, device, test_loader)
    print(original_accuracy)

    fuse(model.conv1, model.batch1)
    fuse(model.conv2, model.batch2)
    fuse(model.conv3, model.batch3)
    fuse(model.conv4, model.batch4)
    fuse(model.conv5, model.batch5)
    fuse(model.conv6, model.batch6)
    fuse(model.conv7, model.batch7)
    fuse(model.conv8, model.batch8)
    fuse(model.conv9, model.batch9)
    fuse(model.conv10, model.batch10)
    fuse(model.conv11, model.batch11)
    fuse(model.conv12, model.batch12)
    fuse(model.conv13, model.batch13)
    fuse(model.shortcut1, model.shortcut_batch1)
    fuse(model.shortcut2, model.shortcut_batch2)
    fuse(model.shortcut3, model.shortcut_batch3)

    quantize(model, weight_name, layer_out_channel, translate_name)
    my_extractor = FeatureExtractor(model, extract_layers, device)  # 输出是一个网络
    test_quantize_accuracy = test_quantize(model, my_extractor, device, test_loader)
    print(test_quantize_accuracy)
    """

    """
    weight_matrix_bit = get_bit_weights(model_name, model, weight_name, translate_name)
    weight_matrix_analyse(model, weight_name)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            feature_maps = my_extractor.get_layer_feature_map(inputs)
            feature_maps_bit = get_bit_feature_maps(model_name, feature_maps, len(layer_in_channel))
            feature_map_analyse(feature_maps_bit, len(layer_in_channel))
            weight_matrix_analyse(weight_matrix_bit, weight_name)
            break
    """