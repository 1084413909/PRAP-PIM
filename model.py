import torch
import torch.nn as nn


class AlexNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(3, 3), padding=1)  # 第1层
        self.batch1 = nn.BatchNorm2d(96, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(3, 3), padding=1)  # 第2层
        self.batch2 = nn.BatchNorm2d(256, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=1)  # 第3层
        self.batch3 = nn.BatchNorm2d(384, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=1)  # 第4层
        self.batch4 = nn.BatchNorm2d(384, affine=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=1)  # 第5层
        self.batch5 = nn.BatchNorm2d(256, affine=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(4096, 4096)  # 第6层
        self.relu6 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)  # 第7层
        self.relu7 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, num_classes)  # 第8层

        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating point to quantized in the quantized model
        x = self.quant(x)

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.batch5(x)
        x = self.relu5(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.fc3(x)

        # manually specify where tensors will be converted from quantized to floating point in the quantized model
        x = self.dequant(x)
        return x


class Vgg16(torch.nn.Module):
    def __init__(self, num_classes):
        super(Vgg16, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1)  # 第1层
        self.batch1 = nn.BatchNorm2d(64, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)  # 第2层
        self.batch2 = nn.BatchNorm2d(64, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)  # 第3层
        self.batch3 = nn.BatchNorm2d(128, affine=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)  # 第4层
        self.batch4 = nn.BatchNorm2d(128, affine=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)  # 第5层
        self.batch5 = nn.BatchNorm2d(256, affine=True)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)  # 第6层
        self.batch6 = nn.BatchNorm2d(256, affine=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)  # 第7层
        self.batch7 = nn.BatchNorm2d(256, affine=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1)  # 第8层
        self.batch8 = nn.BatchNorm2d(512, affine=True)
        self.relu8 = nn.ReLU(inplace=True)

        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)  # 第9层
        self.batch9 = nn.BatchNorm2d(512, affine=True)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)  # 第10层
        self.batch10 = nn.BatchNorm2d(512, affine=True)
        self.relu10 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)  # 第11层
        self.batch11 = nn.BatchNorm2d(512, affine=True)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)  # 第12层
        self.batch12 = nn.BatchNorm2d(512, affine=True)
        self.relu12 = nn.ReLU(inplace=True)

        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1)  # 第13层
        self.batch13 = nn.BatchNorm2d(512, affine=True)
        self.relu13 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512, 4096)  # 第14层
        self.relu14 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)  # 第15层
        self.relu15 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, num_classes)  # 第16层

        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating point to quantized in the quantized model
        x = self.quant(x)

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.batch5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.batch6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.batch7(x)
        x = self.relu7(x)
        x = self.pool3(x)

        x = self.conv8(x)
        x = self.batch8(x)
        x = self.relu8(x)

        x = self.conv9(x)
        x = self.batch9(x)
        x = self.relu9(x)

        x = self.conv10(x)
        x = self.batch10(x)
        x = self.relu10(x)
        x = self.pool4(x)

        x = self.conv11(x)
        x = self.batch11(x)
        x = self.relu11(x)

        x = self.conv12(x)
        x = self.batch12(x)
        x = self.relu12(x)

        x = self.conv13(x)
        x = self.batch13(x)
        x = self.relu13(x)
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu14(x)
        x = self.fc2(x)
        x = self.relu15(x)
        x = self.fc3(x)

        # manually specify where tensors will be converted from quantized to floating point in the quantized model
        x = self.dequant(x)
        return x


class Res18(nn.Module):
    def __init__(self, num_classes):
        super(Res18, self).__init__()
        self.quant = torch.quantization.QuantStub()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第1层
        self.batch1 = nn.BatchNorm2d(64, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第2层
        self.batch2 = nn.BatchNorm2d(64, affine=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第3层
        self.batch3 = nn.BatchNorm2d(64, affine=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第4层
        self.batch4 = nn.BatchNorm2d(64, affine=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第5层
        self.batch5 = nn.BatchNorm2d(64, affine=True)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=True)  # 第6层
        self.batch6 = nn.BatchNorm2d(128, affine=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第7层
        self.batch7 = nn.BatchNorm2d(128, affine=True)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第8层
        self.batch8 = nn.BatchNorm2d(128, affine=True)
        self.relu8 = nn.ReLU(inplace=True)

        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第9层
        self.batch9 = nn.BatchNorm2d(128, affine=True)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv10 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=True)  # 第10层
        self.batch10 = nn.BatchNorm2d(256, affine=True)
        self.relu10 = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第11层
        self.batch11 = nn.BatchNorm2d(256, affine=True)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第12层
        self.batch12 = nn.BatchNorm2d(256, affine=True)
        self.relu12 = nn.ReLU(inplace=True)

        self.conv13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第13层
        self.batch13 = nn.BatchNorm2d(256, affine=True)
        self.relu13 = nn.ReLU(inplace=True)

        self.conv14 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=True)  # 第14层
        self.batch14 = nn.BatchNorm2d(512, affine=True)
        self.relu14 = nn.ReLU(inplace=True)

        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第15层
        self.batch15 = nn.BatchNorm2d(512, affine=True)
        self.relu15 = nn.ReLU(inplace=True)

        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第16层
        self.batch16 = nn.BatchNorm2d(512, affine=True)
        self.relu16 = nn.ReLU(inplace=True)

        self.conv17 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第17层
        self.batch17 = nn.BatchNorm2d(512, affine=True)
        self.relu17 = nn.ReLU(inplace=True)

        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)

        self.shortcut1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(2, 2), padding=0, bias=True)
        self.shortcut_batch1 = nn.BatchNorm2d(128, affine=True)
        self.shortcut2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2, 2), padding=0, bias=True)
        self.shortcut_batch2 = nn.BatchNorm2d(256, affine=True)
        self.shortcut3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), padding=0, bias=True)
        self.shortcut_batch3 = nn.BatchNorm2d(512, affine=True)

        self.fc = nn.Linear(512, num_classes)  # 第18层

        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)

        y = self.conv2(x)
        y = self.batch2(y)
        y = self.relu2(y)
        y = self.conv3(y)
        y = self.batch3(y)
        x = self.relu3(x + y)

        y = self.conv4(x)
        y = self.batch4(y)
        y = self.relu4(y)
        y = self.conv5(y)
        y = self.batch5(y)
        x = self.relu5(x + y)

        y = self.conv6(x)
        y = self.batch6(y)
        y = self.relu6(y)
        y = self.conv7(y)
        y = self.batch7(y)
        x = self.shortcut1(x)
        x = self.shortcut_batch1(x)
        x = self.relu7(x + y)

        y = self.conv8(x)
        y = self.batch8(y)
        y = self.relu8(y)
        y = self.conv9(y)
        y = self.batch9(y)
        x = self.relu9(x + y)

        y = self.conv10(x)
        y = self.batch10(y)
        y = self.relu10(y)
        y = self.conv11(y)
        y = self.batch11(y)
        x = self.shortcut2(x)
        x = self.shortcut_batch2(x)
        x = self.relu11(x + y)

        y = self.conv12(x)
        y = self.batch12(y)
        y = self.relu12(y)
        y = self.conv13(y)
        y = self.batch13(y)
        x = self.relu13(x + y)

        y = self.conv14(x)
        y = self.batch14(y)
        y = self.relu14(y)
        y = self.conv15(y)
        y = self.batch15(y)
        x = self.shortcut3(x)
        x = self.shortcut_batch3(x)
        x = self.relu15(x + y)

        y = self.conv16(x)
        y = self.batch16(y)
        y = self.relu16(y)
        y = self.conv17(y)
        y = self.batch17(y)
        x = self.relu17(x + y)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.dequant(x)
        return x


class Res50(nn.Module):
    def __init__(self, num_classes):
        super(Res50, self).__init__()
        self.quant = torch.quantization.QuantStub()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第1层
        self.batch1 = nn.BatchNorm2d(64, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第2层
        self.batch2 = nn.BatchNorm2d(64, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第3层
        self.batch3 = nn.BatchNorm2d(64, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第4层
        self.batch4 = nn.BatchNorm2d(256, affine=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第5层
        self.batch5 = nn.BatchNorm2d(64, affine=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第6层
        self.batch6 = nn.BatchNorm2d(64, affine=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第7层
        self.batch7 = nn.BatchNorm2d(256, affine=True)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第8层
        self.batch8 = nn.BatchNorm2d(64, affine=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第9层
        self.batch9 = nn.BatchNorm2d(64, affine=True)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第10层
        self.batch10 = nn.BatchNorm2d(256, affine=True)
        self.relu10 = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第11层
        self.batch11 = nn.BatchNorm2d(128, affine=True)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=True)  # 第12层
        self.batch12 = nn.BatchNorm2d(128, affine=True)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv13 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第13层
        self.batch13 = nn.BatchNorm2d(512, affine=True)
        self.relu13 = nn.ReLU(inplace=True)

        self.conv14 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第14层
        self.batch14 = nn.BatchNorm2d(128, affine=True)
        self.relu14 = nn.ReLU(inplace=True)
        self.conv15 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第15层
        self.batch15 = nn.BatchNorm2d(128, affine=True)
        self.relu15 = nn.ReLU(inplace=True)
        self.conv16 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第16层
        self.batch16 = nn.BatchNorm2d(512, affine=True)
        self.relu16 = nn.ReLU(inplace=True)

        self.conv17 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第17层
        self.batch17 = nn.BatchNorm2d(128, affine=True)
        self.relu17 = nn.ReLU(inplace=True)
        self.conv18 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第18层
        self.batch18 = nn.BatchNorm2d(128, affine=True)
        self.relu18 = nn.ReLU(inplace=True)
        self.conv19 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第19层
        self.batch19 = nn.BatchNorm2d(512, affine=True)
        self.relu19 = nn.ReLU(inplace=True)

        self.conv20 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第20层
        self.batch20 = nn.BatchNorm2d(128, affine=True)
        self.relu20 = nn.ReLU(inplace=True)
        self.conv21 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第21层
        self.batch21 = nn.BatchNorm2d(128, affine=True)
        self.relu21 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第22层
        self.batch22 = nn.BatchNorm2d(512, affine=True)
        self.relu22 = nn.ReLU(inplace=True)

        self.conv23 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第23层
        self.batch23 = nn.BatchNorm2d(256, affine=True)
        self.relu23 = nn.ReLU(inplace=True)
        self.conv24 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=True)  # 第24层
        self.batch24 = nn.BatchNorm2d(256, affine=True)
        self.relu24 = nn.ReLU(inplace=True)
        self.conv25 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第25层
        self.batch25 = nn.BatchNorm2d(1024, affine=True)
        self.relu25 = nn.ReLU(inplace=True)

        self.conv26 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第26层
        self.batch26 = nn.BatchNorm2d(256, affine=True)
        self.relu26 = nn.ReLU(inplace=True)
        self.conv27 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第27层
        self.batch27 = nn.BatchNorm2d(256, affine=True)
        self.relu27 = nn.ReLU(inplace=True)
        self.conv28 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第28层
        self.batch28 = nn.BatchNorm2d(1024, affine=True)
        self.relu28 = nn.ReLU(inplace=True)

        self.conv29 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第29层
        self.batch29 = nn.BatchNorm2d(256, affine=True)
        self.relu29 = nn.ReLU(inplace=True)
        self.conv30 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第30层
        self.batch30 = nn.BatchNorm2d(256, affine=True)
        self.relu30 = nn.ReLU(inplace=True)
        self.conv31 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第31层
        self.batch31 = nn.BatchNorm2d(1024, affine=True)
        self.relu31 = nn.ReLU(inplace=True)

        self.conv32 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第32层
        self.batch32 = nn.BatchNorm2d(256, affine=True)
        self.relu32 = nn.ReLU(inplace=True)
        self.conv33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第33层
        self.batch33 = nn.BatchNorm2d(256, affine=True)
        self.relu33 = nn.ReLU(inplace=True)
        self.conv34 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第34层
        self.batch34 = nn.BatchNorm2d(1024, affine=True)
        self.relu34 = nn.ReLU(inplace=True)

        self.conv35 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第35层
        self.batch35 = nn.BatchNorm2d(256, affine=True)
        self.relu35 = nn.ReLU(inplace=True)
        self.conv36 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第36层
        self.batch36 = nn.BatchNorm2d(256, affine=True)
        self.relu36 = nn.ReLU(inplace=True)
        self.conv37 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第37层
        self.batch37 = nn.BatchNorm2d(1024, affine=True)
        self.relu37 = nn.ReLU(inplace=True)

        self.conv38 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第38层
        self.batch38 = nn.BatchNorm2d(256, affine=True)
        self.relu38 = nn.ReLU(inplace=True)
        self.conv39 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第39层
        self.batch39 = nn.BatchNorm2d(256, affine=True)
        self.relu39 = nn.ReLU(inplace=True)
        self.conv40 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第40层
        self.batch40 = nn.BatchNorm2d(1024, affine=True)
        self.relu40 = nn.ReLU(inplace=True)

        self.conv41 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第41层
        self.batch41 = nn.BatchNorm2d(512, affine=True)
        self.relu41 = nn.ReLU(inplace=True)
        self.conv42 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=True)  # 第42层
        self.batch42 = nn.BatchNorm2d(512, affine=True)
        self.relu42 = nn.ReLU(inplace=True)
        self.conv43 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第43层
        self.batch43 = nn.BatchNorm2d(2048, affine=True)
        self.relu43 = nn.ReLU(inplace=True)

        self.conv44 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第44层
        self.batch44 = nn.BatchNorm2d(512, affine=True)
        self.relu44 = nn.ReLU(inplace=True)
        self.conv45 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第45层
        self.batch45 = nn.BatchNorm2d(512, affine=True)
        self.relu45 = nn.ReLU(inplace=True)
        self.conv46 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第46层
        self.batch46 = nn.BatchNorm2d(2048, affine=True)
        self.relu46 = nn.ReLU(inplace=True)

        self.conv47 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第47层
        self.batch47 = nn.BatchNorm2d(512, affine=True)
        self.relu47 = nn.ReLU(inplace=True)
        self.conv48 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第48层
        self.batch48 = nn.BatchNorm2d(512, affine=True)
        self.relu48 = nn.ReLU(inplace=True)
        self.conv49 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)  # 第49层
        self.batch49 = nn.BatchNorm2d(2048, affine=True)
        self.relu49 = nn.ReLU(inplace=True)

        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)

        self.shortcut1 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)
        self.shortcut_batch1 = nn.BatchNorm2d(256, affine=True)
        self.shortcut2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), padding=0, bias=True)
        self.shortcut_batch2 = nn.BatchNorm2d(512, affine=True)
        self.shortcut3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(2, 2), padding=0, bias=True)
        self.shortcut_batch3 = nn.BatchNorm2d(1024, affine=True)
        self.shortcut4 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(2, 2), padding=0, bias=True)
        self.shortcut_batch4 = nn.BatchNorm2d(2048, affine=True)

        self.fc = nn.Linear(2048, num_classes)  # 第50层

        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)

        y = self.conv2(x)
        y = self.batch2(y)
        y = self.relu2(y)
        y = self.conv3(y)
        y = self.batch3(y)
        y = self.relu3(y)
        y = self.conv4(y)
        y = self.batch4(y)
        x = self.shortcut1(x)
        x = self.shortcut_batch1(x)
        x = self.relu4(x + y)

        y = self.conv5(x)
        y = self.batch5(y)
        y = self.relu5(y)
        y = self.conv6(y)
        y = self.batch6(y)
        y = self.relu6(y)
        y = self.conv7(y)
        y = self.batch7(y)
        x = self.relu7(x + y)

        y = self.conv8(x)
        y = self.batch8(y)
        y = self.relu8(y)
        y = self.conv9(y)
        y = self.batch9(y)
        y = self.relu9(y)
        y = self.conv10(y)
        y = self.batch10(y)
        x = self.relu10(x + y)

        y = self.conv11(x)
        y = self.batch11(y)
        y = self.relu11(y)
        y = self.conv12(y)
        y = self.batch12(y)
        y = self.relu12(y)
        y = self.conv13(y)
        y = self.batch13(y)
        x = self.shortcut2(x)
        x = self.shortcut_batch2(x)
        x = self.relu13(x + y)

        y = self.conv14(x)
        y = self.batch14(y)
        y = self.relu14(y)
        y = self.conv15(y)
        y = self.batch15(y)
        y = self.relu15(y)
        y = self.conv16(y)
        y = self.batch16(y)
        x = self.relu16(x + y)

        y = self.conv17(x)
        y = self.batch17(y)
        y = self.relu17(y)
        y = self.conv18(y)
        y = self.batch18(y)
        y = self.relu18(y)
        y = self.conv19(y)
        y = self.batch19(y)
        x = self.relu19(x + y)

        y = self.conv20(x)
        y = self.batch20(y)
        y = self.relu20(y)
        y = self.conv21(y)
        y = self.batch21(y)
        y = self.relu21(y)
        y = self.conv22(y)
        y = self.batch22(y)
        x = self.relu22(x + y)

        y = self.conv23(x)
        y = self.batch23(y)
        y = self.relu23(y)
        y = self.conv24(y)
        y = self.batch24(y)
        y = self.relu24(y)
        y = self.conv25(y)
        y = self.batch25(y)
        x = self.shortcut3(x)
        x = self.shortcut_batch3(x)
        x = self.relu25(x + y)

        y = self.conv26(x)
        y = self.batch26(y)
        y = self.relu26(y)
        y = self.conv27(y)
        y = self.batch27(y)
        y = self.relu27(y)
        y = self.conv28(y)
        y = self.batch28(y)
        x = self.relu28(x + y)

        y = self.conv29(x)
        y = self.batch29(y)
        y = self.relu29(y)
        y = self.conv30(y)
        y = self.batch30(y)
        y = self.relu30(y)
        y = self.conv31(y)
        y = self.batch31(y)
        x = self.relu31(x + y)

        y = self.conv32(x)
        y = self.batch32(y)
        y = self.relu32(y)
        y = self.conv33(y)
        y = self.batch33(y)
        y = self.relu33(y)
        y = self.conv34(y)
        y = self.batch34(y)
        x = self.relu34(x + y)

        y = self.conv35(x)
        y = self.batch35(y)
        y = self.relu35(y)
        y = self.conv36(y)
        y = self.batch36(y)
        y = self.relu36(y)
        y = self.conv37(y)
        y = self.batch37(y)
        x = self.relu37(x + y)

        y = self.conv38(x)
        y = self.batch38(y)
        y = self.relu38(y)
        y = self.conv39(y)
        y = self.batch39(y)
        y = self.relu39(y)
        y = self.conv40(y)
        y = self.batch40(y)
        x = self.relu40(x + y)

        y = self.conv41(x)
        y = self.batch41(y)
        y = self.relu41(y)
        y = self.conv42(y)
        y = self.batch42(y)
        y = self.relu42(y)
        y = self.conv43(y)
        y = self.batch43(y)
        x = self.shortcut4(x)
        x = self.shortcut_batch4(x)
        x = self.relu43(x + y)

        y = self.conv44(x)
        y = self.batch44(y)
        y = self.relu44(y)
        y = self.conv45(y)
        y = self.batch45(y)
        y = self.relu45(y)
        y = self.conv46(y)
        y = self.batch46(y)
        x = self.relu46(x + y)

        y = self.conv47(x)
        y = self.batch47(y)
        y = self.relu47(y)
        y = self.conv48(y)
        y = self.batch48(y)
        y = self.relu48(y)
        y = self.conv49(y)
        y = self.batch49(y)
        x = self.relu49(x + y)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.dequant(x)
        return x


class WRN(torch.nn.Module):
    def __init__(self, num_classes):
        super(WRN, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第1层
        self.batch1 = nn.BatchNorm2d(16, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第2层
        self.batch2 = nn.BatchNorm2d(128, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第3层
        self.batch3 = nn.BatchNorm2d(128, affine=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第4层
        self.batch4 = nn.BatchNorm2d(128, affine=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第5层
        self.batch5 = nn.BatchNorm2d(128, affine=True)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=True)  # 第6层
        self.batch6 = nn.BatchNorm2d(256, affine=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第7层
        self.batch7 = nn.BatchNorm2d(256, affine=True)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第8层
        self.batch8 = nn.BatchNorm2d(256, affine=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第9层
        self.batch9 = nn.BatchNorm2d(256, affine=True)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=True)  # 第10层
        self.batch10 = nn.BatchNorm2d(512, affine=True)
        self.relu10 = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第11层
        self.batch11 = nn.BatchNorm2d(512, affine=True)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第12层
        self.batch12 = nn.BatchNorm2d(512, affine=True)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)  # 第13层
        self.batch13 = nn.BatchNorm2d(512, affine=True)
        self.relu13 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=8, stride=8)

        self.fc = nn.Linear(512, num_classes)  # 第14层

        self.shortcut1 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)
        self.shortcut_batch1 = nn.BatchNorm2d(128, affine=True)
        self.shortcut2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2, 2), padding=0, bias=True)
        self.shortcut_batch2 = nn.BatchNorm2d(256, affine=True)
        self.shortcut3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), padding=0, bias=True)
        self.shortcut_batch3 = nn.BatchNorm2d(512, affine=True)

        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating point to quantized in the quantized model
        x = self.quant(x)

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)

        y = self.conv2(x)
        y = self.batch2(y)
        y = self.relu2(y)
        y = self.conv3(y)
        y = self.batch3(y)
        x = self.shortcut1(x)
        x = self.shortcut_batch1(x)
        x = self.relu3(x + y)

        y = self.conv4(x)
        y = self.batch4(y)
        y = self.relu4(y)
        y = self.conv5(y)
        y = self.batch5(y)
        x = self.relu5(x + y)

        y = self.conv6(x)
        y = self.batch6(y)
        y = self.relu6(y)
        y = self.conv7(y)
        y = self.batch7(y)
        x = self.shortcut2(x)
        x = self.shortcut_batch2(x)
        x = self.relu7(x + y)

        y = self.conv8(x)
        y = self.batch8(y)
        y = self.relu8(y)
        y = self.conv9(y)
        y = self.batch9(y)
        x = self.relu9(x + y)

        y = self.conv10(x)
        y = self.batch10(y)
        y = self.relu10(y)
        y = self.conv11(y)
        y = self.batch11(y)
        x = self.shortcut3(x)
        x = self.shortcut_batch3(x)
        x = self.relu11(x + y)

        y = self.conv12(x)
        y = self.batch12(y)
        y = self.relu12(y)
        y = self.conv13(y)
        y = self.batch13(y)
        x = self.relu13(x + y)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # manually specify where tensors will be converted from quantized to floating point in the quantized model
        x = self.dequant(x)
        return x


class ResNext29(nn.Module):
    def __init__(self, num_classes):
        super(ResNext29, self).__init__()
        self.quant = torch.quantization.QuantStub()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)  # 第1层
        self.batch1 = nn.BatchNorm2d(64, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第2层
        self.batch2 = nn.BatchNorm2d(256, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=4, bias=False)  # 第3层
        self.batch3 = nn.BatchNorm2d(256, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第4层
        self.batch4 = nn.BatchNorm2d(512, affine=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第5层
        self.batch5 = nn.BatchNorm2d(256, affine=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=4, bias=False)  # 第6层
        self.batch6 = nn.BatchNorm2d(256, affine=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第7层
        self.batch7 = nn.BatchNorm2d(512, affine=True)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第8层
        self.batch8 = nn.BatchNorm2d(256, affine=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=4, bias=False)  # 第9层
        self.batch9 = nn.BatchNorm2d(256, affine=True)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第10层
        self.batch10 = nn.BatchNorm2d(512, affine=True)
        self.relu10 = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第11层
        self.batch11 = nn.BatchNorm2d(512, affine=True)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1, groups=4, bias=False)  # 第12层
        self.batch12 = nn.BatchNorm2d(512, affine=True)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0, groups=4, bias=False)  # 第13层
        self.batch13 = nn.BatchNorm2d(1024, affine=True)
        self.relu13 = nn.ReLU(inplace=True)

        self.conv14 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第14层
        self.batch14 = nn.BatchNorm2d(512, affine=True)
        self.relu14 = nn.ReLU(inplace=True)
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=4, bias=False)  # 第15层
        self.batch15 = nn.BatchNorm2d(512, affine=True)
        self.relu15 = nn.ReLU(inplace=True)
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第16层
        self.batch16 = nn.BatchNorm2d(1024, affine=True)
        self.relu16 = nn.ReLU(inplace=True)

        self.conv17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第17层
        self.batch17 = nn.BatchNorm2d(512, affine=True)
        self.relu17 = nn.ReLU(inplace=True)
        self.conv18 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=4, bias=False)  # 第18层
        self.batch18 = nn.BatchNorm2d(512, affine=True)
        self.relu18 = nn.ReLU(inplace=True)
        self.conv19 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第19层
        self.batch19 = nn.BatchNorm2d(1024, affine=True)
        self.relu19 = nn.ReLU(inplace=True)

        self.conv20 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第20层
        self.batch20 = nn.BatchNorm2d(1024, affine=True)
        self.relu20 = nn.ReLU(inplace=True)
        self.conv21 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=1, groups=4, bias=False)  # 第21层
        self.batch21 = nn.BatchNorm2d(1024, affine=True)
        self.relu21 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第22层
        self.batch22 = nn.BatchNorm2d(2048, affine=True)
        self.relu22 = nn.ReLU(inplace=True)

        self.conv23 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第23层
        self.batch23 = nn.BatchNorm2d(1024, affine=True)
        self.relu23 = nn.ReLU(inplace=True)
        self.conv24 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=4, bias=False)  # 第24层
        self.batch24 = nn.BatchNorm2d(1024, affine=True)
        self.relu24 = nn.ReLU(inplace=True)
        self.conv25 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第25层
        self.batch25 = nn.BatchNorm2d(2048, affine=True)
        self.relu25 = nn.ReLU(inplace=True)

        self.conv26 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第26层
        self.batch26 = nn.BatchNorm2d(1024, affine=True)
        self.relu26 = nn.ReLU(inplace=True)
        self.conv27 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=4, bias=False)  # 第27层
        self.batch27 = nn.BatchNorm2d(1024, affine=True)
        self.relu27 = nn.ReLU(inplace=True)
        self.conv28 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第28层
        self.batch28 = nn.BatchNorm2d(2048, affine=True)
        self.relu28 = nn.ReLU(inplace=True)

        self.pool = nn.AvgPool2d(kernel_size=8, stride=8)

        self.shortcut1 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.shortcut_batch1 = nn.BatchNorm2d(512, affine=True)
        self.shortcut2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(2, 2), padding=0, bias=False)
        self.shortcut_batch2 = nn.BatchNorm2d(1024, affine=True)
        self.shortcut3 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(2, 2), padding=0, bias=False)
        self.shortcut_batch3 = nn.BatchNorm2d(2048, affine=True)

        self.fc = nn.Linear(2048, num_classes)  # 第50层

        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)

        y = self.conv2(x)
        y = self.batch2(y)
        y = self.relu2(y)
        y = self.conv3(y)
        y = self.batch3(y)
        y = self.relu3(y)
        y = self.conv4(y)
        y = self.batch4(y)
        x = self.shortcut1(x)
        x = self.shortcut_batch1(x)
        x = self.relu4(x + y)

        y = self.conv5(x)
        y = self.batch5(y)
        y = self.relu5(y)
        y = self.conv6(y)
        y = self.batch6(y)
        y = self.relu6(y)
        y = self.conv7(y)
        y = self.batch7(y)
        x = self.relu7(x + y)

        y = self.conv8(x)
        y = self.batch8(y)
        y = self.relu8(y)
        y = self.conv9(y)
        y = self.batch9(y)
        y = self.relu9(y)
        y = self.conv10(y)
        y = self.batch10(y)
        x = self.relu10(x + y)

        y = self.conv11(x)
        y = self.batch11(y)
        y = self.relu11(y)
        y = self.conv12(y)
        y = self.batch12(y)
        y = self.relu12(y)
        y = self.conv13(y)
        y = self.batch13(y)
        x = self.shortcut2(x)
        x = self.shortcut_batch2(x)
        x = self.relu13(x + y)

        y = self.conv14(x)
        y = self.batch14(y)
        y = self.relu14(y)
        y = self.conv15(y)
        y = self.batch15(y)
        y = self.relu15(y)
        y = self.conv16(y)
        y = self.batch16(y)
        x = self.relu16(x + y)

        y = self.conv17(x)
        y = self.batch17(y)
        y = self.relu17(y)
        y = self.conv18(y)
        y = self.batch18(y)
        y = self.relu18(y)
        y = self.conv19(y)
        y = self.batch19(y)
        x = self.relu19(x + y)

        y = self.conv20(x)
        y = self.batch20(y)
        y = self.relu20(y)
        y = self.conv21(y)
        y = self.batch21(y)
        y = self.relu21(y)
        y = self.conv22(y)
        y = self.batch22(y)
        x = self.shortcut3(x)
        x = self.shortcut_batch3(x)
        x = self.relu22(x + y)

        y = self.conv23(x)
        y = self.batch23(y)
        y = self.relu23(y)
        y = self.conv24(y)
        y = self.batch24(y)
        y = self.relu24(y)
        y = self.conv25(y)
        y = self.batch25(y)
        x = self.relu25(x + y)

        y = self.conv26(x)
        y = self.batch26(y)
        y = self.relu26(y)
        y = self.conv27(y)
        y = self.batch27(y)
        y = self.relu27(y)
        y = self.conv28(y)
        y = self.batch28(y)
        x = self.relu28(x + y)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.dequant(x)
        return x


class Squeeze_Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Squeeze_Net, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)  # 第1层
        self.batch1 = nn.BatchNorm2d(96, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第2层
        self.batch2 = nn.BatchNorm2d(16, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3_1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第3层
        self.batch3_1 = nn.BatchNorm2d(64, affine=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)  # 第3层
        self.batch3_2 = nn.BatchNorm2d(64, affine=True)
        self.relu3_2 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第4层
        self.batch4 = nn.BatchNorm2d(16, affine=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5_1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第5层
        self.batch5_1 = nn.BatchNorm2d(64, affine=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)  # 第5层
        self.batch5_2 = nn.BatchNorm2d(64, affine=True)
        self.relu5_2 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第6层
        self.batch6 = nn.BatchNorm2d(32, affine=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7_1 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第7层
        self.batch7_1 = nn.BatchNorm2d(128, affine=True)
        self.relu7_1 = nn.ReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)  # 第7层
        self.batch7_2 = nn.BatchNorm2d(128, affine=True)
        self.relu7_2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第8层
        self.batch8 = nn.BatchNorm2d(32, affine=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9_1 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第9层
        self.batch9_1 = nn.BatchNorm2d(128, affine=True)
        self.relu9_1 = nn.ReLU(inplace=True)
        self.conv9_2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)  # 第9层
        self.batch9_2 = nn.BatchNorm2d(128, affine=True)
        self.relu9_2 = nn.ReLU(inplace=True)

        self.conv10 = nn.Conv2d(in_channels=256, out_channels=48, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第10层
        self.batch10 = nn.BatchNorm2d(48, affine=True)
        self.relu10 = nn.ReLU(inplace=True)
        self.conv11_1 = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第11层
        self.batch11_1 = nn.BatchNorm2d(192, affine=True)
        self.relu11_1 = nn.ReLU(inplace=True)
        self.conv11_2 = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)  # 第11层
        self.batch11_2 = nn.BatchNorm2d(192, affine=True)
        self.relu11_2 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(in_channels=384, out_channels=48, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第12层
        self.batch12 = nn.BatchNorm2d(48, affine=True)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv13_1 = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第13层
        self.batch13_1 = nn.BatchNorm2d(192, affine=True)
        self.relu13_1 = nn.ReLU(inplace=True)
        self.conv13_2 = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)  # 第13层
        self.batch13_2 = nn.BatchNorm2d(192, affine=True)
        self.relu13_2 = nn.ReLU(inplace=True)

        self.conv14 = nn.Conv2d(in_channels=384, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第14层
        self.batch14 = nn.BatchNorm2d(64, affine=True)
        self.relu14 = nn.ReLU(inplace=True)
        self.conv15_1 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第15层
        self.batch15_1 = nn.BatchNorm2d(256, affine=True)
        self.relu15_1 = nn.ReLU(inplace=True)
        self.conv15_2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)  # 第15层
        self.batch15_2 = nn.BatchNorm2d(256, affine=True)
        self.relu15_2 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv16 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第16层
        self.batch16 = nn.BatchNorm2d(64, affine=True)
        self.relu16 = nn.ReLU(inplace=True)
        self.conv17_1 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第17层
        self.batch17_1 = nn.BatchNorm2d(256, affine=True)
        self.relu17_1 = nn.ReLU(inplace=True)
        self.conv17_2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)  # 第17层
        self.batch17_2 = nn.BatchNorm2d(256, affine=True)
        self.relu17_2 = nn.ReLU(inplace=True)

        self.conv18 = nn.Conv2d(in_channels=512, out_channels=1000, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第18层
        self.batch18 = nn.BatchNorm2d(1000, affine=True)
        self.relu18 = nn.ReLU(inplace=True)

        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.fc = nn.Linear(1000, num_classes)  # 第19层

        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating point to quantized in the quantized model
        x = self.quant(x)

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x1 = self.conv3_1(x)
        x1 = self.batch3_1(x1)
        x1 = self.relu3_1(x1)
        x2 = self.conv3_2(x)
        x2 = self.batch3_2(x2)
        x2 = self.relu3_2(x2)
        x = torch.cat((x1, x2), 1)

        y = self.conv4(x)
        y = self.batch4(y)
        y = self.relu4(y)
        x1 = self.conv5_1(y)
        x1 = self.batch5_1(x1)
        x1 = self.relu5_1(x1)
        x2 = self.conv5_2(y)
        x2 = self.batch5_2(x2)
        x2 = self.relu5_2(x2)
        y = torch.cat((x1, x2), 1)
        x = x + y

        x = self.conv6(x)
        x = self.batch6(x)
        x = self.relu6(x)
        x1 = self.conv7_1(x)
        x1 = self.batch7_1(x1)
        x1 = self.relu7_1(x1)
        x2 = self.conv7_2(x)
        x2 = self.batch7_2(x2)
        x2 = self.relu7_2(x2)
        x = torch.cat((x1, x2), 1)
        x = self.pool2(x)

        y = self.conv8(x)
        y = self.batch8(y)
        y = self.relu8(y)
        x1 = self.conv9_1(y)
        x1 = self.batch9_1(x1)
        x1 = self.relu9_1(x1)
        x2 = self.conv9_2(y)
        x2 = self.batch9_2(x2)
        x2 = self.relu9_2(x2)
        y = torch.cat((x1, x2), 1)
        x = x + y

        x = self.conv10(x)
        x = self.batch10(x)
        x = self.relu10(x)
        x1 = self.conv11_1(x)
        x1 = self.batch11_1(x1)
        x1 = self.relu11_1(x1)
        x2 = self.conv11_2(x)
        x2 = self.batch11_2(x2)
        x2 = self.relu11_2(x2)
        x = torch.cat((x1, x2), 1)

        y = self.conv12(x)
        y = self.batch12(y)
        y = self.relu12(y)
        x1 = self.conv13_1(y)
        x1 = self.batch13_1(x1)
        x1 = self.relu13_1(x1)
        x2 = self.conv13_2(y)
        x2 = self.batch13_2(x2)
        x2 = self.relu13_2(x2)
        y = torch.cat((x1, x2), 1)
        x = x + y

        x = self.conv14(x)
        x = self.batch14(x)
        x = self.relu14(x)
        x1 = self.conv15_1(x)
        x1 = self.batch15_1(x1)
        x1 = self.relu15_1(x1)
        x2 = self.conv15_2(x)
        x2 = self.batch15_2(x2)
        x2 = self.relu15_2(x2)
        x = torch.cat((x1, x2), 1)
        x = self.pool3(x)

        y = self.conv16(x)
        y = self.batch16(y)
        y = self.relu16(y)
        x1 = self.conv17_1(y)
        x1 = self.batch17_1(x1)
        x1 = self.relu17_1(x1)
        x2 = self.conv17_2(y)
        x2 = self.batch17_2(x2)
        x2 = self.relu17_2(x2)
        y = torch.cat((x1, x2), 1)
        x = x + y

        x = self.conv18(x)
        x = self.batch18(x)
        x = self.relu18(x)

        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # manually specify where tensors will be converted from quantized to floating point in the quantized model
        x = self.dequant(x)
        return x


class Mobile_Net_V1(torch.nn.Module):
    def __init__(self, num_classes):
        super(Mobile_Net_V1, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)  # 第1层
        self.batch1 = nn.BatchNorm2d(32, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=32, bias=False)  # 第2层
        self.batch2_1 = nn.BatchNorm2d(32, affine=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第3层
        self.batch2_2 = nn.BatchNorm2d(64, affine=True)
        self.relu2_2 = nn.ReLU(inplace=True)

        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1, groups=64, bias=False)  # 第4层
        self.batch3_1 = nn.BatchNorm2d(64, affine=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第5层
        self.batch3_2 = nn.BatchNorm2d(128, affine=True)
        self.relu3_2 = nn.ReLU(inplace=True)

        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=128, bias=False)  # 第6层
        self.batch4_1 = nn.BatchNorm2d(128, affine=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第7层
        self.batch4_2 = nn.BatchNorm2d(128, affine=True)
        self.relu4_2 = nn.ReLU(inplace=True)

        self.conv5_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=1, groups=128, bias=False)  # 第8层
        self.batch5_1 = nn.BatchNorm2d(128, affine=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第9层
        self.batch5_2 = nn.BatchNorm2d(256, affine=True)
        self.relu5_2 = nn.ReLU(inplace=True)

        self.conv6_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=256, bias=False)  # 第10层
        self.batch6_1 = nn.BatchNorm2d(256, affine=True)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.conv6_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第11层
        self.batch6_2 = nn.BatchNorm2d(256, affine=True)
        self.relu6_2 = nn.ReLU(inplace=True)

        self.conv7_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=1, groups=256, bias=False)  # 第12层
        self.batch7_1 = nn.BatchNorm2d(256, affine=True)
        self.relu7_1 = nn.ReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第13层
        self.batch7_2 = nn.BatchNorm2d(512, affine=True)
        self.relu7_2 = nn.ReLU(inplace=True)

        self.conv8_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=512, bias=False)  # 第14层
        self.batch8_1 = nn.BatchNorm2d(512, affine=True)
        self.relu8_1 = nn.ReLU(inplace=True)
        self.conv8_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第15层
        self.batch8_2 = nn.BatchNorm2d(512, affine=True)
        self.relu8_2 = nn.ReLU(inplace=True)

        self.conv9_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=512, bias=False)  # 第16层
        self.batch9_1 = nn.BatchNorm2d(512, affine=True)
        self.relu9_1 = nn.ReLU(inplace=True)
        self.conv9_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第17层
        self.batch9_2 = nn.BatchNorm2d(512, affine=True)
        self.relu9_2 = nn.ReLU(inplace=True)

        self.conv10_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=512, bias=False)  # 第18层
        self.batch10_1 = nn.BatchNorm2d(512, affine=True)
        self.relu10_1 = nn.ReLU(inplace=True)
        self.conv10_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第19层
        self.batch10_2 = nn.BatchNorm2d(512, affine=True)
        self.relu10_2 = nn.ReLU(inplace=True)

        self.conv11_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=512, bias=False)  # 第20层
        self.batch11_1 = nn.BatchNorm2d(512, affine=True)
        self.relu11_1 = nn.ReLU(inplace=True)
        self.conv11_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第21层
        self.batch11_2 = nn.BatchNorm2d(512, affine=True)
        self.relu11_2 = nn.ReLU(inplace=True)

        self.conv12_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=512, bias=False)  # 第22层
        self.batch12_1 = nn.BatchNorm2d(512, affine=True)
        self.relu12_1 = nn.ReLU(inplace=True)
        self.conv12_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第23层
        self.batch12_2 = nn.BatchNorm2d(512, affine=True)
        self.relu12_2 = nn.ReLU(inplace=True)

        self.conv13_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=1, groups=512, bias=False)  # 第24层
        self.batch13_1 = nn.BatchNorm2d(512, affine=True)
        self.relu13_1 = nn.ReLU(inplace=True)
        self.conv13_2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第25层
        self.batch13_2 = nn.BatchNorm2d(1024, affine=True)
        self.relu13_2 = nn.ReLU(inplace=True)

        self.conv14 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1, 1), stride=(2, 2), padding=0, bias=False)  # 第26层
        self.batch14 = nn.BatchNorm2d(1024, affine=True)
        self.relu14 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(1024, 1000)  # 第27层
        self.relu15 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1000, num_classes)  # 第28层

        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating point to quantized in the quantized model
        x = self.quant(x)

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)

        x = self.conv2_1(x)
        x = self.batch2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.batch2_2(x)
        x = self.relu2_2(x)

        x = self.conv3_1(x)
        x = self.batch3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.batch3_2(x)
        x = self.relu3_2(x)

        x = self.conv4_1(x)
        x = self.batch4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.batch4_2(x)
        x = self.relu4_2(x)

        x = self.conv5_1(x)
        x = self.batch5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.batch5_2(x)
        x = self.relu5_2(x)

        x = self.conv6_1(x)
        x = self.batch6_1(x)
        x = self.relu6_1(x)
        x = self.conv6_2(x)
        x = self.batch6_2(x)
        x = self.relu6_2(x)

        x = self.conv7_1(x)
        x = self.batch7_1(x)
        x = self.relu7_1(x)
        x = self.conv7_2(x)
        x = self.batch7_2(x)
        x = self.relu7_2(x)

        x = self.conv8_1(x)
        x = self.batch8_1(x)
        x = self.relu8_1(x)
        x = self.conv8_2(x)
        x = self.batch8_2(x)
        x = self.relu8_2(x)

        x = self.conv9_1(x)
        x = self.batch9_1(x)
        x = self.relu9_1(x)
        x = self.conv9_2(x)
        x = self.batch9_2(x)
        x = self.relu9_2(x)

        x = self.conv10_1(x)
        x = self.batch10_1(x)
        x = self.relu10_1(x)
        x = self.conv10_2(x)
        x = self.batch10_2(x)
        x = self.relu10_2(x)

        x = self.conv11_1(x)
        x = self.batch11_1(x)
        x = self.relu11_1(x)
        x = self.conv11_2(x)
        x = self.batch11_2(x)
        x = self.relu11_2(x)

        x = self.conv12_1(x)
        x = self.batch12_1(x)
        x = self.relu12_1(x)
        x = self.conv12_2(x)
        x = self.batch12_2(x)
        x = self.relu12_2(x)

        x = self.conv13_1(x)
        x = self.batch13_1(x)
        x = self.relu13_1(x)
        x = self.conv13_2(x)
        x = self.batch13_2(x)
        x = self.relu13_2(x)

        x = self.conv14(x)
        x = self.batch14(x)
        x = self.relu14(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu15(x)
        x = self.fc2(x)

        # manually specify where tensors will be converted from quantized to floating point in the quantized model
        x = self.dequant(x)
        return x


class Mobile_Net_V2(torch.nn.Module):
    def __init__(self, num_classes):
        super(Mobile_Net_V2, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)  # 第1层
        self.batch1 = nn.BatchNorm2d(32, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第2层
        self.batch2_1 = nn.BatchNorm2d(32, affine=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=32, bias=False)  # 第3层
        self.batch2_2 = nn.BatchNorm2d(32, affine=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.conv2_3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第4层
        self.batch2_3 = nn.BatchNorm2d(16, affine=True)

        self.conv3_1 = nn.Conv2d(in_channels=16, out_channels=96, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第5层
        self.batch3_1 = nn.BatchNorm2d(96, affine=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(2, 2), padding=1, groups=96, bias=False)  # 第6层
        self.batch3_2 = nn.BatchNorm2d(96, affine=True)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=96, out_channels=24, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第7层
        self.batch3_3 = nn.BatchNorm2d(24, affine=True)

        self.conv4_1 = nn.Conv2d(in_channels=24, out_channels=144, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第8层
        self.batch4_1 = nn.BatchNorm2d(144, affine=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=144, out_channels=144, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=144, bias=False)  # 第9层
        self.batch4_2 = nn.BatchNorm2d(144, affine=True)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=144, out_channels=24, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第10层
        self.batch4_3 = nn.BatchNorm2d(24, affine=True)

        self.conv5_1 = nn.Conv2d(in_channels=24, out_channels=144, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第11层
        self.batch5_1 = nn.BatchNorm2d(144, affine=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=144, out_channels=144, kernel_size=(3, 3), stride=(2, 2), padding=1, groups=144, bias=False)  # 第12层
        self.batch5_2 = nn.BatchNorm2d(144, affine=True)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=144, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第13层
        self.batch5_3 = nn.BatchNorm2d(32, affine=True)

        self.conv6_1 = nn.Conv2d(in_channels=32, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第14层
        self.batch6_1 = nn.BatchNorm2d(192, affine=True)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.conv6_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=192, bias=False)  # 第15层
        self.batch6_2 = nn.BatchNorm2d(192, affine=True)
        self.relu6_2 = nn.ReLU(inplace=True)
        self.conv6_3 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第16层
        self.batch6_3 = nn.BatchNorm2d(32, affine=True)

        self.conv7_1 = nn.Conv2d(in_channels=32, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第17层
        self.batch7_1 = nn.BatchNorm2d(192, affine=True)
        self.relu7_1 = nn.ReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=192, bias=False)  # 第18层
        self.batch7_2 = nn.BatchNorm2d(192, affine=True)
        self.relu7_2 = nn.ReLU(inplace=True)
        self.conv7_3 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第19层
        self.batch7_3 = nn.BatchNorm2d(32, affine=True)

        self.conv8_1 = nn.Conv2d(in_channels=32, out_channels=192, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第20层
        self.batch8_1 = nn.BatchNorm2d(192, affine=True)
        self.relu8_1 = nn.ReLU(inplace=True)
        self.conv8_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(2, 2), padding=1, groups=192, bias=False)  # 第21层
        self.batch8_2 = nn.BatchNorm2d(192, affine=True)
        self.relu8_2 = nn.ReLU(inplace=True)
        self.conv8_3 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第22层
        self.batch8_3 = nn.BatchNorm2d(64, affine=True)

        self.conv9_1 = nn.Conv2d(in_channels=64, out_channels=384, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第23层
        self.batch9_1 = nn.BatchNorm2d(384, affine=True)
        self.relu9_1 = nn.ReLU(inplace=True)
        self.conv9_2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=384, bias=False)  # 第24层
        self.batch9_2 = nn.BatchNorm2d(384, affine=True)
        self.relu9_2 = nn.ReLU(inplace=True)
        self.conv9_3 = nn.Conv2d(in_channels=384, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第25层
        self.batch9_3 = nn.BatchNorm2d(64, affine=True)

        self.conv10_1 = nn.Conv2d(in_channels=64, out_channels=384, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第26层
        self.batch10_1 = nn.BatchNorm2d(384, affine=True)
        self.relu10_1 = nn.ReLU(inplace=True)
        self.conv10_2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=384, bias=False)  # 第27层
        self.batch10_2 = nn.BatchNorm2d(384, affine=True)
        self.relu10_2 = nn.ReLU(inplace=True)
        self.conv10_3 = nn.Conv2d(in_channels=384, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第28层
        self.batch10_3 = nn.BatchNorm2d(64, affine=True)

        self.conv11_1 = nn.Conv2d(in_channels=64, out_channels=384, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第29层
        self.batch11_1 = nn.BatchNorm2d(384, affine=True)
        self.relu11_1 = nn.ReLU(inplace=True)
        self.conv11_2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=384, bias=False)  # 第30层
        self.batch11_2 = nn.BatchNorm2d(384, affine=True)
        self.relu11_2 = nn.ReLU(inplace=True)
        self.conv11_3 = nn.Conv2d(in_channels=384, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第31层
        self.batch11_3 = nn.BatchNorm2d(64, affine=True)

        self.conv12_1 = nn.Conv2d(in_channels=64, out_channels=384, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第32层
        self.batch12_1 = nn.BatchNorm2d(384, affine=True)
        self.relu12_1 = nn.ReLU(inplace=True)
        self.conv12_2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=384, bias=False)  # 第33层
        self.batch12_2 = nn.BatchNorm2d(384, affine=True)
        self.relu12_2 = nn.ReLU(inplace=True)
        self.conv12_3 = nn.Conv2d(in_channels=384, out_channels=96, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第34层
        self.batch12_3 = nn.BatchNorm2d(96, affine=True)

        self.conv13_1 = nn.Conv2d(in_channels=96, out_channels=576, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第35层
        self.batch13_1 = nn.BatchNorm2d(576, affine=True)
        self.relu13_1 = nn.ReLU(inplace=True)
        self.conv13_2 = nn.Conv2d(in_channels=576, out_channels=576, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=576, bias=False)  # 第36层
        self.batch13_2 = nn.BatchNorm2d(576, affine=True)
        self.relu13_2 = nn.ReLU(inplace=True)
        self.conv13_3 = nn.Conv2d(in_channels=576, out_channels=96, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第37层
        self.batch13_3 = nn.BatchNorm2d(96, affine=True)

        self.conv14_1 = nn.Conv2d(in_channels=96, out_channels=576, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第38层
        self.batch14_1 = nn.BatchNorm2d(576, affine=True)
        self.relu14_1 = nn.ReLU(inplace=True)
        self.conv14_2 = nn.Conv2d(in_channels=576, out_channels=576, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=576, bias=False)  # 第39层
        self.batch14_2 = nn.BatchNorm2d(576, affine=True)
        self.relu14_2 = nn.ReLU(inplace=True)
        self.conv14_3 = nn.Conv2d(in_channels=576, out_channels=96, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第40层
        self.batch14_3 = nn.BatchNorm2d(96, affine=True)

        self.conv15_1 = nn.Conv2d(in_channels=96, out_channels=576, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第41层
        self.batch15_1 = nn.BatchNorm2d(576, affine=True)
        self.relu15_1 = nn.ReLU(inplace=True)
        self.conv15_2 = nn.Conv2d(in_channels=576, out_channels=576, kernel_size=(3, 3), stride=(2, 2), padding=1, groups=576, bias=False)  # 第42层
        self.batch15_2 = nn.BatchNorm2d(576, affine=True)
        self.relu15_2 = nn.ReLU(inplace=True)
        self.conv15_3 = nn.Conv2d(in_channels=576, out_channels=160, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第43层
        self.batch15_3 = nn.BatchNorm2d(160, affine=True)

        self.conv16_1 = nn.Conv2d(in_channels=160, out_channels=960, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第44层
        self.batch16_1 = nn.BatchNorm2d(960, affine=True)
        self.relu16_1 = nn.ReLU(inplace=True)
        self.conv16_2 = nn.Conv2d(in_channels=960, out_channels=960, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=960, bias=False)  # 第45层
        self.batch16_2 = nn.BatchNorm2d(960, affine=True)
        self.relu16_2 = nn.ReLU(inplace=True)
        self.conv16_3 = nn.Conv2d(in_channels=960, out_channels=160, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第46层
        self.batch16_3 = nn.BatchNorm2d(160, affine=True)

        self.conv17_1 = nn.Conv2d(in_channels=160, out_channels=960, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第47层
        self.batch17_1 = nn.BatchNorm2d(960, affine=True)
        self.relu17_1 = nn.ReLU(inplace=True)
        self.conv17_2 = nn.Conv2d(in_channels=960, out_channels=960, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=960, bias=False)  # 第48层
        self.batch17_2 = nn.BatchNorm2d(960, affine=True)
        self.relu17_2 = nn.ReLU(inplace=True)
        self.conv17_3 = nn.Conv2d(in_channels=960, out_channels=160, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第49层
        self.batch17_3 = nn.BatchNorm2d(160, affine=True)

        self.conv18_1 = nn.Conv2d(in_channels=160, out_channels=960, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第50层
        self.batch18_1 = nn.BatchNorm2d(960, affine=True)
        self.relu18_1 = nn.ReLU(inplace=True)
        self.conv18_2 = nn.Conv2d(in_channels=960, out_channels=960, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=960, bias=False)  # 第51层
        self.batch18_2 = nn.BatchNorm2d(960, affine=True)
        self.relu18_2 = nn.ReLU(inplace=True)
        self.conv18_3 = nn.Conv2d(in_channels=960, out_channels=320, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第52层
        self.batch18_3 = nn.BatchNorm2d(320, affine=True)

        self.conv19 = nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)  # 第53层
        self.batch19 = nn.BatchNorm2d(1280, affine=True)
        self.relu19 = nn.ReLU(inplace=True)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.shortcut1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.shortcut_batch1 = nn.BatchNorm2d(16, affine=True)
        self.shortcut2 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.shortcut_batch2 = nn.BatchNorm2d(96, affine=True)
        self.shortcut3 = nn.Conv2d(in_channels=160, out_channels=320, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.shortcut_batch3 = nn.BatchNorm2d(320, affine=True)

        self.fc = nn.Linear(1280, num_classes)  # 第54层

        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating point to quantized in the quantized model
        x = self.quant(x)

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)

        y = self.conv2_1(x)
        y = self.batch2_1(y)
        y = self.relu2_1(y)
        y = self.conv2_2(y)
        y = self.batch2_2(y)
        y = self.relu2_2(y)
        y = self.conv2_3(y)
        y = self.batch2_3(y)
        x = self.shortcut1(x)
        x = self.shortcut_batch1(x)
        x = x + y

        x = self.conv3_1(x)
        x = self.batch3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.batch3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.batch3_3(x)

        y = self.conv4_1(x)
        y = self.batch4_1(y)
        y = self.relu4_1(y)
        y = self.conv4_2(y)
        y = self.batch4_2(y)
        y = self.relu4_2(y)
        y = self.conv4_3(y)
        y = self.batch4_3(y)
        x = x + y

        x = self.conv5_1(x)
        x = self.batch5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.batch5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.batch5_3(x)

        y = self.conv6_1(x)
        y = self.batch6_1(y)
        y = self.relu6_1(y)
        y = self.conv6_2(y)
        y = self.batch6_2(y)
        y = self.relu6_2(y)
        y = self.conv6_3(y)
        y = self.batch6_3(y)
        x = x + y

        y = self.conv7_1(x)
        y = self.batch7_1(y)
        y = self.relu7_1(y)
        y = self.conv7_2(y)
        y = self.batch7_2(y)
        y = self.relu7_2(y)
        y = self.conv7_3(y)
        y = self.batch7_3(y)
        x = x + y

        x = self.conv8_1(x)
        x = self.batch8_1(x)
        x = self.relu8_1(x)
        x = self.conv8_2(x)
        x = self.batch8_2(x)
        x = self.relu8_2(x)
        x = self.conv8_3(x)
        x = self.batch8_3(x)

        y = self.conv9_1(x)
        y = self.batch9_1(y)
        y = self.relu9_1(y)
        y = self.conv9_2(y)
        y = self.batch9_2(y)
        y = self.relu9_2(y)
        y = self.conv9_3(y)
        y = self.batch9_3(y)
        x = x + y

        y = self.conv10_1(x)
        y = self.batch10_1(y)
        y = self.relu10_1(y)
        y = self.conv10_2(y)
        y = self.batch10_2(y)
        y = self.relu10_2(y)
        y = self.conv10_3(y)
        y = self.batch10_3(y)
        x = x + y

        y = self.conv11_1(x)
        y = self.batch11_1(y)
        y = self.relu11_1(y)
        y = self.conv11_2(y)
        y = self.batch11_2(y)
        y = self.relu11_2(y)
        y = self.conv11_3(y)
        y = self.batch11_3(y)
        x = x + y

        y = self.conv12_1(x)
        y = self.batch12_1(y)
        y = self.relu12_1(y)
        y = self.conv12_2(y)
        y = self.batch12_2(y)
        y = self.relu12_2(y)
        y = self.conv12_3(y)
        y = self.batch12_3(y)
        x = self.shortcut2(x)
        x = self.shortcut_batch2(x)
        x = x + y

        y = self.conv13_1(x)
        y = self.batch13_1(y)
        y = self.relu13_1(y)
        y = self.conv13_2(y)
        y = self.batch13_2(y)
        y = self.relu13_2(y)
        y = self.conv13_3(y)
        y = self.batch13_3(y)
        x = x + y

        y = self.conv14_1(x)
        y = self.batch14_1(y)
        y = self.relu14_1(y)
        y = self.conv14_2(y)
        y = self.batch14_2(y)
        y = self.relu14_2(y)
        y = self.conv14_3(y)
        y = self.batch14_3(y)
        x = x + y

        x = self.conv15_1(x)
        x = self.batch15_1(x)
        x = self.relu15_1(x)
        x = self.conv15_2(x)
        x = self.batch15_2(x)
        x = self.relu15_2(x)
        x = self.conv15_3(x)
        x = self.batch15_3(x)

        y = self.conv16_1(x)
        y = self.batch16_1(y)
        y = self.relu16_1(y)
        y = self.conv16_2(y)
        y = self.batch16_2(y)
        y = self.relu16_2(y)
        y = self.conv16_3(y)
        y = self.batch16_3(y)
        x = x + y

        y = self.conv17_1(x)
        y = self.batch17_1(y)
        y = self.relu17_1(y)
        y = self.conv17_2(y)
        y = self.batch17_2(y)
        y = self.relu17_2(y)
        y = self.conv17_3(y)
        y = self.batch17_3(y)
        x = x + y

        y = self.conv18_1(x)
        y = self.batch18_1(y)
        y = self.relu18_1(y)
        y = self.conv18_2(y)
        y = self.batch18_2(y)
        y = self.relu18_2(y)
        y = self.conv18_3(y)
        y = self.batch18_3(y)
        x = self.shortcut3(x)
        x = self.shortcut_batch3(x)
        x = x + y

        x = self.conv19(x)
        x = self.batch19(x)
        x = self.relu19(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # manually specify where tensors will be converted from quantized to floating point in the quantized model
        x = self.dequant(x)
        return x
