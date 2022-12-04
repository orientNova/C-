# -*- coding: utf-8 -*-
import torchsketch
import torchsketch.data.dataloaders as dataloaders
import torch.optim as optim
import torchsketch.data.datasets as datasets
import torchsketch.networks.cnn as cnns

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# 库里的cnns.sketchnet()  网络里有几处需要更改，大家直接把我发的sketchnet.py文件放在工作空间下，然后import
from sketchnet import SketchaNet
import torch.nn.functional
import torch.nn as nn

# 超参数设置
EPOCH = 1000   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 64      #批处理尺寸(batch_size)
LR = 0.0001        #学习率

transform_data = transforms.Compose([
        transforms.Resize((255,255)),
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5,), (0.5,)), # 归一化
                             ])

# 程序下载比较慢 建议直接从该网址下载png格式的数据集"http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch"  页面下载sketches_png.zip
# 解压到当前工作空间下， sketches_png/png/以类命名的文件夹+一个filelist.txt  ， 我把它随便分成了训练集和测试集（大家可自己细分），把trainlist.txt和testlist.txt放在png文件夹里

train_dataset = dataloaders.TUBerlin('sketches_png/png', "sketches_png\png\\trainlist.txt", transform_data)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
test_dataset = dataloaders.TUBerlin('sketches_png/png', "sketches_png\png\\testlist.txt", transform_data)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 网络模型实例化、损失函数、优化器
net = SketchaNet()
SketchNet = net.to(device)

# 定义损失函数和优化方式   多分类问题使用交叉熵损失函数，采用mini-batch momentum-SGD优化器，并采用L2正则化（权重衰减）
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题,此标准将LogSoftMax和NLLLoss集成到一个类中。
optimizer = optim.SGD(SketchNet.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）


def sketchanet_train():
    print("Start Training -- SketchaNet !")
    for epoch in range(pre_epoch, EPOCH):  # 从先前次数开始训练
        print('\nEpoch: %d' % (epoch + 1))  # 输出当前次数
        SketchNet.train()  # 这两个函数只要适用于Dropout与BatchNormalization的网络，会影响到训练过程中这两者的参数
        # 运用net.train()时，训练时每个min - batch时都会根据情况进行上述两个参数的相应调整，所有BatchNormalization的训练和测试时的操作不同。
        sum_loss = 0.0  # 损失数量
        correct = 0.0  # 准确数量
        total = 0.0  # 总共数量
        for i, data in enumerate(train_dataloader,0):  # 训练集合enumerate(sequence, [start=0])用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            # 准备数据  i是顺序序号 data是遍历的数据元素，包括图像tensor数据和其标签
            length = len(train_dataloader)  # 训练数量
            inputs, labels = data
            #  inputs是当前输入的图像，label是当前图像的标签，这个data中每一个sample对应一个label
            inputs, labels = inputs.to(device), labels.to(device)
            # labels = torch.nn.functional.one_hot(labels)          # 将每个样本的类别数转为one-hot编码，用于计算loss
            optimizer.zero_grad()  # 清空所有被优化过的Variable的梯度.

            # forward + backward
            outputs = SketchNet(inputs)  # 得到训练后的一个输出
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # 进行单次优化 (参数更新).

            _, predicted = torch.max(outputs.data, 1)  # 返回输入张量所有元素的最大值。 将dim维设定为1，指每行的最大值。output输出为[1,num_class]
            # 这里采用torch.max。torch.max()的第一个输入是tensor格式，所以用outputs.data而不是outputs作为输入；第二个参数1是代表dim的意思，也就是取每一行的最大值，其实就是我们常见的取概率最大的那个index；第三个参数loss也是torch.autograd.Variable格式。
            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()

            correct += (predicted == labels).sum()
            total += labels.size(0)
            # predicted.eq(labels.data)指将 predicted这个tensor中的数字与labels.data中的数字一一比较，相同为1，不同为0
            # 然后.cpu()指移动到cpu计算，.sum()将结果相加，作用是得到预测正确的图片个数
            if i % 100 == 0 :
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                    % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

        if epoch % 100 == 0 and epoch != 0:
            # 每训练完一个epoch测试一下准确率
            print("Waiting Test!")
            with torch.no_grad():  # 没有求导
                correct = 0
                total = 0
                for data in test_dataloader:
                    SketchNet.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = SketchNet(images)
                    # 取得分最高的那个类 (outputs.data的索引号)
                    _, predicted = torch.max(outputs.data, 1)

                    # print(predicted)
                    # print(".............")

                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            # 将每次测试结果实时写入acc.txt文件中
            if epoch % 10 == 0:
                torch.save(SketchNet, "SketchNet_model_{}.pth".format(epoch + 1))
                print("模型已保存：tu_berlin_test_model_{}.pth".format(epoch + 1))

    print("Training Finished, TotalEPOCH=%d" % EPOCH)


if __name__ == "__main__":
    sketchanet_train()