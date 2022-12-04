import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #图像预处理 
    transform = transforms.Compose(
        [transforms.ToTensor(),  
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    #batch_size每个批次数据数目 shuffle=True打乱图片 num_workers=0载入数据的工作线程数ubuntu下可以设置 win下只可为0
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                             shuffle=False, num_workers=0)

    #生成迭代器                                    
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)
    val_image, val_label = val_image.to(device), val_label.to(device)
    
    #标签 元组类型不可改变
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    

    net = LeNet()
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001) #优化器使用Adm net.parameters() 训练所有参数 lr学习率
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = 6


    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0  #损失
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()#历史梯度清零
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step() #参数更新

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:    # print every 500 mini-batches
                with torch.no_grad(): #上下文管理器 接下来的语句不计算每个节点的梯度
                    outputs = net(val_image) # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1] #outputs是网络输出 0维度是batch 1维度节点 [1] index
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0) #item()：tensor转数值

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path) #保存模型权重


if __name__ == '__main__':
    main()


# pytorch tensor [batch, channel, height, width]
'''
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()     # 图像转换为npo格式
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(val_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(val_image))
    # print labels
    print(' '.join(f'{classes[val_label[j]]:5s}' for j in range(7)))


'''