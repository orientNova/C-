#==============================================================================================#
# Imoport
#==============================================================================================#
import torch
import torch.nn as nn                       # 神经网络
import torch.optim as optim                 # 优化器
import torch.nn.functional as F             # 无学习参数的函数 如relu tanh
from torch.utils.data import DataLoader     # 数据加载器
import torchvision.datasets as datasets     # 数据集
import torchvision.transforms as transforms # 处理数据集
import os
import sys
import json

from model import sketchanet
from resnet34 import resnet34
from tqdm import tqdm
info_str = '[RUNNING INFO] '
#==============================================================================================#
# Set Device
#==============================================================================================#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("{} Using {} device.".format(info_str,device))

#==============================================================================================#
# Create NN & Set Hyperparameters
#==============================================================================================#
num_classes = 250
lr = 0.0001              # 学习率
batch_size = 64         # 批量大小
epochs = 8               # 训练轮次
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('{} Using {} dataloader workers every process'.format(info_str,nw))
#net = sketchanet(num_classes=num_classes).to(device)

# 还是KaimingHe NB
net = resnet34()
model_weight_path = "./resnet34-pre.pth"
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
# change fc layer structure
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 250)
net.to(device)



#==============================================================================================#
# Load Data
#==============================================================================================#
data_path = os.path.join( "../dataset", "tuberlin")
assert os.path.exists(data_path), " {} path does not exist.".format(data_path)

# Raw Date Process
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
                                   
# train_dataset = datasets.MNIST(root='../dataset/tuberlin/train', train=True, transform=transforms.ToTensor(), download=True)
# train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)# shuffle是否打乱数据
# test_dataset = datasets.MNIST(root='../dataset/tuberlin/test', train=False,transform=transforms.ToTensor(), download=True)
# test_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True) # 返回(data, targets)
train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), 
                                    transform = data_transform["train"])
validate_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"), 
                                        transform = data_transform["val"])
train_loader = torch.utils.data.DataLoader( train_dataset,batch_size=batch_size, 
                                            shuffle=True,num_workers=nw)
validate_loader = torch.utils.data.DataLoader( validate_dataset,batch_size=batch_size, 
                                            shuffle=True,num_workers=nw)                                         

train_num = len(train_dataset)
val_num = len(validate_dataset)
print("{} Using {} images for training, {} images for validation.".format(info_str,train_num, val_num))

data_list = train_dataset.class_to_idx # 提取class:idx字典 之后做键值反转
cla_dict = dict((val,key) for key,val in data_list.items()) #dict.items() 返回可遍历的(键, 值) 元组数组
# write dict into json file
json_str = json.dumps(cla_dict, indent=1) #indent缩进 让文件竖着排列
with open('class_indices.json','w') as json_file:
    json_file.write(json_str)

#==============================================================================================#
# Loss and Optimizer
#==============================================================================================#
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()                
optimizer = optim.Adam(net.parameters(), lr=lr)  

#==============================================================================================#
# Train
#==============================================================================================#
def trian():
    best_acc = 0.0
    save_path = './torchsketch.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
       # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data   # 获取数据
            logits = net(images.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()   # 清除历史梯度

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            train_bar.desc = "{} train epoch[{}/{}] loss:{:.3f}".format(info_str, epoch + 1,epochs,loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                outputs = net(val_images)
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "{} valid epoch[{}/{}]".format(info_str,epoch + 1,epochs)

        val_accurate = acc / val_num
        print(info_str,'[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
            (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print('{} Finished Training'.format(info_str))
    print('='*100)

if __name__ == '__main__':
    trian()