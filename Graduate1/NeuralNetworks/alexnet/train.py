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
from tqdm import tqdm                       # 进度条模块
from pathlib import Path
from alexnet import AlexNet

info_str = '[AlexNet INFO] '

# Get the directory where the running file is located
FILE = Path(__file__).resolve() # absolute path # 获取文件绝对路径           
ROOT_AP = FILE.parents[0]  # root directory # 获取文件所在目录
ROOT_RP = Path(os.path.relpath(ROOT_AP, Path.cwd()))  # relative path #获取与工作目录的相对路径
# print('ROOT_AP',ROOT_AP)
# print('ROOT_RP',ROOT_RP)
# print('ROOT.parent',ROOT.parent)
# root1 = os.getcwd() # 获得运行环境目录
# Path.cwd() # 返回当前工作目录，实际为WindowsPath 对象，若只要字符串，使用str(Path.cwd())
# Path(__file__) # 返回当前当前文件路径，实际为WindowsPath 对象，若只要字符串，使用str(Path(__file__))
# Path(__file__).resolve() # 返回当前当前文件路径
#==============================================================================================#
# Set Device
#==============================================================================================#
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print("{} Using {} device.".format(info_str,device))

#==============================================================================================#
# Create NN & Set Hyperparameters
#==============================================================================================#

# Hyperparameters -----------------------#
NUM_CLASSES = 250        # 分类类别
LR = 0.00001             # 学习率
BATCH_SIZE = 64          # 批量大小
EPOCHS = 8               # 训练轮次
nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
# nw = 0 # windows下设置为0
print('{} Using {} dataloader workers every process'.format(info_str,nw))

# Create NN -----------------------#
net = AlexNet(num_classes=NUM_CLASSES).to(device)

# 如果预训练文件存在就载入
PTH_PATH = "./alexnet.pth"
if (os.path.exists(PTH_PATH)): 
    # 载入预训练参数
    pretrained_dict = torch.load("./alexnet.pth")
    model_dict = net.state_dict()
    # 重新制作预训练的权重，主要是减去参数不匹配的层
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
    model_dict.update(pretrained_dict) # 更新权重
    net.load_state_dict(model_dict)

#==============================================================================================#
# Load Data
#==============================================================================================#
DATA_PATH = os.path.join(str(ROOT_AP.parent.parent), "dataset/tu_berlin")
assert os.path.exists(DATA_PATH), " {} path does not exist.".format(DATA_PATH)

# Raw Date Process
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(227),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(227),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

train_dataset = datasets.ImageFolder(root=os.path.join(DATA_PATH, "train"), 
                                    transform = data_transform["train"])
validate_dataset = datasets.ImageFolder(root=os.path.join(DATA_PATH, "val"), 
                                        transform = data_transform["val"])
train_loader = torch.utils.data.DataLoader( train_dataset,batch_size=BATCH_SIZE, 
                                            shuffle=True,num_workers=nw)
validate_loader = torch.utils.data.DataLoader( validate_dataset,batch_size=BATCH_SIZE, 
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
optimizer = optim.Adam(net.parameters(), lr=LR)  


#==============================================================================================#
# Train
#==============================================================================================#
def trian():
    best_acc = 0.67
    save_path = './alexnet.pth'
    train_steps = len(train_loader)
    for epoch in range(EPOCHS):
       # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data   # 获取数据
            outputs = net(images.to(device))
            loss = criterion(outputs, labels.to(device))
            optimizer.zero_grad()   # 清除历史梯度

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            train_bar.desc = "{} train epoch[{}/{}] loss:{:.3f}".format(info_str, epoch + 1,EPOCHS,loss)

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

                val_bar.desc = "{} valid epoch[{}/{}]".format(info_str,epoch + 1,EPOCHS)

        val_accurate = acc / val_num
        print(info_str,'[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
            (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print('{} Finished Training'.format(info_str))
    print('=='*100)
    
#==============================================================================================#
# Main
#==============================================================================================#   
if __name__ == '__main__':
    trian()