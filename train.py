import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import dataset
import CNNmodel

data1 = dataset.DataCry()
data2 = dataset.DataNoise()
data = data2+data1
train_size = int(len(data) * 0.7)
test_size = len(data) - train_size
train_data, test_data = random_split(data, [train_size, test_size])
train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=True, num_workers=0)

model = CNNmodel.CNN()
model.train()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
epoch_size = 20
best_acc=0

writer = SummaryWriter("logs")

def test_acc(test_iter, model):
    cnt = 0
    l = 0
    n = test_size
    for X, t in test_iter:
        y = model.forward(X)
        l += criterion(y, t)
        if(abs(y-t)<0.5):
            cnt+=1
    return cnt / n, l.item() / n


for epoch in range(epoch_size):
    train_cnt = 0
    train_loss = 0
    n = train_size
    for X, t in train_dataloader:
        y = model.forward(X)
        l = criterion(y, t)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if abs((y-t))<0.5 :
            train_cnt+=1
        train_loss+=l.item()
    print('train_loss:',train_loss/n,'train_acc:',train_cnt/n)
    test_cnt,test_loss=test_acc(test_dataloader,model)
    print('test_loss:',test_loss,'test_acc:',test_cnt)
    print('\n')
    writer.add_scalar("test_loss",test_loss,epoch+1)
    if test_cnt>best_acc:
        torch.save(model,'CNN.ckpt')