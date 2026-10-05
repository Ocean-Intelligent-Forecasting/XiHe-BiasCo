from arch import ClimaX
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

model = ClimaX().cuda().half()
optimizer = optim.SGD(model.parameters(),lr=0.0001)
x = np.load('mra5_20180101.npy')
x = torch.tensor(x,device='cuda').half()
print(x.shape)
y = torch.clone(x)
y = y[:,2:,:,:].cuda()
y_hat = model(x,y,None,None,None,None,None,None)
loss = torch.nn.MSELoss()
optimizer.zero_grad()
l = loss(y,y_hat)
l.backward()
for name,param in model.named_parameters():
    if param.grad is None:
        print(name)
optimizer.step()





