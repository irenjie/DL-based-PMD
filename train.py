import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model import PMD_model
from SurfaceDataset import loadDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = loadDataset()
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=False, num_workers=0)
test_data = loadDataset(is_test=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0)

model = PMD_model()
model = model.to(device)
epoch = 5000
lr = 0.0001
loss = nn.MSELoss()
loss = loss.to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
for i in range(epoch):
    model.train()
    for data in train_loader:
        refer_solpe, SUT = data
        refer_solpe = refer_solpe.float()
        SUT = SUT.float()
        refer_solpe = refer_solpe.to(device)
        SUT = SUT.to(device)
        pred_sut = model(refer_solpe)
        result_loss = loss(pred_sut, SUT)
        optim.zero_grad()
        result_loss.backward()
        optim.step()

    # 测试数据集检验训练效果
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            refer_solpe, SUT = data
            refer_solpe = refer_solpe.float()
            SUT = SUT.float()
            refer_solpe = refer_solpe.to(device)
            SUT = SUT.to(device)
            pred_sut = model(refer_solpe)
            result_loss = loss(pred_sut, SUT)
            total_test_loss += result_loss
    print("epoch {}, total test loss: {}".format(i, total_test_loss))
    if (i + 1) % 100 == 0 or (i + 1) == epoch:
        torch.save(model.state_dict(), "../model_epoch_{}.pth".format(i+1))
