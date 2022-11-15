import torch
import torch.nn as nn
from torchvision.transforms import transforms
from model import PMD_model
from matplotlib.colors import LinearSegmentedColormap

clist = ['darkblue', 'blue', 'limegreen', 'orange', 'yellow']
mycmp = LinearSegmentedColormap.from_list('chaos', clist)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PMD_model()
model.load_state_dict(torch.load('./DLPMD_epoch_1500.pth'))
model.eval()
model = model.to(device)
loss = nn.MSELoss()
loss = loss.to(device)

SUT_t1 = []
pred_sut_t1 = []



def predict(input):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    input = transform(input)
    input = input.unsqueeze(0)
    with torch.no_grad():
        input = input.to(device)
        output = model(input)
        output = torch.squeeze(output,dim=0)

        return output.cpu().numpy()


