import torch
import torch.nn as nn
import torchvision.transforms.functional as fn


class PMD_model(nn.Module):
    def __init__(self, width=None):
        super(PMD_model, self).__init__()
        if width is None:
            width = [32, 128, 256, 1024]
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=width[0], kernel_size=7, stride=1, padding=3, bias=False,
                               padding_mode="replicate")
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(width[0], width[1], 7, 1, 3, bias=False, padding_mode="replicate")
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(width[1], width[2], 7, 1, 3, bias=False, padding_mode="replicate")
        self.pool3 = nn.AvgPool2d(2, 2)
        self.conv4 = nn.Conv2d(width[2], width[3], 1, 1, bias=False, padding_mode="replicate")

        # decoder
        self.conv5 = nn.Conv2d(width[3] + width[2], width[2], 7, 1, 3, bias=False, padding_mode="replicate")
        self.conv6 = nn.Conv2d(width[2] + width[1], width[1], 7, 1, 3, bias=False, padding_mode="replicate")
        self.conv7 = nn.Conv2d(width[1] + width[0], width[0], 7, 1, 3,  bias=False, padding_mode="replicate")
        self.conv8 = nn.Conv2d(width[0], 1, 7, 1, 3, bias=False, padding_mode="replicate")

    def forward(self, x):
        # x: refer_slope
        x = x.float()
        conv1 = self.conv1(x)
        x = self.pool1(conv1)
        conv2 = self.conv2(x)
        x = self.pool2(conv2)
        conv3 = self.conv3(x)
        x = self.pool3(conv3)
        x = self.conv4(x)
        x = fn.resize(x, size=[conv3.shape[2], conv3.shape[3]])  # resize1
        x = torch.cat((x, conv3), 1)  # stack1
        x = self.conv5(x)  # conv5
        x = fn.resize(x, size=[conv2.shape[2], conv2.shape[3]])  # resize2
        x = torch.cat((x, conv2), 1)  # stack2
        x = self.conv6(x)  # conv6
        x = fn.resize(x, size=[conv1.shape[2], conv1.shape[3]])  # resize3
        x = torch.cat((x, conv1), 1)  # stack3
        x = self.conv7(x)
        x = self.conv8(x)
        x = torch.squeeze(x,dim=1)

        return x

class PMD_loss(nn.Module):
    def __init__(self):
        super(PMD_loss,self).__init__()

    def forward(self,pred,truth):
        pass

class PMD_model_Unet(nn.Module):
    def __init__(self, width=None):
        super(PMD_model_Unet, self).__init__()
        if width is None:
            width = [32, 128, 512, 1024]
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=width[0], kernel_size=7, stride=1, padding=3, bias=False,
                               padding_mode="replicate")
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(width[0], width[1], 7, 1, 3, bias=False, padding_mode="replicate")
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(width[1], width[2], 7, 1, 3, bias=False, padding_mode="replicate")
        self.pool3 = nn.AvgPool2d(2, 2)
        self.conv4 = nn.Conv2d(width[2], width[3], 1, 1, bias=False, padding_mode="replicate")
        self.conv5 = nn.Conv2d(width[3] + width[2], width[2], 7, 1, 3, bias=False, padding_mode="replicate")
        self.conv6 = nn.Conv2d(width[2] + width[1], width[1], 7, 1, 3, bias=False, padding_mode="replicate")
        self.conv7 = nn.Conv2d(width[1] + width[0], width[0], 7, 1, 3,  bias=False, padding_mode="replicate")
        self.conv8 = nn.Conv2d(width[0], 1, 7, 1, 3, bias=False, padding_mode="replicate")

    def forward(self, x):
        # x: refer_slope
        x = x.float()
        conv1 = self.conv1(x)
        x = self.pool1(conv1)
        conv2 = self.conv2(x)
        x = self.pool2(conv2)
        conv3 = self.conv3(x)
        x = self.pool3(conv3)
        x = self.conv4(x)
        x = fn.resize(x, size=[conv3.shape[2], conv3.shape[3]])  # resize1
        x = torch.cat((x, conv3), 1)  # stack1
        x = self.conv5(x)  # conv5
        x = fn.resize(x, size=[conv2.shape[2], conv2.shape[3]])  # resize2
        x = torch.cat((x, conv2), 1)  # stack2
        x = self.conv6(x)  # conv6
        x = fn.resize(x, size=[conv1.shape[2], conv1.shape[3]])  # resize3
        x = torch.cat((x, conv1), 1)  # stack3
        x = self.conv7(x)
        x = self.conv8(x)
        x = torch.squeeze(x,dim=1)

        return x