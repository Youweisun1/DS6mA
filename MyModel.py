
import torch
import torch.nn as nn
class DS6mA(nn.Module):
    def __init__(self, vocab_size=24):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv4 = convmixer_block(32, 5)
        # self.drop = nn.Sequential(
        #     nn.Dropout(0.1)
        # )
        self.out = nn.Linear(1312, 1024)  # fully connected layer, output 10 classes　　　
        self.block1 = nn.Sequential(nn.Linear(1024, 256),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    )

        self.block2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.drop(x)
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return self.block1(out)
    def trainModel(self, x):
        with torch.no_grad():
            output = self.forward(x)
        return self.block2(output)

class convmixer_block(nn.Module):
    def __init__(self,dim,kernel_size):
        super().__init__()
        self.dw=nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size, groups=dim, stride=1, padding=2),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Conv1d(dim, dim, kernel_size, groups=dim, stride=1, padding=2),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
        )
        self.pw=nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.Dropout(0.1)
        )
    def forward(self,x):
        x=self.dw(x)+x
        x=self.pw(x)
        return x

