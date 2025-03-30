import torch.nn as nn

class ConvolutionalNeuralNetworks(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworks, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, padding=0, stride=1),
            nn.Tanh()
            )
            
        self.fc = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10)
            )


    def forward(self, x):        
        x = self.conv(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x