from torch import nn, flatten


class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Add more layers...
        self.fc1 = nn.Linear(128, 200)  # 200 is the number of classes in TinyImageNet

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Define forward pass

        # B x 3 x 224 x 224
        x = self.pool(self.conv1(x).relu())  # B x 64 x 112 x 112
        x = self.pool(self.conv2(x).relu())  # B x 128 x 56 x 56
        x = self.pool(self.conv3(x).relu())  # B x 256 x 28 x 28
        x = self.gap(x)  # B x 256 x 1 x 1
        x = flatten(x, 1)  # B x 256
        x = self.fc1(x)  # B x 200
        return x
