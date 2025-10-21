from torch import nn

class BasicCNNModule(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return self.relu(x)
    

class BasicCNN(nn.Module):
    def __init__(self, in_channels=3, hdim=64, num_classes=3, input_size=256):
        super().__init__()
        # 5 layers of conv with downsampling each spatial dimension by factor of 1/2 while
        # doubling the channels
        self.conv_modules = nn.ModuleList(
            [BasicCNNModule(in_channels=in_channels, out_channels=hdim, kernel_size=3, stride=1, padding=1)]
            + [BasicCNNModule(
                in_channels=hdim * (2**i), 
                out_channels=hdim * (2**(i+1)), 
                kernel_size=3, 
                stride=1, 
                padding=1) for i in range(4)
            ]
        )
        final_features = int((input_size * input_size) * hdim * (2 ** 4) / (2 ** 10))
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(in_features=final_features, out_features=num_classes)

    def forward(self, x):
        for module in self.conv_modules:
            x = module(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x