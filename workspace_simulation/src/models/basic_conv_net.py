import torch
from torch import nn
from src.models.parameter_vector import FederatedModelMixin


class BasicConv(FederatedModelMixin, nn.Module):
    def __init__(self, in_channels=1, device_num=10, random_seed=47, num_class=10):
        torch.random.default_generator.manual_seed(random_seed + device_num)
        super(BasicConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 3, padding=1)
        self.avgpool1 = nn.AvgPool2d(2, 2) # 1/2
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.avgpool2 = nn.AvgPool2d(2, 2)  # 1/2
        self.relu2 = nn.ReLU(inplace=True)

        self.flat3 = nn.Flatten(1)
        self.fc4 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(32, num_class)

        self.blocks = [
            self.conv1,
            self.avgpool1,
            self.relu1,
            self.conv2,
            self.avgpool2,
            self.relu2,
            self.flat3,
            self.fc4,
            self.relu4,
            self.classifier
        ]

        self.all_modules = self.blocks
        self.conv_modules = [self.conv1, self.conv2]
        self.fc_modules = [self.fc4, self.classifier]
        self.finalize_model_setup()


    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

if __name__ == '__main__':
    model = BasicConv()
    params = model.get_parameter_vector("all")
    model.load_parameter_vector(params, "all")
