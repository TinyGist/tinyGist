import torch
from torch import nn
from torch.functional import F
from src.models.parameter_vector import FederatedModelMixin


class LeNet5(FederatedModelMixin, nn.Module):
    def __init__(self, in_channels=1, device_num=10, random_seed=42, num_class=10):
        super(LeNet5, self).__init__()
        torch.random.default_generator.manual_seed(random_seed + device_num)

        # LeNet-5 conv layers
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # AvgPool and final FC layers
        self.avgpool = nn.AvgPool2d(2, 2)   # size halves
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

        self.blocks = [
            self.conv1,
            self.conv2,
            self.fc1,
            self.fc2,
            self.fc3
        ]

        self.all_modules = self.blocks
        self.conv_modules = [self.conv1, self.conv2]
        self.fc_modules = [self.fc1, self.fc2, self.fc3]
        self.finalize_model_setup()

    def forward(self, x):
        x = self.avgpool(F.tanh(self.conv1(x)))
        x = self.avgpool(F.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        output = x
        return output

if __name__ == '__main__':
    model = LeNet5()
    params = model.get_parameter_vector("all")
    model.load_parameter_vector(params, "all")
