import torch
from torch import nn
from torch.functional import F
from src.models.parameter_vector import FederatedModelMixin


class ConvolutionalNet(FederatedModelMixin, nn.Module):
    def __init__(self, in_channels=1, device_num=10, random_seed=42, num_class=10):
        super(ConvolutionalNet, self).__init__()
        torch.random.default_generator.manual_seed(random_seed + device_num)

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64 * 8 * 8, num_class)

        self.blocks = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.fc1
        ]

        self.all_modules = self.blocks
        self.conv_modules = [self.conv1, self.conv2, self.conv3]
        self.fc_modules = [self.fc1]
        self.finalize_model_setup()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = x
        return output

if __name__ == '__main__':
    model = ConvolutionalNet()
    params = model.get_parameter_vector("all")
    model.load_parameter_vector(params, "all")
