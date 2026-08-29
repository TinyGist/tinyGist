import torch
from torch import nn
from src.models.parameter_vector import FederatedModelMixin


class DeepFC(FederatedModelMixin, nn.Module):
    def __init__(self, input_size=784, device_num=10, random_seed=42, num_class=10):
        super(DeepFC, self).__init__()
        torch.random.default_generator.manual_seed(random_seed + device_num)
        self.l1 = nn.Linear(input_size, 392)
        self.relu1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(392, 196)
        self.relu2 = nn.ReLU(inplace=True)
        self.l3 = nn.Linear(196, 32)
        self.relu3 = nn.ReLU(inplace=True)
        self.l4 = nn.Linear(32, num_class)

        self.blocks = [self.l1, self.relu1, self.l2, self.relu2, self.l3, self.relu3, self.l4]
        self.all_modules = self.blocks
        self.conv_modules = []
        self.fc_modules = [self.l1, self.l2, self.l3, self.l4]
        self.finalize_model_setup()

    def forward(self, x):
        x = torch.flatten(x, 1)
        for block in self.blocks:
            x = block(x)
        output = x
        return output
