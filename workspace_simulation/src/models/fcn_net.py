import torch
from torch import nn
from src.models.parameter_vector import FederatedModelMixin


class FCN(FederatedModelMixin, nn.Module):
    def __init__(self, input_size=784, device_num=10, random_seed=42, num_class=10):
        super(FCN, self).__init__()
        torch.random.default_generator.manual_seed(random_seed + device_num)
        self.l1 = nn.Linear(input_size, 12)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(12, num_class)

        #nn.init.normal_(self.l1.weight, mean=0.0, std=0.05)
        #nn.init.constant_(self.l1.bias, 0.0)
        #nn.init.normal_(self.l3.weight, mean=0.0, std=0.05)
        #nn.init.constant_(self.l3.bias, 0.0)

        self.blocks = [self.l1, self.relu, self.l3]
        self.all_modules = self.blocks
        self.conv_modules = []
        self.fc_modules = [self.l1, self.l3]
        self.finalize_model_setup()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l3(x)
        output = x
        return output

if __name__ == '__main__':
    from torch.optim import Adam
    from torch.nn import CrossEntropyLoss
    from torch.utils.data import DataLoader
    import fiftyone as fo
    import tqdm
    from torchvision.transforms import Compose, Normalize
    from src.data_getter.kws_getter import SpeechCommandsDataset

    model = FCN(input_size=1024)
    model.to('cuda')
    lr = 1e-3
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    trian_fo = fo.Dataset.from_dir(
        dataset_dir='../../data/kws/train',
        dataset_type=fo.types.FiftyOneDataset
    )
    test_fo = fo.Dataset.from_dir(
        dataset_dir='../../data/kws/test',
        dataset_type=fo.types.FiftyOneDataset
    )
    transform = Compose([Normalize(-0.4, 37)])
    train_dataset = SpeechCommandsDataset(trian_fo, transform)
    test_dataset = SpeechCommandsDataset(test_fo, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    EPOCHS = 100
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_pbar = tqdm.tqdm(train_dataloader, unit="batches", desc=f"Train Epoch {epoch}:")
        running_loss = 0.0
        running_accuracy = 0.0
        n = 0
        for data, label in train_pbar:
            data = data.to('cuda')
            label = label.to('cuda')
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            y_pred = torch.argmax(output, dim=1)
            running_accuracy += (y_pred == label).float().mean().item()
            running_loss += loss.item()
            n += 1

            train_pbar.set_postfix(loss=f"{running_loss / n:.3f}", accuracy=f"{running_accuracy / n:.3f}")

        model.eval()
        test_pbar = tqdm.tqdm(test_dataloader, unit="batches", desc=f"Test Epoch {epoch}:")
        running_loss = 0.0
        running_accuracy = 0.0
        n = 0
        with torch.no_grad():
            for data, label in test_pbar:
                data = data.to('cuda')
                label = label.to('cuda')
                output = model(data)
                loss = criterion(output, label)
                y_pred = torch.argmax(output, dim=1)
                running_accuracy += (y_pred == label).float().mean().item()
                running_loss += loss.item()
                n += 1

                test_pbar.set_postfix(loss=f"{running_loss / n:.3f}", accuracy=f"{running_accuracy / n:.3f}")
