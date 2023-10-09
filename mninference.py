import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torchvision


# Define model 
class build_model(nn.Module):
    def __init__(self):
        super(build_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load model 
model = build_model()
model.load_state_dict(torch.load('/tmp/pt_model.pt'))
model.eval()


# Load one random test image
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/tmp/data', train=False, download=True,
  transform=torchvision.transforms.Compose([
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize(
       (0.1307,), (0.3081,))
  ])),
  batch_size=1, shuffle=True)

image, label = next(iter(test_loader))

# Make prediction
output = model(image)
pred = output.argmax(dim=1, keepdim=True)

print('Predicted label: ', pred, label)
for row_index, row in enumerate(output):
    for col_index, value in enumerate(row):
        print(f"{col_index} : {value}")