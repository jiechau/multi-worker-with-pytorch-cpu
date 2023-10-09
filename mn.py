import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm 

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

# Load data
train_loader = torch.utils.data.DataLoader(
  datasets.MNIST('/tmp/data', train=True, download=True, 
                 transform=transforms.Compose([
                   transforms.ToTensor(), 
                   transforms.Normalize((0.1307,), (0.3081,))
                  ])),
  batch_size=64, shuffle=True)

print('train_loader', len(train_loader))

# Initialize model and optimizer
model = build_model()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Train model
for epoch in range(3):
  enu_train_loader = enumerate(train_loader)
  for batch_idx in tqdm(range(len(train_loader)), desc ="Step"):
  #for batch_idx, (data, target) in enumerate(train_loader):
    _, (data, target) = next(enu_train_loader)

    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()

  print('Epoch:', epoch, ', Loss:', loss.item())

# Save model
torch.save(model.state_dict(), '/tmp/pt_model.pt')