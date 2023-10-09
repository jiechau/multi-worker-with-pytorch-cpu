import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim

import os
os.environ['GLOO_LOG_LEVEL'] = 'DEBUG'
os.environ['MASTER_PORT'] = '8088'
os.environ['MASTER_ADDR'] = '172.17.2.15'
os.environ['WORLD_SIZE'] = '2' 


import sys
worker_id = int(sys.argv[1])

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

from torchvision import datasets, transforms
# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# Load training data
trainset = datasets.MNIST('/tmp/data', download=True, train=True, transform=transform)

# Initialize distributed backend  
dist.init_process_group(backend='gloo', # Use 'nccl' for GPU or 'gloo' for CPU
#                        init_method='tcp://127.0.0.1:8088',
                        init_method='tcp://172.17.2.15:8088',
                        world_size=2,
                        rank=worker_id)

dist.barrier()

# Distribute the data using DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

# Load model and data
model = build_model() 
optimizer = optim.SGD(model.parameters(), lr=0.01)
train_loader = torch.utils.data.DataLoader(
  dataset=trainset,
  batch_size=64,
  shuffle=False,
  num_workers=0,
  drop_last=True,
  sampler=train_sampler  # Use the distributed sampler
)

# Wrap model for distributed training
#model = nn.parallel.DistributedDataParallel(single_model, device_ids=[dist.get_rank()])
model = nn.parallel.DistributedDataParallel(model)


# Train model 
for epoch in range(1):

  for batch_idx, (data, target) in enumerate(train_loader):
    
    output = model(data)
    loss = F.nll_loss(output, target)
    
    # Average gradients
    loss.backward()
    optimizer.step()

    dist.all_reduce(loss) 
    loss /= dist.get_world_size()

  print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', loss.item())

# Save model
if dist.get_rank() == 0: 
  torch.save(model.model.state_dict(), '/tmp/pt_model.pt')