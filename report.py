

import torch
from torch.utils import data
import torchvision.transforms as transforms


class My_data_set(data.Dataset):

    def __init__(self):
        # load images and annotations
        self.imgs = []
        self.annotations = []

    def __len__(self):
        # return the length of images
        return len(self.img_paths)

    def __getitem__(self, index):
        # do data augmentation
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)
        # return image and annotation of specific index
        return self.imgs[index], self.annotations[index]



dataset = My_data_set()
data_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=16,
                                        shuffle=True,
                                        num_workers=8,
                                        drop_last=True,
                                        pin_memory=True)




import torchvision.models as models
import torch.nn as nn

class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()

        self.resnet18 = models.resnet18(pretrained = True)

    def forward(self, x):

        x = self.resnet18(x)

        return x



from torch import optim

# build the model
model = my_model()
# multi GPUs training
model = torch.nn.DataParallel(model).cuda()
# build optimizer and learning rate scheduler
optimizer = optim.SGD(model.parameters(), lr = 1e-5, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.1)



import torch.nn as nn

criterion = nn.CrossEntropyLoss()




epochs = 100

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

for epoch in range(epochs):

    for times, data in enumerate(data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()



state = dict(epoch=epoch + 1,
                iter=0,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict())
file_path = '/root/...' + 'checkpoint.pth.tar'
torch.save(state, file_path)


