
import torch 
import torch.nn as nn 
import torchvision

import numpy as np


from nas import NAS



def ingestionEngine(resize:list, path:str, download:bool, batch_size:int, shuffle:bool): 
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize[-1]),
        torchvision.transforms.CenterCrop(resize[-1]),
        torchvision.transforms.ToTensor()])
        

    train = torchvision.datasets.MNIST(path, train=True, download=download, transform=transform)
    test  = torchvision.datasets.MNIST(path, train=False, download=download, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)

    return train_loader, val_loader



# -> training hyperparams
lr = 1e-3 # learning rate
wd = 1e-5 # weight decay
num_epochs = 11 # max. number of epochs
max_patience = 100 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
no_of_blocks = 5  #no. of child networks in population 

params = {
    'filter':[8,16,32],
    'kernel_size':[3,5],
    'activation':[nn.ReLU(),nn.Sigmoid(),nn.Tanh(),nn.Softplus(),nn.ELU()],
    'pooling':[nn.MaxPool2d(2),nn.MaxPool2d(1),nn.AvgPool2d(2),nn.AvgPool2d(1)],
}

resize = [1,8,8]

train_loader, val_loader = ingestionEngine(resize, './', True, 64, True)

networks = NAS(params, resize)
population = networks.blocks(no_of_blocks)
class_errors =  networks.train(population=population,
                                num_epochs=num_epochs,
                                training_loader=train_loader,
                                val_loader=val_loader,
                                test_loader=val_loader,lr=lr,wd=wd,max_patience=max_patience)

networks.find_best_model(population,class_errors)



