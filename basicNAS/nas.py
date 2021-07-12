import random
import torch 
import torch.nn as nn  
import torch.nn.functional as F 

import numpy as np 

from basemodel import basemodel
from engine import training, evaluation

class Reshape(nn.Module):
    def __init__(self, size):
        super(Reshape, self).__init__()
        self.size = size # a list

    def forward(self, x): 
        if len(x.shape) == 4:
            return x
        return x.view(x.shape[0], *self.size)

# This module flattens an input (tensor -> matrix) by blending dimensions 
# beyond the batch size.
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
  
    def forward(self, x):
        return x.view(x.shape[0], -1)

class NAS():
    def __init__(self, params, size):
        self.params = params
        self.size = size
    
    def network(self):
        fltr = random.choice(self.params['filter'])
        ker_size = random.choice(self.params['kernel_size'])
        pad = 1 if ker_size == 3 else 2
        pooling = random.choice(self.params['pooling'])
        activation = random.choice(self.params['activation'])
        l_neuron = random.randrange(10,110,10)

        for_size = nn.Sequential(Reshape(self.size),
                             nn.Conv2d(1,fltr,ker_size,1,pad),
                             activation,
                             pooling,
                             Flatten())
        flat_size = for_size(torch.randn(1,self.size[1]*self.size[2])).shape[1]


        classnet = nn.Sequential(Reshape(self.size),
                             nn.Conv2d(1,fltr,ker_size,1,pad),
                             activation,
                             pooling,
                             Flatten(),
                             nn.Linear(flat_size,l_neuron),
                             activation,
                             nn.Linear(l_neuron,10),
                             nn.LogSoftmax(dim=1))
        return classnet


    def blocks(self, no_of_blocks):
        networks = []
        for i in range(no_of_blocks):
            networks.append(self.network())
        return networks

    def count_parameters(self,model):
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params+=param
        return total_params

    def objective(self,population,class_errors):
        weights = []
        for child in population:
            weights.append(self.count_parameters(child))
        max_weight = max(weights)
        objs = {}
        for i in range(len(weights)):
            obj = (class_errors[i]) + 0.01 * (weights[i]/max_weight)
            objs[obj] = i
        return objs
    
    def find_best_model(self,population,class_errors):
        best_model_index = sorted(self.objective(population,class_errors).items())[0][1]
        best_model_objective = sorted(self.objective(population,class_errors).items())[0][0]
        print(f'\n Best Model with objecive {best_model_objective:.4f}: \n {population[best_model_index]}')

    def train(self,population,num_epochs,training_loader,val_loader,test_loader,lr,wd,max_patience):
        pop = population
        class_errors = []
        for i,child in enumerate(pop):
            print('Training Child: ',i)
            model = basemodel(child)
            optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr, weight_decay=wd) 
            nll_val, error_val = training(max_patience=max_patience,
                                    num_epochs=num_epochs,
                                    model=model,
                                    optimizer=optimizer,
                                    training_loader=training_loader,
                                    val_loader=val_loader)
            test_loss, test_error = evaluation(test_loader=test_loader,model_best=model)
            class_errors.append(test_error)
        return class_errors


params = {
    'filter':[8,16,32],
    'kernel_size':[3,5],
    'activation':[nn.ReLU(),nn.Sigmoid(),nn.Tanh(),nn.Softplus(),nn.ELU()],
    'pooling':[nn.MaxPool2d(2),nn.MaxPool2d(1),nn.AvgPool2d(2),nn.AvgPool2d(1)],
}

