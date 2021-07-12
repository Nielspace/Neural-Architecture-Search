import torch 
import torch.nn as nn
import torch.nn.functional as F 


class basemodel(nn.Module):

    """This model will be used to mutate with best configuration. Remember this is a simple
       classifier base model.     
    """

    def __init__(self, classnet):
        super(basemodel, self).__init__()
        # We provide a sequential module with layers and activations
        self.classnet = classnet
        # The loss function (the negative log-likelihood)
        self.nll = nn.NLLLoss(reduction='none') 

    #mapping input with the output
    def classify(self, x):

        y_pred = self.classnet(x)
        _,pred_label = torch.max(y_pred, dim = 1)

        return pred_label

    
    def forward(self, x, y, reduction='avg'):

        loss = self.nll(self.classnet(x),y.long())

        if reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()