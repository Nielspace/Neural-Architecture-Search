import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 


def training(max_patience, num_epochs, model, optimizer, training_loader, val_loader):
    nll_val = []
    error_val = []
    best_nll = 1000.
    patience = 0

    # Main training loop
    for e in range(num_epochs):
        model.train() # set the model to the training mode
        # load batches
        for indx_batch, (batch, targets) in enumerate(training_loader):
          # calculate the forward pass (loss function for given images and labels)
          loss = model.forward(batch, targets)
          # remember we need to zero gradients! Just in case!
          optimizer.zero_grad()
          # calculate backward pass
          loss.backward(retain_graph=True)
          # run the optimizer
          optimizer.step()

        # Validation: Evaluate the model on the validation data
        loss_e, error_e = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_e)  # save for plotting
        error_val.append(error_e)  # save for plotting

        # Early-stopping: update the best performing model and break training if no 
        # progress is observed.
        if e == 0:
            pass
        else:
            if loss_e < best_nll:
                patience = 0
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    # Return nll and classification error.
    nll_val = np.asarray(nll_val)
    error_val = np.asarray(error_val)
    return nll_val, error_val


def evaluation(test_loader, model_best=None, epoch=None):
  
    model_best.eval()# set the model to the evaluation mode
    loss_test = 0.
    loss_error = 0.
    N = 0.
    # start evaluation
    for indx_batch, (test_batch, test_targets) in enumerate(test_loader):
        # loss (nll)
        loss_test_batch = model_best.forward(test_batch, test_targets, reduction='sum')
        loss_test = loss_test + loss_test_batch.item()
        # classification error
        y_pred = model_best.classify(test_batch)
        e = 1.*(y_pred == test_targets)
        loss_error = loss_error + (1. - e).sum().item()
        # the number of examples
        N = N + test_batch.shape[0]
    # divide by the number of examples
    loss_test = loss_test / N
    loss_error = loss_error / N

    # Print the performance
    if epoch is None:
        print(f'-> FINAL PERFORMANCE: nll={loss_test}, ce={loss_error}')
    else:
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, val nll={loss_test}, val ce={loss_error}')

    return loss_test, loss_error