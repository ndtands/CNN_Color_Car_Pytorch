import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.Optimizer,
                device: str='cpu'):
    """
    > The function takes a model, a dataloader, a loss function, an optimizer, and a device, and returns
    the average loss and f1-score for the model on the dataloader
    
    :param model: torch.nn.Module - the model to train
    :type model: torch.nn.Module
    :param dataloader: the dataloader for the training set
    :type dataloader: torch.utils.data.DataLoader
    :param loss_fn: torch.nn.Module
    :type loss_fn: torch.nn.Module
    :param optimizer: torch.optim.Optimizer
    :type optimizer: torch.optim.Optimizer
    :param device: str='cpu', defaults to cpu
    :type device: str (optional)
    """
               
    # Put model in train mode
    model.train()
    
    train_loss =  0
    all_predicts = []
    all_labels = []
    for step, batch in enumerate(dataloader):
        # Send data to target device
        batch_images, batch_labels = batch[0].to(device), batch[1].to(device)

        # 1. Forward pass
        batch_preds = model(batch_images)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(batch_preds, batch_labels)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        if scheduler is not None:
            scheduler.step(train_loss)

        # Calculate and accumulate f1-score metric across all batches
        batch_pred_class = torch.argmax(torch.softmax(batch_preds, dim=1), dim=1)
        all_predicts.extend(batch_pred_class.cpu().numpy().tolist())
        all_labels.extend(batch_labels.cpu().numpy().tolist())
    
    # Adjust metrics to get average loss and f1-score per batch 
    train_loss = train_loss / len(dataloader)
    train_f1 = f1_score(
        y_true = all_labels,
        y_pred = all_predicts,
        average='macro'
    )
    return train_loss, train_f1

def test_step(model: torch.nn.Module, 
            dataloader: torch.utils.data.DataLoader, 
            loss_fn: torch.nn.Module,
            device: str='cpu'):
    """
    > This function takes a model, a dataloader, a loss function, and a device, and returns the average
    loss and accuracy of the model on the dataloader
    
    :param model: torch.nn.Module - the model to train
    :type model: torch.nn.Module
    :param dataloader: A DataLoader object that iterates over the test dataset
    :type dataloader: torch.utils.data.DataLoader
    :param loss_fn: torch.nn.Module
    :type loss_fn: torch.nn.Module
    :param device: The device to run the training on, defaults to cpu
    :type device: str (optional)
    """
            
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss = 0
    
    # Turn on inference context manager
    all_predicts = []
    all_labels = []
    # Loop through DataLoader batches
    for _ , batch in enumerate(dataloader):
        # Send data to target device
        batch_images, batch_labels = batch[0].to(device), batch[1].to(device)

        # 1. Forward pass
        test_pred_logits = model(batch_images)

        # 2. Calculate and accumulate loss
        loss = loss_fn(test_pred_logits, batch_labels)
        test_loss += loss.item()
        
        # Calculate and accumulate accuracy
        test_pred = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
        all_predicts.extend(test_pred.cpu().numpy().tolist())
        all_labels.extend(batch_labels.cpu().numpy().tolist())
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_f1 = f1_score(
        y_true = all_labels,
        y_pred = all_predicts,
        average='macro'
    )
    return test_loss, test_f1

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 20,
          patient: int = 4,
          model_path: str = 'best_model.pt',
          device: str ='cpu'):
    
    results = {"train_loss": [],
                "train_f1": [],
                "test_loss": [],
                "test_f1":  []
            }
    f1_best = 0
    count_es = 0
    for epoch in tqdm(range(epochs)):
        train_loss, train_f1 = train_step(
                                model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                device=device,
                                )
        test_loss, test_f1 = test_step(
                                model=model,
                                dataloader=test_dataloader,
                                loss_fn=loss_fn,
                                device=device,
                                )
        if test_f1 > f1_best:
            print(f'Model improve from {f1_best : .4f} to {test_f1 : .4f}')
            f1_best = test_f1
            torch.save(model,model_path)
            count_es = 0
        else:
            count_es += 1
            print(f'Model not improver {count_es}/{patient}')
            if count_es == patient:
                print(f'Model Early stopping at {epoch} eps with F1_score best is {f1_best: .4f}')
                break

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_f1"].append(train_f1)
        results["test_loss"].append(test_loss)
        results["test_f1"].append(test_f1)
        
    # 6. Return the filled results at the end of the epochs
    return results

