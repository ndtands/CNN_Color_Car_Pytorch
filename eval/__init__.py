import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sn
from configs import *
import matplotlib.pyplot as plt
def eval(model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        device: str='cpu'):   
    """
        > Evaluate the model on the given dataset and return the predicted labels and true labels
        
        :param model: the model to be evaluated
        :type model: torch.nn.Module
        :param dataloader: The DataLoader object that contains the data to be evaluated
        :type dataloader: torch.utils.data.DataLoader
        :param device: The device to run the model on, defaults to cpu
        :type device: str (optional)
    """
    # Put model in eval mode
    model.eval() 
    # Turn on inference context manager
    all_predicts = []
    all_labels = []
    # Loop through DataLoader batches
    for _ , batch in enumerate(dataloader):
        # Send data to target device
        batch_images, batch_labels = batch[0].to(device), batch[1].to(device)

        # 1. Forward pass
        test_pred_logits = model(batch_images)

        # Calculate and accumulate accuracy
        test_pred = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
        all_predicts.extend(test_pred.cpu().numpy().tolist())
        all_labels.extend(batch_labels.cpu().numpy().tolist())
    all_labels_tags = [IDX2TAG[str(idx)] for idx in  all_labels]
    all_predicts_tags = [IDX2TAG[str(idx)] for idx in  all_predicts]
    print(classification_report(y_true=all_labels_tags, y_pred=all_predicts_tags))
    return all_predicts, all_labels

def plot_confusion_matrix(y_true: list, y_predict: list) -> None:
    """
    It takes in the true labels and predicted labels, and plots a heatmap of the confusion matrix
    
    :param y_true: list of actual labels
    :type y_true: list
    :param y_predict: list of predicted tags
    :type y_predict: list
    """
    cm = confusion_matrix(y_true, y_predict)
    cm_normalized = np.round(cm/np.sum(cm,axis=1).reshape(-1,1),2)
    plt.figure(figsize=(10,7.5))
    sn.heatmap(cm_normalized, cmap="Blues", annot=True,
                cbar_kws={"orientation":"vertical", "label":"color bar"},
                xticklabels=[i for i in TAGS], yticklabels=[i for i in TAGS])
    plt.xlabel("Predicted")
    plt.ylabel('Actual')
    plt.title('NormalConfusion Matrix')
    plt.show()