# Build model CNN for Detection Color Car

## 1. Architecture
[Link paper](https://arxiv.org/pdf/1510.07391.pdf) 

![CNN architecture](image/architecture.jpg)

## 3. Build model 
- Implementation paper with pytorch
- Custorm with metrics F1-score, Early Stopping, Scheduler learning rate.
- You can try with mobinet or resnet18. But it using pretrain not neccessary for simple classification.
## 2. Evaluation

- Confusion Matrix of Train dataset
![classification_report](image/classification_report_train.png "classificaion_report_Train") ![confusion matrix](image/confusion_matrix_train.png "Confusion Matrix of Train")
- confusion matrix of Test dataset
![classification_report](image/classification_report_test.png "classificaion_report_Train") ![confusion matrix](image/confusion_matrix_test.png "Confusion Matrix of Train")

## 3. Prediction
![KQ](image/KQ.png "KQ") 
- Inference Time: 0.028s