# Problem 4 results (CIFAR-10, 100 epochs, Adam, batch 128, StepLR /1.5 every 10 epochs)

## (d) Learning-rate tuning (trained on 45k, selected on the 5k validation split)

| model | lr | epochs run | final train loss | val loss | val acc | test acc |
|---|---|---|---|---|---|---|
| (i) ResNet18 | 0.003 | 100 | 0.1199 | 0.6108 | 0.8356 | 0.8394 |
| (i) ResNet18 | 0.01 | 100 | 0.3495 | 0.5783 | 0.8102 | 0.8078 |
| (i) ResNet18 | 0.03 | 100 | 0.7465 | 0.8141 | 0.7230 | 0.7096 |
| (ii) ResNet18 w/o residual | 0.003 | 100 | 0.5439 | 0.6909 | 0.7726 | 0.7691 |
| (ii) ResNet18 w/o residual | 0.01 | 100 | 0.6708 | 0.7914 | 0.7438 | 0.7353 |
| (ii) ResNet18 w/o residual | 0.03 | 100 | 1.6250 | 11.8054 | 0.1388 | 0.1308 |
| (iii) ResNet18 w/o BatchNorm | 0.003 | 100 | 0.4340 | 0.6825 | 0.7878 | 0.7767 |
| (iii) ResNet18 w/o BatchNorm | 0.01 | 100 | 2.3026 | 2.3028 | 0.0950 | 0.1000 |
| (iii) ResNet18 w/o BatchNorm | 0.03 | 100 | 126.9673 | 2.3028 | 0.0950 | 0.1000 |

## (e) Best learning rate per model

| model | best lr | val acc | final test loss | **final test accuracy** |
|---|---|---|---|---|
| (i) ResNet18 | 0.003 | 0.8356 | 0.6153 | **0.8394** |
| (ii) ResNet18 w/o residual | 0.003 | 0.7726 | 0.7199 | **0.7691** |
| (iii) ResNet18 w/o BatchNorm | 0.003 | 0.7878 | 0.7224 | **0.7767** |
