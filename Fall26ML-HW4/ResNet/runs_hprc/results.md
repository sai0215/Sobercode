# Problem 4 results (CIFAR-10, 100 epochs, Adam, batch 128, StepLR /1.5 every 10 epochs)

## (d) Learning-rate tuning (trained on 45k, selected on the 5k validation split)

| model | lr | epochs run | final train loss | val loss | val acc | test acc |
|---|---|---|---|---|---|---|
| (i) ResNet18 | 0.003 | 100 | 0.1163 | 0.6069 | 0.8412 | 0.8335 |
| (i) ResNet18 | 0.01 | 100 | 0.3459 | 0.5438 | 0.8298 | 0.8117 |
| (i) ResNet18 | 0.03 | 100 | 0.7677 | 0.8186 | 0.7190 | 0.7087 |
| (ii) ResNet18 w/o residual | 0.003 | 100 | 0.6920 | 0.7819 | 0.7508 | 0.7441 |
| (ii) ResNet18 w/o residual | 0.01 | 100 | 0.6006 | 0.7427 | 0.7620 | 0.7492 |
| (ii) ResNet18 w/o residual | 0.03 | 100 | 1.6133 | 1.6668 | 0.3662 | 0.3563 |
| (iii) ResNet18 w/o BatchNorm | 0.003 | 100 | 0.4411 | 0.6963 | 0.7880 | 0.7780 |
| (iii) ResNet18 w/o BatchNorm | 0.01 | 100 | 0.6198 | 0.7829 | 0.7422 | 0.7356 |
| (iii) ResNet18 w/o BatchNorm | 0.03 | 100 | 2.3027 | 2.3028 | 0.0950 | 0.1000 |

## (e) Best learning rate per model

| model | best lr | val acc | final test loss | **final test accuracy** |
|---|---|---|---|---|
| (i) ResNet18 | 0.003 | 0.8412 | 0.6483 | **0.8335** |
| (ii) ResNet18 w/o residual | 0.01 | 0.7620 | 0.7951 | **0.7492** |
| (iii) ResNet18 w/o BatchNorm | 0.003 | 0.7880 | 0.7222 | **0.7780** |
