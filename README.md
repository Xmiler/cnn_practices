## CIFAR10

Model: `resnet50`

| Parameters | Acc. | Comments |
| ----------------- | ----------- | ----------- |
| Baseline | 93.58% | `14bc8dd` |
| Adam, const lr=0.01 | 46.62% | `9bee9dc` tried also lr=0.1 but it doesn't converge |



[Baseline](https://github.com/kuangliu/pytorch-cifar/):
```
SGD
lr = 0.1 for epoch [0, 150), 0.01 for epoch [150, 250), 0.001 for epoch [250, 350)
momentum=0.9
weight_decay=5e-4
```

