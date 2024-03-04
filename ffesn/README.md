# FFESN Experiments

## 1. Model Training

The following command starts training of FFESN
```
python main.py
```

the optional args are defined as:
- epochs: the number of training epochs, default set to 80
- batch_size: the batch size of the dataloader, default is 64
- rho: the spectral radius od the FFESN network
- iteration: the number of iteration
- n: number of repeat, default set to 1, you may want to disable saving model when n > 1
- lr: learning rate of SGD optimizor
- seed: seed, defualt set to 42
- dataset: we support MNIST or Fashoin-MNIST, any input other than 'mnist' will use fashion-minst
- save: whether save the model weight and the metrics
- device: cuda device, e.g. [0, 1, 2]

To train the model using a range of spectral radius and iteration, you may use the provided bash script as an example

## 2. Calculate FTMLE

After training the FFESN models, you can evaluate its FTMLE values by
```
python cal_ffesn_ftmle.py 
```

The script calculate the values for a batch of FFESN models with rho ranging from 0.1 to 2.0 and a specified iteration number. You may want to change the iteration number in its settings.

## 3. Plot Heat Map

The heat map of accuracy, converging speed, and FTMLE values can be produced in the notebook `mnist_accuracy.ipynb`.

