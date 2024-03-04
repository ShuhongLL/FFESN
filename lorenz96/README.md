# Experiment of Lorenz96

## 1. Model Training

To start training Lorenz96 models meanwhile recording all states and figures, run the following:
```
python main.py
```

The following script simply train Lorenz96 models with `f` ranging from 0.25 to 8.0, with specified iteration time. The iteration time is recommended to be smaller than or equal to 5, as the nerual ODE can return NaN when iteration time becomes too large.
```
python train_lorenz.py
```

The optional arguments are

- epochs: number of epochs to train
- batch_size: number of batches on datasets
- f: externel force to determine dynamics of the internal system
- lr: learning rate of the SGD optimizor
- n: number of repeat, default is 5
- total_iterations: total # of iterations, float values, default is 5.0
- iteration_step: step length of iterations, default is 0.2


To train a single Lorenz96 model and calculate its FTMLE values on MNIST, run the following:
```
python train_lorenz_and_eval.py
```


## 2. Calculate FTMLE

After training the Lorenz96 models, you can evaluate their FTMLE values by:
```
python cal_ffesn_ftmle.py 
```

The script calculate the values for a batch of Lorenz96 models with f ranging from 0.25 to 8.0 and a specified iteration number. You may want to change the iteration number in its settings.


## 3. Plot Heat Map

After training and calculating the FTMLE values, the figure generation can be found in corresponding notebooks.