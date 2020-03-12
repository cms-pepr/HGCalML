# Object condensation


## Training
For now, the training is only in eager mode. You can do plots using jupyter
notebook. The script is included. For experimentation, you have a choice of 4 models:

1. Overfitting king: a very dense network
2. Overfitting queen: a GRU based recurrent network
3. Overfitting prince: a conv based recurrent network
4. GravNet


As the name suggests, the first 3 are only for overfitting and I recommend you to try
overfitting using the second model `overfitting-queen`.


Source HGCal as you normally do and then:
```
python training_eager.py --trainid comparison_loss_both --modeltype dgcnn-alpha --shouldoverfit True
```

Let it train for a while. It will save summaries and models to:
```
$PWD/train_data/comparison_loss_both/summaries
$PWD/train_data/comparison_loss_both/checkpoints

```

## Plotting
Open the jupyter notebook `visualizing.ipynb`. And rest is documented there!


## Resuming training

Use the same command:

```
python training_eager.py --trainid comparison_loss_both --modeltype dgcnn-alpha --shouldoverfit True
```


Where `comparison_loss_both` is a training id generated before.