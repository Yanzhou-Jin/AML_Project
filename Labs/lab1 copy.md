# Model Training Commands
A defult configuration has been established to systematically evaluate the impact of the three key factors on the model. These parameters include: 
* batch-size
* max-epochs
* learning-rate
## Default Configuration

The defult configuration, as defined, conpried the parameters as follows:
* batch-size = 256
* max-epochs = 10
* learning-rate = 1e-05

```
./ch train jsc-tiny jsc --max-epochs 10 --batch-size 256 --learning-rate 1e-05

tensorboard --logdir ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_Base/software --port 16006

./ch test jsc-tiny jsc --load ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_Base/software/training_ckpts/best.ckpt --load-type pl
```

This defult configuration serve as the baseline for the evaluation. Systematic deviations will be introduced from this baselineto investigate their impact on the model's performance.

''''This approach ensures a comprehensive analysis of the model's behavior under different parameter settings, allowing for a nuanced understanding of its robustness and adaptability.'''

## Training with Different Batch Sizes
```
./ch train jsc-tiny jsc --max-epochs 10 --batch-size 64 --learning-rate 1e-05
./ch test jsc-tiny jsc --load ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_BS64/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_BS64/software --port 16007

./ch train jsc-tiny jsc --max-epochs 10 --batch-size 1024 --learning-rate 1e-05
./ch test jsc-tiny jsc --load ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_BS1024/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_BS1024/software --port 16006

./ch train jsc-tiny jsc --max-epochs 5 --batch-size 8 --learning-rate 1e-05
```


## Training with Different Epochs
```
./ch train jsc-tiny jsc --max-epochs 5 --batch-size 256 --learning-rate 1e-05

./ch test jsc-tiny jsc --load ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_ME5/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_ME5/software --port 16006

./ch train jsc-tiny jsc --max-epochs 15 --batch-size 256 --learning-rate 1e-05

./ch test jsc-tiny jsc --load ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_ME15/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_ME15/software --port 16006

```


## Training with Different Learning Rates
```
./ch train jsc-tiny jsc --max-epochs 10 --batch-size 256 --learning-rate 1e-03
./ch test jsc-tiny jsc --load ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_LR3/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_LR3/software --port 16006

./ch train jsc-tiny jsc --max-epochs 10 --batch-size 256 --learning-rate 1e-07
./ch test jsc-tiny jsc --load ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_LR7/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_LR7/software --port 16006
```

## Training a 10x Model

```
./ch train jsc-toy jsc --max-epochs 10 --batch-size 256 --learning-rate 1e-03
./ch test jsc-toy jsc --load /home/super_monkey/mase_new/mase_output/Lab1-Task1/jsc-toy_classification_jsc_1X/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-toy_classification_jsc_1X/software --port 16006


./ch train jsc-toy10X jsc --max-epochs 10 --batch-size 256 --learning-rate 1e-03

./ch test jsc-toy10X jsc --load /home/super_monkey/mase_new/mase_output/Lab1-Task1/jsc-toy10X_classification_jsc_10Xn/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-toy10X_classification_jsc_10Xn/software --port 16006


./ch train jsc-toy10X jsc --max-epochs 15 --batch-size 256 --learning-rate 1e-03
./ch test jsc-toy10X jsc --load /home/super_monkey/mase_new/mase_output/Lab1-Task1/jsc-toy10X_classification_jsc_10X12/software/training_ckpts/best.ckpt --load-type pl

```

## Model Testing
./ch test jsc-tiny jsc --load ../mase_output/jsc-tiny_classification_jsc_Base/software/training_ckpts/best.ckpt --load-type pl

# TensorBoard and Checkpoint

## TensorBoard Command
/mase/machop$ tensorboard --logdir ../mase_output/jsc-tiny_classification_jsc_Base/software --port 16006

## Checkpoint Load Command
./ch test jsc-tiny jsc --load ../mase_output/jsc-tiny_classification_jsc_Base/software/training_ckpts/best.ckpt --load-type pl

# Other Model Training Commands
```
./ch train jsc-tiny jsc --max-epochs 10 --batch-size 256 --learning-rate 1e1
./ch test jsc-tiny jsc --load ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_LR-1/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_LR-1/software --port 16006

./ch train jsc-tiny jsc --max-epochs 10 --batch-size 64 --learning-rate 1e-03
./ch test jsc-tiny jsc --load ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_BS64LR3/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_BS64LR3/software --port 16006

./ch train jsc-tiny jsc --max-epochs 10 --batch-size 64 --learning-rate 1e-07
./ch test jsc-tiny jsc --load ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_BS64LR7/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_BS64LR7/software --port 16006


./ch train jsc-tiny jsc --max-epochs 10 --batch-size 1024 --learning-rate 1e-03
./ch test jsc-tiny jsc --load ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_BS1024LR3/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_BS1024LR3/software --port 16006


./ch train jsc-tiny jsc --max-epochs 10 --batch-size 1024 --learning-rate 1e-07
./ch test jsc-tiny jsc --load ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_BS1024LR7/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-tiny_classification_jsc_BS1024LR7/software --port 16006

```

