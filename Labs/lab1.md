# Pratical 1
## Model Training Commands
A defult configuration has been established to systematically evaluate the impact of the three key factors on the model. These parameters include: 
* batch-size
* max-epochs
* learning-rate
### Default Configuration

The defult configuration, as defined, contains the parameters as follows:
* batch-size = 256
* max-epochs = 10
* learning-rate = 1e-05

This approach ensures a comprehensive analysis of the model's behavior under different parameter settings, 将根据训练测试训练的精度从以下方面进行评估，如非特别说明，则参数与Base一致。由于本门课注重提升模型运算速度方面，因此我还在训练的时候执行了`$ nivtop`指令
* Training with Different Batch Sizes: 64, 256, 1024, while other coefficient remains the same.
* Training with Different Epochs： 5, 10, 15
* Training with Different Learning Rates 1e-3,1e-5, 1e-7

结果图如下图所示：
![示例图片](example.jpg)

下面对问题进行分析：
1. What is the impact of varying batch sizes and why?
越大，收敛越慢，

2. What is the impact of varying maximum epoch number?

3. What is happening with a large learning and what is happening with a small learning rate and why? What is the relationship between learning rates and batch sizes?

## Training a 10x Model

```
./ch train jsc-toy jsc --max-epochs 10 --batch-size 256 --learning-rate 1e-03
./ch test jsc-toy jsc --load /home/super_monkey/mase_new/mase_output/Lab1-Task1/jsc-toy_classification_jsc_1X/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-toy_classification_jsc_1X/software --port 16006


./ch train jsc-toy10X jsc --max-epochs 10 --batch-size 256 --learning-rate 1e-03

./ch test jsc-toy10X jsc --load /home/super_monkey/mase_new/mase_output/Lab1-Task1/jsc-toy10X_classification_jsc_10X/software/training_ckpts/best.ckpt --load-type pl
tensorboard --logdir ../mase_output/Lab1-Task1/jsc-toy10X_classification_jsc_10X/software --port 16006
```

## Model Testing
./ch test jsc-tiny jsc --load ../mase_output/jsc-tiny_classification_jsc_Base/software/training_ckpts/best.ckpt --load-type pl

# TensorBoard and Checkpoint

## TensorBoard Command
/mase/machop$ tensorboard --logdir ../mase_output/jsc-tiny_classification_jsc_Base/software --port 16006

## Checkpoint Load Command
./ch test jsc-tiny jsc --load ../mase_output/jsc-tiny_classification_jsc_Base/software/training_ckpts/best.ckpt --load-type pl

# Other Model Training Commands


