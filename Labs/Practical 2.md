# Lab 3:
## 1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.
<!-- 在搜索过程中，除了accuracy and loss，还添加了一些额外的metrics用于评估模型的质量。在本实验中考虑的metrics的定义陈列如下：

Latency： 在本实验中，Latency被定义为模型运算所需的时间，即从输入数据到输出结果生成的时间间隔。较低的延迟意味着模型具有更好的实时性。

Model Size： 指模型所占用的存储空间大小，在本实验中由参数数目表示。较小的模型大小可以减小RAM的占用，并且有助于在IO bounded的平台上运行。

FLOPs： 模型进行推理或训练时所涉及的浮点运算数的数量。 FLOPs 越少，说明模型的运算量越少，在计算上更高效，计算速度越快。 -->
During the search process, in addition to accuracy and loss, some additional metrics are added to evaluate the quality of the model. The definitions of metrics considered in this task are listed below:

* Latency: In this task, Latency is defined as the time required for model operation, that is, the time interval from input data to output result generation of each batch. Lower latency means better real-time performance of the model.
```
    ...
        elif j == num_batchs:
            end_time.record()
            torch.cuda.synchronize()  # Wait for all GPU operations to finish
            latency = start_time.elapsed_time(end_time) / num_batchs
    ...
```
* Model Size: refers to the size of the storage space occupied by the model, which is represented by the number of parameters in this experiment. Smaller model sizes reduce RAM usage and help run on IO bounded platforms. This information is collected together with FLOPS using the function `get_model_profile` , which is from the package of deepspeed.

* FLOPs: The number of floating point operations involved in a model's inference or training. The fewer FLOPs, the less computational complexity the model requires, the more computationally efficient it is, and the faster the calculation speed.
```
    flops, macs, params = get_model_profile(model=mg.model, input_shape=tuple(dummy_in['x'].shape))
```


## 2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It’s important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).

By implementing such metrics searching on SEARCH SPACE, and obtained the table below:

| Search History    | Recorded Accuracies | Recorded Losses | Recorded Latencies | Recorded Model Sizes | Recorded FLOPS |
|-------------------|---------------------|-----------------|--------------------|----------------------|-----------------|
| [(16, 8), (16, 8)] | 0.47261905670166   | 1.2069          | 1.33173122406006  | 117                  | 1.92 K          |
| [(16, 8), (8, 6)]  | 0.316071437937873  | 1.4612          | 1.10885763168335  | 117                  | 1.92 K          |
| [(16, 8), (8, 4)]  | 0.376190483570099  | 1.4007          | 1.10261764526367  | 117                  | 1.92 K          |
| [(16, 8), (4, 2)]  | 0.34523810765573   | 1.4049          | 0.975513553619385 | 117                  | 1.92 K          |
| [(8, 6), (16, 8)]  | 0.428571445601327  | 1.3946          | 0.965023994445801 | 117                  | 1.92 K          |
| [(8, 6), (8, 6)]   | 0.390476199133055  | 1.3525          | 0.981292819976807 | 117                  | 1.92 K          |
| [(8, 6), (8, 4)]   | 0.446428579943521  | 1.3556          | 0.997523212432861 | 117                  | 1.92 K          |
| [(8, 6), (4, 2)]   | 0.445238104888371  | 1.3406          | 0.990246391296387 | 117                  | 1.92 K          |
| [(8, 4), (16, 8)]  | 0.466666681425912  | 1.2558          | 0.975551986694336 | 117                  | 1.92 K          |
| **[(8, 4), (8, 6)]**   | **0.533333337732724**  |**1.2949**           | **1.006764793396**    | **117**                  | **1.92 K**          |
| [(8, 4), (8, 4)]   | 0.523412704467773  | 1.3038          | 1.02436475753784  | 117                  | 1.92 K          |
| [(8, 4), (4, 2)]   | 0.383928579943521  | 1.3025          | 1.22784004211426  | 117                  | 1.92 K          |
| [(4, 2), (16, 8)]  | 0.516071438789368  | 1.3339          | 0.942265605926514 | 117                  | 1.92 K          |
| [(4, 2), (8, 6)]   | 0.515476199133056  | 1.3096          | 0.992268753051758 | 117                  | 1.92 K          |
| [(4, 2), (8, 4)]   | 0.514880959476743  | 1.3370          | 1.11617279052734  | 117                  | 1.92 K          |
| [(4, 2), (4, 2)]   | 0.490476195301328  | 1.2638          | 1.11920003890991  | 117                  | 1.92 K          |


可以看到，[(8, 4), (8, 6)]配置时准确率最高0.533333337732724，是最优解。


<!-- 同时可以注意到，在表格中，accuracy和loss表征了一致的性能，这是因为这是一个由简单模型计算的分类问题，并未出现明显的过拟合。在这种情况下，accuracy and loss 相互关联，可以视为相同的metric。对于分类问题而言，大多数情况下accuracy是被关注的指标，但是accuracy本身是个不可导的方程，因而用交叉熵作为损失函数来优化，但最终关心的还是准确度。

对于loss和accuracy不一致的情况，可能有以下原因：

1. 如果数据集的标签很不平均，比如80%的数据是Class 1，那么模型增加输出Class 1的比例，可能会让准确度上升，但loss的上升比例更大。本实验的数据集分布较为均匀，不属于这种情形
2. 如果模型的准确率很大，大多数Class的正确分类概率都接近1，此时如果出现一个错误，准确率的降低会很少，但交叉熵可能会非常高。显然本实验不属于这种情形。
3. 如果模型过拟合，则其在训练集上表现的很好，但是在测试数据上表现很差，此时模型的loss会比较小而accuracy会很低。本模型复杂度非常低，训练数据也较为充足，没有发生过拟合 -->

At the same time, it can be noted that in the table, accuracy and loss represent consistent performance. This is because this is a classification problem calculated by a simple model and there is no obvious overfitting. In this case, accuracy and loss are related to each other and can be regarded as the same metric. For classification problems, accuracy is the index of concern in most cases, but accuracy itself is a non-differentiable equation, so cross-entropy is used as the loss function for optimization, but the final concern is accuracy.

For the inconsistency between loss and accuracy, there may be the following reasons:

1. If the labels of the data set are very uneven, for example, 80% of the data is Class 1, then increasing the proportion of Class 1 output by the model may increase the accuracy, but the increase in loss will be greater. The data set of this experiment is relatively evenly distributed and does not belong to this situation.
2. If the accuracy of the model is very high, the correct classification probability of most Classes is close to 1. If an error occurs at this time, the accuracy will be reduced very little, but the cross entropy may be very high. Obviously this is not the case in this experiment.
3. If the model is overfitted, it performs very well on the training set, but performs poorly on the test data. At this time, the loss of the model will be relatively small and the accuracy will be very low. The complexity of this model is very low, the training data is sufficient, and there is no overfitting.






## 3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.
<!-- 在Mase中没有集成brute-force的策略，但是可以在`optuna.py`中加入这个方法。参考了https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.BruteForceSampler.html 网页。只需要在sampler_map函数中增加如下两行即可完成。
 -->
 In Mase, there isn't an integrated brute-force strategy, but such a method can be added to `optuna.py`. We can refer to the documentation page at https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.BruteForceSampler.html. To accomplish this, simply add the following two lines to the `sampler_map` function:
```
            case "brute-force":
                sampler = optuna.samplers.BruteForceSampler()
```
and the result are as follows:

Best trial(s):

|    | number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----|--------|------------------------------------|---------------------------------------------------|----------------------------------------------|
|  0 | 4      | {'loss': 1.436, 'accuracy': 0.476} | {'average_bitwidth': 4.0, 'memory_density': 8.0}  | {'accuracy': 0.476, 'average_bitwidth': 0.8} |
|  1 | 6      | {'loss': 1.413, 'accuracy': 0.486} | {'average_bitwidth': 16.0, 'memory_density': 2.0} | {'accuracy': 0.486, 'average_bitwidth': 3.2} |
|  2 | 18     | {'loss': 1.408, 'accuracy': 0.477} | {'average_bitwidth': 8.0, 'memory_density': 4.0}  | {'accuracy': 0.477, 'average_bitwidth': 1.6} |



## 4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods
<!-- "sample efficiency" 指的是在超参数优化过程中所需的样本数量与最终达到的性能之间的关系。换句话说，它衡量了算法在给定样本数量的情况下的性能提升速度。

Brute-force Search： Brute-force 搜索方法简单直接，它会尝试所有可能的超参数组合，因此它的样本效率通常较低。它的优点是能够保证找到全局最优解（如果搜索空间足够大），但代价是需要大量的计算资源和时间。因此，在大型超参数空间中，brute-force 搜索可能不太实用，因为它的样本效率较低。

TPE-based Search： TPE（Tree-structured Parzen Estimator）是一种基于贝叶斯优化的方法，它通过构建一个概率模型来估计不同超参数配置的性能，并选择最有希望的配置进行下一次评估。与brute-force搜索相比，TPE-based搜索方法通常更加高效，因为它能够在搜索过程中根据之前的结果动态调整搜索空间，更有效地探索有可能产生更好性能的超参数配置。这种动态调整的特性使得TPE-based搜索在给定样本数量的情况下通常能够找到更好的超参数配置。

综上所述，TPE-based搜索方法通常比brute-force搜索方法具有更高的样本效率，因为它能够更有效地探索超参数空间，并在给定的样本数量下获得更好的性能提升。然而，值得注意的是，TPE-based搜索仍然受到初始样本数量和超参数空间的限制，因此在特定情况下可能会表现不佳。 -->
"sample efficiency" refers to the relationship between the number of samples required during hyperparameter optimization and the final performance achieved. In other words, it measures how quickly an algorithm improves performance for a given number of samples.

Brute-force Search: The Brute-force search method is simple and straightforward. It tries all possible hyperparameter combinations, so its sample efficiency is usually lower. Its advantage is that it is guaranteed to find the global optimal solution (if the search space is large enough), but the cost is that it requires a lot of computing resources and time. Therefore, in large hyperparameter spaces, brute-force search may be less practical because it is less sample efficient.

TPE-based Search: TPE (Tree-structured Parzen Estimator) is a method based on Bayesian optimization that estimates the performance of different hyperparameter configurations by building a probabilistic model and selects the most promising configuration for the next evaluation. . Compared with brute-force search, the TPE-based search method is generally more efficient because it is able to dynamically adjust the search space based on previous results during the search process and more effectively explore hyperparameter configurations that are likely to produce better performance. This dynamic adjustment feature allows TPE-based search to generally find better hyperparameter configurations for a given number of samples.

In summary, the TPE-based search method is generally more sample efficient than the brute-force search method because it is able to explore the hyperparameter space more efficiently and obtain better performance improvement for a given number of samples. However, it is worth noting that TPE-based search is still limited by the initial number of samples and hyperparameter space, and thus may perform poorly in specific situations.

Best trial(s):
|    | number | software_metrics                   | hardware_metrics                                 | scaled_metrics                               |
|----|--------|------------------------------------|--------------------------------------------------|----------------------------------------------|
|  0 | 1      | {'loss': 1.429, 'accuracy': 0.485} | {'average_bitwidth': 8.0, 'memory_density': 4.0} | {'accuracy': 0.485, 'average_bitwidth': 1.6} |
|  1 | 17     | {'loss': 1.439, 'accuracy': 0.427} | {'average_bitwidth': 4.0, 'memory_density': 8.0} | {'accuracy': 0.427, 'average_bitwidth': 0.8} |

# Lab 4:
## 1. Can you edit your code, so that we can modify the above network to have layers expanded to double their sizes? Note: you will have to change the ReLU also.
<!-- 代码被修改了，使得能够将每层的参数乘以2.更改的思路是增加`instantiate_relu`函数,使得ReLU函数也能被修改。同时在`redefine_linear_Relu_transform_pass`中增加对于`name`的判断，如果`name="inplace"`则进入对于ReLU的修改。反之则进入对`linear`层的修改。
需要特别声明的是`nn.ReLU`中只有一个可以更改的参数`inplace`，这是一个bool变量，表示是否在原地进行ReLU计算，只要大于0，对于结果都没有影响。因此我认为对于`ReLU`层，无需特别注明参数数目。 -->

The code has been modified to double the parameters of each layer (including ReLU). The modification method includes adding a function called `instantiate_relu` to be able to modify the `ReLU` function. Apart from that, a check of the name parameter is added to the function`redefine_linear_Relu_transform_pass`. If `name` equals to `"inplace"`, the `ReLU` layer will be modified; otherwise, the linear layer is modified.

It is worth noting that there is only one changeable parameter inplace in `nn.ReLU()`, which is a Bool variable indicating whether to perform the `ReLU` operation in origianl place. Thus, as long as it is greater than 0, the result is not affected. Therefore, I think there is no need to specify the number of parameters when making modifications to the ReLU layer.

## 2. In lab3, we have implemented a grid search, can we use the grid search to search for the best channel multiplier value?
<!-- 可以，我们只需要定义`search_spaces`，分别改变每一层的`channel_multiplier`即可。像第三问一样，首先建立一个list,存储所有可能的乘数。然后建立包含所有乘数的`search_spaces`。需要注意的是由于三个线性层第一个只改了输出层，第二个将输入和输出scales uniformly，第三个只改变了输入层，因此这三层网络的乘数应该一样。所以严格意义上，这个过程是从list中一一枚举。
结果如下： -->
Yes, we only need to define `search_spaces` and change the `channel_multiplier` of each layer separately. Like the third question, first create a list to store all possible multipliers. Then we create `search_spaces` containing all multipliers. It should be noted that since the first of those three linear layers only affects the output layer, the second scales both the input and output uniformly, and the third only modifies the input layer, the multipliers for these three layers of networks should be the same. Therefore, to be extact, this process involves enumerating one by one from the list.

The outcome of the search process are as follows.

| Search multipliers | Recorded Accuracies | Recorded Losses | Recorded Latencies | Recorded Model Sizes | Recorded FLOPS |
|----------------|---------------------|-----------------|--------------------|----------------------|-----------------|
| 1            | 0.183333335178239   | 1.6014          | 0.63633918762207  | 3.01 K               | 47.27 K         |
| 2            | 0.261111118963786   | 1.6086          | 0.609401607513428 | 10.09 K              | 159.66 K        |
| 4            | 0.107738098927907   | 1.6197          | 0.62485761642456  | 36.52 K              | 581.03 K        |
| 8            | 0.050000000212874   | 1.6231          | 0.705734395980835 | 138.53 K             | 2.21 M          |
| 16           | 0.188690478248256   | 1.6157          | 0.776793622970581 | 539.17 K             | 8.61 M          |
| 64           | 0.14880952664784    | 1.608           | 2.37203197479248  | 8.45 M               | 135.12 M        |

<!-- 根据上表可知，乘以2时，准确率最高并且模型参数数目较少，FLOPS较小。因此乘以2是最佳选项 -->
According to the table above, when multiplied by 2, the accuracy is the highest and the number of model parameters is small, resulting in smaller FLOPS. So multiplying by 2 is the best option.

## 3. You may have noticed, one problem with the channel multiplier is that it scales all layers uniformly, ideally, we would like to be able to construct networks like the following, can you then design a search so that it can reach a network that can have this kind of structure?
<!-- 可以，只需要设计程序，将`name:"both"`中的 `channel_multiplier`改为(a,b)的形式即可。这样就可以给input fearures 和output features的数目乘以两个不同的值。 -->
Indeed, it is possible to modify the function by adjust the `channel_multiplier` parameter under the case `name: "both"` to take the form of (a, b). 
This configuration constructed the multiplication of the number of input features and output features by two different values, therefore, makes the network transformations more flexible.
```
"seq_blocks_6": {
        "config": {
            "name": "both",
            "channel_multiplier": (2,4),
        }
    },
```
<!-- 考虑到兼容问题，需要修改`redefine_linear_Relu_transform_pass`中name的判断逻辑，通过判断`channel_multiplier`的数据类型，实现对于uniform scales和 nonuniform scales指令的自动识别。 -->
In light of the compatibility concerns, it is necessary to modify the decision logic of name in `redefine_linear_Relu_transform_pass`, and realize automatic recognition of uniform and nonuniform scales instructions by evaluating the data type of `channel_multiplier`.
```
elif name == "both":
                    if type(config["channel_multiplier"])== int:
                        in_features = in_features * config["channel_multiplier"]
                        out_features = out_features * config["channel_multiplier"]
                    else:
                        in_features = in_features * config["channel_multiplier"][0]
                        out_features = out_features * config["channel_multiplier"][1]
```
<!-- 同时，还需要考虑模型正确性的问题，需要保证上一层的输入和下一层的输出参数数目相同，否则模型不能够正常运行。这样做的思路是首先建立一个list，例如`a=[(1,x),(x,y),(y,1)]`，将相邻两个层的输入和输出定义为相同的参数，在搜索时只需要更改这些参数就行(本例中改变x和y的值即可)，这样就可以保证模型的正确性。
搜索的结果如下表所示: -->
Meanwhile, the integrity of the model must be taken into account. It is necessary to ensure that the number of input parameters of the previous layer and the output parameters of the next layer are the same, otherwise the model cannot run normally. The idea is to first create a list, such as `a=[(1,x),(x,y),(y,1)]`, where the input and output parameters of two adjacent layers are defined as identical variables x and y. Consequently, when searching, only these parameters need to be modified (in this case, altering the values of x and y), thereby safeguarding the accuracy and functionality of the model.

The search results are shown in the following table:

| Search History                     | Recorded Accuracies | Recorded Losses | Recorded Latencies | Recorded Model Sizes | Recorded FLOPS |
|------------------------------------|---------------------|------------------|---------------------|----------------------|-----------------|
| [1, (1, 1), (1, 1), (1, 1), 1]     | 0.227976194449833   | 1.605            | 0.556281614303589  | 3.01 K               | 47.27 K         |
| **[1, (1, 1), (1, 2), (2, 1), 1]**     | **0.251349210739136**   | **1.6107**           | **0.578835201263428**  | **5.45 K**               | **85.67 K**         |
| [1, (1, 1), (1, 4), (4, 1), 1]     | 0.164285718330315   | 1.6147           | 0.608767986297607  | 10.31 K              | 162.47 K        |
| [1, (1, 1), (1, 8), (8, 1), 1]     | 0.200000000851495   | 1.6066           | 0.62422399520874   | 20.04 K              | 316.07 K        |
| [1, (1, 1), (1, 16), (16, 1), 1]   | 0.228571429848671   | 1.6086           | 0.707788801193237  | 39.49 K              | 623.27 K        |
| [1, (1, 2), (2, 1), (1, 1), 1]     | 0.179761906819684   | 1.6122           | 0.623545598983765  | 5.61 K               | 88.49 K         |
| [1, (1, 2), (2, 2), (2, 1), 1]     | 0.164285715137209   | 1.6141           | 0.575072002410889  | 10.09 K              | 159.66 K        |
| [1, (1, 2), (2, 4), (4, 1), 1]     | 0.248809527073588   | 1.6076           | 0.689766407012939  | 19.05 K              | 301.99 K        |
| [1, (1, 2), (2, 8), (8, 1), 1]     | 0.327380959476743   | 1.6048           | 0.719059181213379  | 36.97 K              | 586.66 K        |
| [1, (1, 2), (2, 16), (16, 1), 1]   | 0.211904766304152   | 1.6118           | 0.75284481048584   | 72.81 K              | 1.16 M          |
| [1, (1, 4), (4, 1), (1, 1), 1]     | 0.20000000510897    | 1.6012           | 0.672563219070435  | 10.79 K              | 170.92 K        |
| [1, (1, 4), (4, 2), (2, 1), 1]     | 0.127976194024086   | 1.6100           | 0.683251190185547  | 19.36 K              | 307.62 K        |
| [1, (1, 4), (4, 4), (4, 1), 1]     | 0.145238096160548   | 1.6035           | 0.693862390518188  | 36.52 K              | 581.03 K        |
| [1, (1, 4), (4, 8), (8, 1), 1]     | 0.204761907458305   | 1.6091           | 0.695679998397827  | 70.82 K              | 1.13 M          |
| [1, (1, 4), (4, 16), (16, 1), 1]   | 0.164285714072841   | 1.6127           | 0.753260803222656  | 139.43 K             | 2.22 M          |
| [1, (1, 8), (8, 1), (1, 1), 1]     | 0.221428575260299   | 1.6008           | 0.731129598617554  | 21.16 K              | 335.78 K        |
| [1, (1, 8), (8, 2), (2, 1), 1]     | 0.159523812787873   | 1.6217           | 0.689951992034912  | 37.92 K              | 603.56 K        |
| [1, (1, 8), (8, 4), (4, 1), 1]     | 0.11547619423696    | 1.6101           | 0.689267206192017  | 71.46 K              | 1.14 M          |
| [1, (1, 8), (8, 8), (8, 1), 1]     | 0.15000000170299    | 1.6145           | 0.767404794692993  | 138.53 K             | 2.21 M          |
| [1, (1, 8), (8, 16), (16, 1), 1]   | 0.159523811723505   | 1.6181           | 0.732582378387451  | 272.68 K             | 4.35 M          |
| [1, (1, 16), (16, 1), (1, 1), 1]   | 0.132142862038953   | 1.5975           | 0.783155202865601  | 41.89 K              | 665.51 K        |
| [1, (1, 16), (16, 2), (2, 1), 1]   | 0.042857143495764   | 1.6160           | 0.771737623214722  | 75.05 K              | 1.2 M           |
| [1, (1, 16), (16, 4), (4, 1), 1]   | 0.185714289546013   | 1.6086           | 0.732959985733032  | 141.35 K             | 2.26 M          |
| [1, (1, 16), (16, 8), (8, 1), 1]   | 0.235714288694518   | 1.5942           | 0.710342407226563  | 273.96 K             | 4.37 M          |
| [1, (1, 16), (16, 16), (16, 1), 1] | 0.227619051933289   | 1.5767           | 0.733241605758667  | 539.17 K             | 8.61 M          |

<!-- 可以看出，当配置为[1, (1, 1), (1, 2), (2, 1), 1]时，accuracy达到最高值0.251349210739136	。此时模型的大小较小，表现最好。 -->
It can be seen that when the configuration is [1, (1, 1), (1, 2), (2, 1), 1], the accuracy reaches the highest value of $0.251349210739136$. This is when the model size is smaller and performs best.

## 4. Integrate the search to the chop flow, so we can run it from the command line.
<!-- 设计了代码将上述搜索集成到了chop flow中，使得代码能够在命令行中运行。搜索的任务分为四部分：
首先在graph.py中新建`GraphSearchSpaceMixedPrecisionPTQmy`函数，用于执行search操作。
其次，将`redefine_linear_Relu_transform_pass`集成到pass目录中，使得其能够被`GraphSearchSpaceMixedPrecisionPTQmy`调用。在`redefine_linear_Relu_transform_pass`中设置`safe_lock`，确保修改参数后的模型上一层的输出参数数目等于下一层的参数输入数目，确保模型能够正确运行。

输出结果如下所示： -->
The code was designed to integrate the above search into the chop flow, so that the code can be run on the command line. The search task is divided into four parts:
First, create a new `GraphSearchSpaceMixedPrecisionPTQmy` function in graph.py to perform search operations.
Secondly, integrate `redefine_linear_Relu_transform_pass` into the pass directory so that it can be called by `GraphSearchSpaceMixedPrecisionPTQmy`. Set `safe_lock` in `redefine_linear_Relu_transform_pass` to ensure that the number of output parameters of the upper layer of the model after modifying parameters is equal to the number of parameter inputs of the next layer to ensure that the model can run correctly.

The output is as follows:


Best trial(s):

|  | number | software_metrics                        | hardware_metrics                          | scaled_metrics  |
|------|--------|----------------------------------------|-------------------------------------------|-----------------|
| 0    | 8      | {'loss': 1.609, 'accuracy': 0.22}      | {'average_bitwidth': 32, 'memory_density': 1.0} | {'accuracy': 0.22} |



