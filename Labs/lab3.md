# Lab 3:
## 1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.
在搜索过程中，除了accuracy and loss，还添加了一些额外的metrics用于评估模型的质量。在本实验中考虑的metrics的定义陈列如下：

Latency： 在本实验中，Latency被定义为模型运算所需的时间，即从输入数据到输出结果生成的时间间隔。较低的延迟意味着模型具有更好的实时性。

Model Size： 指模型所占用的存储空间大小，在本实验中由参数数目表示。较小的模型大小可以减小RAM的占用，并且有助于在IO bounded的平台上运行。

FLOPs： 模型进行推理或训练时所涉及的浮点运算数的数量。 FLOPs 越少，说明模型的运算量越少，在计算上更高效，计算速度越快。




## 2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It’s important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).
通过在search space上搜索，得到表格如下：

可以看到，XXX配置时，是最优解。

综合度量： 可以将额外的度量指标（如延迟、模型大小、FLOPs数量）与精度或损失指标进行综合，得到一个综合评分。例如，可以使用加权平均或多目标优化方法来将这些指标结合起来，以获得一个全面的评估结果。

性能-成本平衡： 可以将精度或损失作为性能度量，而将延迟、模型大小或FLOPs数量作为成本度量。然后，可以在性能和成本之间进行权衡，寻找最佳的性能-成本平衡点。例如，可以使用多目标优化方法来找到 Pareto 前沿，即无法在其中改善一个指标而不损害另一个指标的情况下找到的最佳性能-成本权衡。

通过结合精度/损失与额外的度量指标，我们可以获得更全面的模型评估，帮助决策者在模型选择和部署过程中做出更加综合和理性的决策。

可以注意到，在表格中，accuracy和loss表征了一致的性能，这是因为这是一个由简单模型计算的分类问题，并未出现明显的过拟合。在这种情况下，accuracy and loss 相互关联，可以视为相同的metric。对于分类问题而言，大多数情况下accuracy是被关注的指标，但是accuracy本身是个不可导的方程，因而用交叉熵作为损失函数来优化，但最终关心的还是准确度。

对于loss和accuracy不一致的情况，可能有以下原因：

1. 如果数据集的标签很不平均，比如80%的数据是Class 1，那么模型增加输出Class 1的比例，可能会让准确度上升，但loss的上升比例更大。本实验的数据集分布较为均匀，不属于这种情形
2. 如果模型的准确率很大，大多数Class的正确分类概率都接近1，此时如果出现一个错误，准确率的降低会很少，但交叉熵可能会非常高。显然本实验不属于这种情形。
3. 如果模型过拟合，则其在训练集上表现的很好，但是在测试数据上表现很差，此时模型的loss会比较小而accuracy会很低。本模型复杂度非常低，训练数据也较为充足，没有发生过拟合








## 3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.
在Mase中没有集成brute-force的策略，但是可以在`optuna.py`中加入这个方法。参考了https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.BruteForceSampler.html[brus] 网页。只需要在sampler_map函数中增加如下两行即可完成。
```
            case "brute-force":
                sampler = optuna.samplers.BruteForceSampler()
```
## 4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods
"sample efficiency" 指的是在超参数优化过程中所需的样本数量与最终达到的性能之间的关系。换句话说，它衡量了算法在给定样本数量的情况下的性能提升速度。

Brute-force Search： Brute-force 搜索方法简单直接，它会尝试所有可能的超参数组合，因此它的样本效率通常较低。它的优点是能够保证找到全局最优解（如果搜索空间足够大），但代价是需要大量的计算资源和时间。因此，在大型超参数空间中，brute-force 搜索可能不太实用，因为它的样本效率较低。

TPE-based Search： TPE（Tree-structured Parzen Estimator）是一种基于贝叶斯优化的方法，它通过构建一个概率模型来估计不同超参数配置的性能，并选择最有希望的配置进行下一次评估。与brute-force搜索相比，TPE-based搜索方法通常更加高效，因为它能够在搜索过程中根据之前的结果动态调整搜索空间，更有效地探索有可能产生更好性能的超参数配置。这种动态调整的特性使得TPE-based搜索在给定样本数量的情况下通常能够找到更好的超参数配置。

综上所述，TPE-based搜索方法通常比brute-force搜索方法具有更高的样本效率，因为它能够更有效地探索超参数空间，并在给定的样本数量下获得更好的性能提升。然而，值得注意的是，TPE-based搜索仍然受到初始样本数量和超参数空间的限制，因此在特定情况下可能会表现不佳。
