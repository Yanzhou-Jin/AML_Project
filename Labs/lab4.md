# Lab 4:
## 1. Can you edit your code, so that we can modify the above network to have layers expanded to double their sizes? Note: you will have to change the ReLU also.
代码被修改了，使得能够将每层的参数乘以2.更改的思路是增加`instantiate_relu`函数,使得ReLU函数也能被修改。同时在`redefine_linear_Relu_transform_pass`中增加对于`name`的判断，如果`name="inplace"`则进入对于ReLU的修改。反之则进入对`linear`层的修改。
需要特别声明的是`nn.ReLU`中只有一个可以更改的参数`inplace`，这是一个bool变量，表示是否在原地进行ReLU计算，只要大于0，对于结果都没有影响。因此我认为对于`ReLU`层，无需特别注明参数数目。

## 2. In lab3, we have implemented a grid search, can we use the grid search to search for the best channel multiplier value?
可以，我们只需要定义，改变每一层的

## 3. You may have noticed, one problem with the channel multiplier is that it scales all layers uniformly, ideally, we would like to be able to construct networks like the following，can you then design a search so that it can reach a network that can have this kind of structure?