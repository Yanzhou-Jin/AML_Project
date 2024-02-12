# Lab 2:

In this task, following work has been done:
1. Set up a dataset
2. Set up a model
3. Generate a `MaseGraph` from the model
4. Run Analysis and Transform passes on the `MaseGraph`

Now consider the following problems:

## 1. Explain the functionality of `report_graph_analysis_pass` and its printed jargons such as `placeholder`, `get_attr` ... You might find the doc of [torch.fx](https://pytorch.org/docs/stable/fx.html) useful.

The functionality of `report_graph_analysis_pass` is generalizing analysis report for the
MaseGraph in the context of the torch.fx module. 
It will print the summary of the model in the terminal or external file.

Its parameters are listed as follows:

graph: a MaseGraph object.
pass_args: a dictionary which contains argument such as `file_name`, this is a optional parameter, if there is no file name, the
result will be print in the terminal.

Function:

First, the code extract `file_name` from the dictionary `pass_args`. Then, the function called `graph.fx_graph` to get information.
and initiallize a void string `buff` to store the information of the report.

Then, it emulated attribute `module` and collected the types of nodes, such as `placeholder`, `get_attr`, `call_function`, `call_method`, `call_module`and `output`
and store the information in the string `layer_types`.

    placeholder represents a function input. The name attribute specifies the name this value will take on. target is similarly the name of the argument. args holds either: 1) nothing, or 2) a single argument denoting the default parameter of the function input. kwargs is don’t-care. Placeholders correspond to the function parameters (e.g. x) in the graph printout.

    get_attr retrieves a parameter from the module hierarchy. name is similarly the name the result of the fetch is assigned to. target is the fully-qualified name of the parameter’s position in the module hierarchy. args and kwargs are don’t-care

    call_function applies a free function to some values. name is similarly the name of the value to assign to. target is the function to be applied. args and kwargs represent the arguments to the function, following the Python calling convention

    call_module applies a module in the module hierarchy’s forward() method to given arguments. name is as previous. target is the fully-qualified name of the module in the module hierarchy to call. args and kwargs represent the arguments to invoke the module on, excluding the self argument.

    call_method calls a method on a value. name is as similar. target is the string name of the method to apply to the self argument. args and kwargs represent the arguments to invoke the module on, including the self argument

    output contains the output of the traced function in its args[0] attribute. This corresponds to the “return” statement in the Graph printout.


Next, it counted the numbers of different type of nodes, and store them in the dictionary count.

Finally, the function appends the `counts` and `layer_types` into the buffer and print the buffer in the terminal or a external file 
according to if the name of file is given.

In conclusion, the function provides information about the summary of MaseGraph and the name of different types of node and layers 
as well as number of each type of node. It return the original `MaseGraph` and a empty parameter.


## 2. What are the functionalities of `profile_statistics_analysis_pass` and `report_node_meta_param_analysis_pass` respectively?

    profile_statistics_analysis_pass Functionality:

    This function is designed to perform profile statistics analysis on MaseGraph. The input parameters includes
    Arguments for the analysis pass, specifying various parameters for the analysis, such as the mode of analysis, 
    target nodes for weight and activation statistics, configurations for weight and activation statistics, 
    and input generator for activation profiling.

    First, it calls several graph iterator functions to perform different aspects of the analysis,
    such as registering statistic collections, profiling weights, profiling activations, 
    and computing and unregistering statistics.
    then, it returns a tuple containing the modified MaseGraph and an empty dictionary.


    report_node_meta_param_analysis_pass Functionality:
    This function is designed to perform meta parameter analysis on nodes in MaseGraph and generate a report.
    It contains MaseGraph itself and optional arguments for the analysis pass, including options like which parameters to include in the report and the path to save the report.
    First, It iterates through nodes in the graph and collects information about node name, Fx Node operation, Mase type, Mase operation, and additional parameters based on the specified options.
    It formats the collected information into a table and logs the analysis report using a logger. If a save path is provided, it also saves the report to a file.
    Finally, it Returns a tuple containing the analyzed MaseGraph and an empty dictionary.

    In summary:

    profile_statistics_analysis_pass focuses on profiling and analyzing statistics related to weights and activations in a neural network graph.
    report_node_meta_param_analysis_pass focuses on analyzing and reporting meta parameters associated with nodes in a neural network graph, including common, hardware, and software parameters.

## MASE OPs and MASE Types

MASE is designed to be a very high-level intermediate representation (IR), this is very different from the classic [LLVM IR](https://llvm.org/docs/LangRef.html) that you might be familiar with.

The following MASE Types are available:
(Note from Aaron: do we have a page somewhere that have summarized this?)


## A deeper dive into the quantisation transform

## 3. Explain why only 1 OP is changed after the `quantize_transform_pass` .
    pass_args by=type, therefore, only execute `graph_iterator_quantize_by_type`, and pass_args only have `linear`, thus, only the OP type=`linear` is changed.

## 4. Write some code to traverse both `mg` and `ori_mg`, check and comment on the nodes in these two graphs. You might find the source code for the implementation of `summarize_quantization_analysis_pass` useful.

The code is simple as follows

## 5. Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the `pass_args` for your custom network might be different if you have used more than the `Linear` layer in your network.




## 6. Write code to show and verify that the weights of these layers are indeed quantised. You might need to go through the source code of the implementation of the quantisation pass and also the implementation of the [Quantized Layers](../../machop/chop/passes/transforms/quantize/quantized_modules/linear.py) .

## The command line interface

The same flow can also be executed on the command line throw the `transform` action.

```bash
# make sure you have the same printout
pwd
# it should show
# your_dir/mase-tools/machop

# enter the following command
./ch transform --config configs/examples/jsc_toy_by_type.toml --task cls --cpu=0
```
## 7. Load your own pre-trained JSC network, and perform perform the quantisation using the command line interface.

## \[Optional] Write your own pass

Many examples of existing passes are in the [source code](../..//machop/chop/passes/__init__.py), the [test files](../../machop/test/passes) for these passes also contain useful information on helping you to understand how these passes are used.

Implement a pass to count the number of FLOPs (floating-point operations) and BitOPs (bit-wise operations).