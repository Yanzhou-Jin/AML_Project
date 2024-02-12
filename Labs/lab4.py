import sys

sys.path.append('/home/super_monkey/mase_new/machop')
import logging
import os
import copy
import numpy as np
from pathlib import Path
from pprint import pprint as pp

import torch 
from deepspeed.profiling.flops_profiler import get_model_profile

from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)

from chop.passes.graph.utils import (
    deepcopy_mase_graph,
    get_mase_op,
    get_mase_type,
    get_node_actual_target,
    get_parent_name,
    get_similar_node_actual_target,
    match_a_pattern,
    get_node_target_by_name,
)

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity, get_logger

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model
from chop.passes.graph import report_graph_analysis_pass
from torchmetrics.classification import MulticlassAccuracy

from torch import nn
from chop.passes.graph.utils import get_parent_name

set_logging_verbosity("info")

logger = get_logger("chop")
logger.setLevel(logging.INFO)

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"

data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}

###############




# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear  2
            nn.Linear(16, 16),  # linear  3
            nn.Linear(16, 5),  # linear  4
            nn.ReLU(5),  # 5
        )

    def forward(self, x):
        return self.seq_blocks(x)


model = JSC_Three_Linear_Layers()

# generate the mase graph and initialize node metadata
def reload_mg():
    mg = MaseGraph(model=model)
    mg, _ = init_metadata_analysis_pass(mg, None)
    return mg



###############################
def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)


def instantiate_relu(inplace):
    return nn.ReLU(inplace=inplace)


def redefine_linear_Relu_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        if name is not None:
            ori_module = graph.modules[node.target]
            if name == "inplace":
                inplace = ori_module.inplace
                inplace = inplace * config["channel_multiplier"]
                new_module = instantiate_relu(inplace)
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
                pass
            else:
                in_features = ori_module.in_features
                out_features = ori_module.out_features
                bias = ori_module.bias
                if name == "output_only":
                    out_features = out_features * config["channel_multiplier"]
                elif name == "both":
                    if type(config["channel_multiplier"])== int:
                        in_features = in_features * config["channel_multiplier"]
                        out_features = out_features * config["channel_multiplier"]
                    else:
                        in_features = in_features * config["channel_multiplier"][0]
                        out_features = out_features * config["channel_multiplier"][1]
                elif name == "input_only":
                    in_features = in_features * config["channel_multiplier"]
                new_module = instantiate_linear(in_features, out_features, bias)
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
    return graph, {}

pass_base_cfg = {
    "by": "name",
    "default": {"config": {"name": None}},
    "seq_blocks_1": {
        "config": {
            "name": "inplace",
            "channel_multiplier": 1,
        }
    },
    "seq_blocks_2": {
        "config": {
            "name": "both",
            "channel_multiplier": 1,
        }
    },
    "seq_blocks_3": {
        "config": {
            "name": "both",
            "channel_multiplier": 1,
        }
    },
    "seq_blocks_4": {
        "config": {
            "name": "both",
            "channel_multiplier": 1,
        }
    },
    "seq_blocks_5": {
        "config": {
            "name": "inplace",
            "channel_multiplier": 1,
        }
    },
}

# build a search space
mul=[1,2,4,8,16]

search_spaces = []
for x in (mul):
    for y in (mul):
        C_M = [1,(1,x),(x,y),(y,1),1]
        for j, cm in enumerate(C_M, start=1):
            pass_base_cfg[f'seq_blocks_{j}']['config']['channel_multiplier'] = cm
            # dict.copy() and dict(dict) only perform shallow copies
            # in fact, only primitive data types in python are doing implicit copy when a = b happens
        search_spaces.append(copy.deepcopy(pass_base_cfg))

metric = MulticlassAccuracy(num_classes=5)
# get_model_profile
num_batchs = 5
# This first loop is basically our search strategy,
# in this case, it is a simple brute force search
recorded_accs = []
recorded_loss = []
recorded_latencies = []
recorded_model_sizes = []
recorded_flops = []
for i, config in enumerate(search_spaces):
    print(i)
    mg=reload_mg()
    mg, _ = redefine_linear_Relu_transform_pass(
    graph=mg, pass_args={"config": config})

    j = 0
    flops, macs, params = get_model_profile(model=mg.model,print_profile=False, input_shape=tuple(dummy_in['x'].shape))
    recorded_model_sizes.append(params)
    recorded_flops.append(flops)

    # this is the inner loop, where we also call it as a runner.
    acc_avg, loss_avg = 0, 0
    accs, losses = [], []

    # Additional code for measuring latency
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        preds = mg.model(xs)
        loss = np.array(torch.nn.functional.cross_entropy(preds, ys))
        acc = np.array(metric(preds, ys))
        accs.append(acc)
        losses.append(loss)
        if j == 0:
            start_time.record()
        elif j == num_batchs:
            end_time.record()
            torch.cuda.synchronize()  # Wait for all GPU operations to finish
            latency = start_time.elapsed_time(end_time) / num_batchs
            recorded_latencies.append(latency)
        if j > num_batchs:
            break
        j += 1
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    recorded_accs.append(acc_avg)
    recorded_loss.append(loss_avg)

print('recorded_accs')
print(recorded_accs)
print('recorded_loss')
print(recorded_loss)
print('recorded_latencies')
print(recorded_latencies)
print(recorded_model_sizes)
print(recorded_flops)
