import argparse
import logging
import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt

import tvm
from tvm import relay, auto_scheduler
import tvm.contrib.graph_runtime as runtime

from dump_network_info import get_network_with_key


def get_network(network_args):
    name, batch_size = network_args['network'], network_args['batch_size']
    if name in ['resnet_18', 'resnet_50', 'mobilenet_v2', 'mobilenet_v3',
                'wide_resnet_50', 'resnext_50', 'densenet_121']:
        network_key = (name, [(batch_size, 3, 224, 224)])
    elif name in ['inception_v3']:
        network_key = (name, [(batch_size, 3, 299, 299)])
    elif name in ['bert_tiny', 'bert_base', 'bert_medium', 'bert_large']:
        network_key = (name, [(batch_size, 128)])
    elif name == 'dcgan':
        network_key = (name, [(batch_size, 3, 64, 64)])
    else:
        raise ValueError("Invalid network: " + name)

    return get_network_with_key(network_key)


def make_plot(network_args, log_file, target):
    mean_inf_time = []
    mod, params, inputs = get_network(network_args)
    for i in range(0, 100):
        # Build module
        print(f"each task is measured {i} time")
        with auto_scheduler.ApplyHistoryBest(log_file, n_line_per_task=i):
            with tvm.transform.PassContext(
                    opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(mod, target=target, params=params)
        ctx = tvm.context(str(target), 0)
        module = runtime.GraphModule(lib["default"](ctx))

        # Feed input data
        for name, shape, dtype in inputs:
            data_np = np.random.uniform(size=shape).astype(dtype)
            module.set_input(name, data_np)

        # Evaluate
        ftimer = module.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=3)
        prof_res = np.array(ftimer().results)
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res) * 1000, np.std(prof_res) * 1000))
    
        mean_inf_time.append(np.mean(prof_res) * 1000)

    plt.plot(list(range(1, 100)), mean_inf_time[1:])
    plt.savefig(f"{network_args['network']}_trials_vs_latency.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target", type=str, default='llvm -mcpu=core-avx2')
    parser.add_argument("--log-file", type=str)
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    args= parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    
    target = tvm.target.Target(args.target)
    if target.model == "unknown":
        log_file = args.log_file or "%s-B%d-%s.json" % (args.network, args.batch_size,
                                                        target.kind)
    else:
        log_file = args.log_file or "%s-B%d-%s-%s.json" % (args.network, args.batch_size,
                                                           target.kind, target.model)
    network_args = {
        "network": args.network,
        "batch_size": args.batch_size,
    }
    make_plot(network_args, log_file, target)
