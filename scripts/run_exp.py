import pickle
from tqdm import tqdm
import os

from common import (load_and_register_tasks, get_task_info_filename,
                    get_measure_record_filename)
from dump_network_info import build_network_keys
from exp_util import get_hold_out_task, train, eval

import tvm
from tvm import auto_scheduler

targets = [
    "llvm -model=e5-2673",
    # "llvm -model=epyc-7452",
    # "llvm -model=graviton2",
    # "llvm -model=i7",
    # "llvm -model=platinum-8272",
    # "cuda -model=k80",
    # "cuda -model=t4"
]

train_ratio = 0.9
model_names = "mlp"
split_scheme = "within_task"
use_gpu = True

hold_out_network_keys = []
for batch_size in [1, 4, 8]:
    for image_size in [224, 240, 256]:
        for layer in [18, 50]:
            hold_out_network_keys.append((f'resnet_{layer}',
                                [(batch_size, 3, image_size, image_size)]))
for batch_size in [1, 4, 8]:
    for image_size in [224, 240, 256]:
        for name in ['mobilenet_v2', 'mobilenet_v3']:
            hold_out_network_keys.append((f'{name}',
                                [(batch_size, 3, image_size, image_size)]))
for batch_size in [1, 2, 4]:
    for seq_length in [64, 128, 256]:
        for scale in ['tiny', 'base']:
            hold_out_network_keys.append((f'bert_{scale}',
                                [(batch_size, seq_length)]))
for batch_size in [1, 2, 4]:
        for image_size in [299]:
            hold_out_network_keys.append((f'inception_v3',
                                [(batch_size, 3, image_size, image_size)]))

def main():

    for target_str in targets:
        target = tvm.target.Target(target_str)
        hardware = target.model
        if "llvm" in target_str:
            dataset_path = "/home/zhaowe58/tenset_dataset/dataset_cpu"
        else:
            dataset_path = "/home/zhaowe58/tenset_dataset/dataset_gpu"
        os.makedirs("data", exist_ok=True)
        train_dataset_filename = f"data/tenset_train_and_val_{hardware}.pkl"
        test_dataset_filename = f"data/tenset_test_{hardware}.pkl"

        # ======================= Dataset Preprocessing ===========================
        hold_out_workload_keys = get_hold_out_task(hold_out_network_keys, target, dataset_path)
        all_network_keys = build_network_keys()

        print("Load tasks...")
        print(f"target: {target}")
        train_tasks = []
        hold_tasks = []

        for network_key in tqdm(all_network_keys):
            task_info_filename = get_task_info_filename(network_key, target, dataset_path)
            tasks, _ = pickle.load(open(task_info_filename, "rb"))
            for task in tasks:
                if task.workload_key in hold_out_workload_keys:
                    hold_tasks.append(task)
                else:
                    train_tasks.append(task)

        train_files = [get_measure_record_filename(task, target, dataset_path) for task in train_tasks]
        test_files = [get_measure_record_filename(task, target, dataset_path) for task in hold_tasks]

        # ==================== End Dataset Preprocessing ==========================

        # ========================== Training =====================================
        print("Load all tasks...")
        load_and_register_tasks(dataset_path)

        train_dataset, val_dataset = auto_scheduler.dataset.make_train_dataset_from_log_file(
                                        train_files, train_dataset_filename, 10, train_ratio)
        models = train(train_dataset, val_dataset, hardware, model_names, use_gpu)

        del train_dataset
        del val_dataset

        # ======================= End Training ====================================


        # ======================== Evaluation =====================================

        test_testset = auto_scheduler.dataset.make_test_dataset_from_log_file(
                            test_files, test_dataset_filename, 10)
        eval(test_testset, model_names, models)

        # ======================== End Evaluation =================================


if __name__ == "__main__":
    main()