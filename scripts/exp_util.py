from tqdm import tqdm
import pickle
import os

from common import (get_task_info_filename)
from train_model import (evaluate_model, make_model)

from tvm.auto_scheduler.utils import to_str_round

def get_hold_out_task(network_keys, target, dataset_path):
    hold_out_workload_keys = set()
    print("hold out...")
    for network_key in tqdm(network_keys):
        # Read tasks of the network
        task_info_filename = get_task_info_filename(network_key, target, dataset_path=dataset_path)
        tasks, _ = pickle.load(open(task_info_filename, "rb"))
        for task in tasks:
            if task.workload_key not in hold_out_workload_keys:
                hold_out_workload_keys.add(task.workload_key)

    return hold_out_workload_keys

def train(train_set, valid_set, hardware, model_names, use_gpu):
    
    print("Train set: %d. Task 0 = %s" % (len(train_set), train_set.tasks()[0]))
    print("Valid set:  %d. Task 0 = %s" % (len(valid_set), valid_set.tasks()[0]))

    # Make models
    names = model_names.split("@")
    models = []
    for name in names:
        models.append(make_model(name, use_gpu))

    for name, model in zip(names, models):
        # Train the model
        os.makedirs("models", exist_ok=True)
        model_filename = f"models/{name}_{hardware}.pkl"

        if os.path.exists(model_filename):
            print(f"{model_filename} already exists. Skip training")
        else:
            model.fit_base(train_set, valid_set=valid_set)
            # print("Saving model to %s" % model_filename)
            # model.save(model_filename)

    return models

def eval(testset, model_names, models):
    print("Test set:  %d. Task 0 = %s" % (len(testset), testset.tasks()[0]))

    names = model_names.split("@")

    eval_results = []
    for name, model in zip(names, models):
        eval_res = evaluate_model(model, testset)
        print(name, to_str_round(eval_res))
        eval_results.append(eval_res)

    # Print evaluation results
    for i in range(len(models)):
        print("-" * 60)
        print("Model: %s" % names[i])
        for key, val in eval_results[i].items():
            print("%s: %.4f" % (key, val))