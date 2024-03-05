import os
import wandb
import math
import time

from AlphaZero.train_alphazero import train
from AlphaZero.AlphaZeroPlayer.networks import create_simple_nn, create_normal_nn


def run_train(
    run_settings,
    model_params,
    selfplay_params,
    fit_params,
    test_params,
):
    budget = run_settings["budget"]
    n_cores = run_settings["n_cores"]
    starting_step = run_settings["starting_step"]
    model_name = run_settings["model_name"]
    multiprocessing = run_settings["multiprocessing"]
    learning_rate = model_params["learning_rate"]
    l1 = model_params["l1"]
    l2 = model_params["l2"]
    rounds_per_step = selfplay_params["rounds_per_step"]
    extra_noise_ratio = selfplay_params["extra_noise_ratio"]
    training_size_multiplier = fit_params["training_size_multiplier"]
    mcts_params = selfplay_params["mcts_params"]
    max_memory_multiplier = selfplay_params["max_memory_multiplier"]

    test_params["test_rounds"] = (
        math.ceil(test_params["test_rounds"] / n_cores) * n_cores
    )  # make sure rounds is divisible by n_cores and not devide to 0
    rounds_per_step = (
        math.ceil(rounds_per_step / n_cores) * n_cores
    )  # make sure rounds is divisible by n_cores and not devide to 0
    selfplay_params["rounds_per_step"] = rounds_per_step
    max_memory = rounds_per_step * 36 * max_memory_multiplier

    wandb.init(
        # set the wandb project where this run will be logged
        project=run_settings["project_name"],
        # name of the run
        name=model_name,
        # track hyperparameters and run metadata
        config={
            "run_settings": run_settings,
            "model_params": model_params,
            "selfplay_params": selfplay_params,
            "fit_params": fit_params,
            "test_params": test_params,
        },
    )
    print("starting training")
    print("run settings:", run_settings)
    print("model params:", model_params)
    print("selfplay params:", selfplay_params)
    print("fit params:", fit_params)
    print("test params:", test_params)

    if starting_step == 0:
        try:
            os.mkdir(f"Data/RL_data/{model_name}/")
        except:
            print("\n\n\n============model already exists============\n\n\n")
        if model_params["model_type"] == "simple":
            model = create_simple_nn(learning_rate, l1, l2)
        elif model_params["model_type"] == "normal":
            model = create_normal_nn(learning_rate, l1, l2)
        else:
            raise Exception("model type not recognized")

        model.save(f"Data/Models/{model_name}/{model_name}_0.h5")

    total_time, selfplay_time, training_time, testing_time = train(
        budget,
        starting_step,
        model_name,
        max_memory,
        multiprocessing,
        n_cores,
        rounds_per_step,
        training_size_multiplier,
        mcts_params,
        fit_params,
        test_params,
        extra_noise_ratio,
    )
    print("total time:", total_time)
    print("selfplay time:", selfplay_time)
    print("training time:", training_time)
    print("testing time:", testing_time)
    wandb.finish()


def main():
    try:
        n_cores = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        cluster = "cluster"
    except:
        n_cores = 10
        cluster = "local"
    print(f"Using {n_cores} cores on {cluster}")

    model_name = "final?"
    run_settings = {
        "project_name": "Thesis_test17",
        "model_name": model_name,
        "starting_step": 0,
        "budget": 3.85,  # hours
        "multiprocessing": True,
        "n_cores": n_cores,
    }
    model_params = {
        "model_type": "normal",
        "learning_rate": 0.01,
        "l1": 0.0,
        "l2": 0.0,
    }
    selfplay_params = {
        "rounds_per_step": 60,  # amount of selfplay rounds per step
        "max_memory_multiplier": 10,  # memory size = rounds_per_step * 36 * max_memory_multiplier
        "extra_noise_ratio": 0.1,  # when training extra_noise_ratio * mcts_steps is added to all visit counts
        "mcts_params": {
            "mcts_steps": 50,
            "n_of_sims": 0,
            "nn_scaler": 1,
            "ucb_c": 600,
        },
    }
    fit_params = {
        "training_size_multiplier": 1,  # training size = training_size_multiplier * rounds_per_step * 36
        "epochs": 1,
        "batch_size": 2048,
    }
    test_params = {
        "test_rounds": 10000,
        "test_frequency": 60,
        "mcts_params": {
            "mcts_steps": 10,
            "n_of_sims": 0,
            "nn_scaler": 1,
            "ucb_c": 50,
        },
    }
    run_train(
        run_settings,
        model_params,
        selfplay_params,
        fit_params,
        test_params,
    )


if __name__ == "__main__":
    starting_time = time.time()
    main()
    print("total run time:", time.time() - starting_time)
