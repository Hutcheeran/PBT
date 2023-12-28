import os
import tempfile

import torch
import torch.optim as optim

import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, test_func
from ray.tune.schedulers import PopulationBasedTraining
# codice sorgente per ray e robe varie qui sopra: https://github.com/ray-project/ray/tree/master/python/ray/tune/examples
# docs da cui é preso il codice https://docs.ray.io/en/latest/tune/examples/pbt_guide.html


def train_convnet(config):
    # Create our data loaders, model, and optmizer.
    step = 1
    train_loader, test_loader = get_data_loaders()
    model = ConvNet()
    # SDG = stochastic gradient descent. 
    # https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31
    # https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21#:~:text=Gradient%20descent%20(GD)%20is%20an,e.g.%20in%20a%20linear%20regression).
    # viene usato per arrivare al punto minimo di "errore" di una funzione
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("lr", 0.01), # lr = learning rate. Questo parametro puó essere tweakato, peró piú é alto, piú é la possibilitá che "salti" il punto minimo
        momentum=config.get("momentum", 0.9), # ottimizza ancora di piú l'algoritmo - https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d  
    )

    # If `train.get_checkpoint()` is populated, then we are resuming from a checkpoint.
    # Dalla documentazione: questo step é necessario per il Population Based Training (PBT)
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))

        # Load model state and iteration step from checkpoint.
        model.load_state_dict(checkpoint_dict["model_state_dict"])
        # Load optimizer state (needed since we're using momentum),
        # then set the `lr` and `momentum` according to the config.
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            if "lr" in config:
                param_group["lr"] = config["lr"]
            if "momentum" in config:
                param_group["momentum"] = config["momentum"]

        # Note: Make sure to increment the checkpointed step by 1 to get the current step.
        last_step = checkpoint_dict["step"]
        step = last_step + 1

    while True:
        ray.tune.examples.mnist_pytorch.train_func(model, optimizer, train_loader)
        acc = test_func(model, test_loader)
        metrics = {"mean_accuracy": acc, "lr": config["lr"]}

        # Every `checkpoint_interval` steps, checkpoint our current state.
        if step % config["checkpoint_interval"] == 0:
            with tempfile.TemporaryDirectory() as tmpdir:
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(tmpdir, "checkpoint.pt"),
                )
                train.report(metrics, checkpoint=Checkpoint.from_directory(tmpdir))
        else:
            train.report(metrics)

        step += 1

# PBT scheduler 
perturbation_interval = 5
scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=perturbation_interval,
    metric="mean_accuracy",
    mode="max",
    hyperparam_mutations={
        # distribution for resampling
        "lr": tune.uniform(0.0001, 1),
        # allow perturbations within this set of categorical values
        "momentum": [0.8, 0.9, 0.99],
    },
)

if ray.is_initialized():
    ray.shutdown()
ray.init()

tuner = tune.Tuner(
    train_convnet,
    run_config=train.RunConfig(
        name="pbt_test",
        # Stop when we've reached a threshold accuracy, or a maximum
        # training_iteration, whichever comes first
        stop={"mean_accuracy": 0.96, "training_iteration": 50},
        checkpoint_config=train.CheckpointConfig(
            checkpoint_score_attribute="mean_accuracy",
            num_to_keep=4,
        ),
        storage_path="/tmp/ray_results",
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=4,
    ),
    param_space={
        "lr": tune.uniform(0.001, 1),
        "momentum": tune.uniform(0.001, 1),
        "checkpoint_interval": perturbation_interval,
    },
)

results_grid = tuner.fit()

import matplotlib.pyplot as plt
import os

# Get the best trial result
best_result = results_grid.get_best_result(metric="mean_accuracy", mode="max")

# Print `path` where checkpoints are stored
print('Best result path:', best_result.path)

# Print the best trial `config` reported at the last iteration
# NOTE: This config is just what the trial ended up with at the last iteration.
# See the next section for replaying the entire history of configs.
print("Best final iteration hyperparameter config:\n", best_result.config)

# Plot the learning curve for the best trial
df = best_result.metrics_dataframe
# Deduplicate, since PBT might introduce duplicate data
df = df.drop_duplicates(subset="training_iteration", keep="last")
df.plot("training_iteration", "mean_accuracy")
plt.xlabel("Training Iterations")
plt.ylabel("Test Accuracy")
plt.show()