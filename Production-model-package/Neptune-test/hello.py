from statistics import mode
import neptune.new as neptune

run = neptune.init(
    # metadata is saved in .neptune folder in the root dir
    # mode="offline", 
    # debug mode doesn't save any metadata
    mode="debug",
    project="lingjun/Remote-Community-Power-Usage", 
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2ZDQ3M2M3Yy0xYjNkLTRmMDYtOTYyNi00ODE4ZjgzZjA2MGMifQ=="
)  # your credentials

# Track metadata and hyperparameters of your Run
run["JIRA"] = "NPT-952"
run["algorithm"] = "ConvNet"

params = {"batch_size": 64, "dropout": 0.2, "learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params


# Track the training process by logging your training metrics
for epoch in range(100):
    run["train/accuracy"].log(epoch * 0.6)
    run["train/loss"].log(epoch * 0.4)

# Log the final results
run["f1_score"] = 0.66

# Stop logging to your Run
run.stop()