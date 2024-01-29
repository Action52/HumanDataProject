from base_model import WandbKerasModel

# Initialize a WandbKerasModel without a model (or with a dummy model)
wandb_model = WandbKerasModel(model=None, project_name='mnist_project', entity='leonvillapun')

# Load the model from the artifact
wandb_model.load_model_from_artifact('leonvillapun/mnist_project/model-mnist_simple_nn:latest')

# Now, wandb_model.model contains the loaded model
print(wandb_model.model.summary())
