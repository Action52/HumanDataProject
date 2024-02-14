import pickle
import wandb

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from wandb.keras import WandbCallback


class WandbManager:
    def __init__(self, project_name, entity=None, config=None):
        """
        Initialize wandb setup and connect to a specific project.

        :param project_name: Name of the wandb project to connect to.
        :param entity: Optional. The team or user who owns the project.
        :param config: Optional. Configuration parameters for the experiment.
        """
        self.project_name = project_name
        self.entity = entity
        self.config = config or {}

    def start_experiment(self, experiment_name, config_updates=None):
        """
        Start a new experiment/run in wandb.

        :param experiment_name: Unique name for the experiment/run.
        :param config_updates: Optional. Dictionary of configuration parameters to update.
        """
        # Update the experiment configuration if provided
        if config_updates:
            self.config.update(config_updates)

        # Initialize the wandb run
        wandb.init(
            project=self.project_name,
            entity=self.entity,
            config=self.config,
            name=experiment_name,
            reinit=True,
        )

    def log_metrics(self, metrics):
        """
        Log metrics for the current experiment.

        :param metrics: Dictionary of metric names and their values.
        """
        wandb.log(metrics)

    def log_model(self, model, model_name="model"):
        """
        Log model architecture and parameters.

        :param model: The model to log.
        :param model_name: Optional. A name for the logged model.
        """
        wandb.watch(model, log="all")

        # Optionally, save and log the model explicitly if needed
        # model.save(model_name)
        # wandb.save(model_name)

    def finish_experiment(self):
        """
        Finish the current experiment/run.
        """
        wandb.finish()


class WandbKerasModel(WandbManager):
    def __init__(self, model, project_name, entity=None, config=None):
        """
        Initialize the WandbManager and set up the Keras model.

        :param model: The Keras model to be used.
        :param project_name: Name of the wandb project to connect to.
        :param entity: Optional. The team or user who owns the project.
        :param config: Optional. Configuration parameters for the experiment.
        """
        # Initialize the WandbManager
        super(WandbKerasModel, self).__init__(project_name, entity, config)

        # Set the Keras model
        self.model = model

    def compile_and_fit(self, compile_args=None, fit_args=None, config_path=None):
        # Start the wandb experiment
        experiment_name = fit_args.pop("experiment_name", "Unnamed Experiment")
        self.start_experiment(experiment_name=experiment_name)

        # Prepare callbacks for fitting, including WandbCallback
        callbacks = fit_args.get("callbacks", [])
        callbacks.append(WandbCallback(save_model=False))
        fit_args["callbacks"] = callbacks

        # Compile the model
        self.model.compile(**(compile_args or {}))

        # Fit the model
        history = self.model.fit(**(fit_args or {}))

        # Save the model locally
        model_dir = "saved_model"
        self.model.save(model_dir)

        # Create a W&B artifact for the model
        artifact = wandb.Artifact(name=f"{experiment_name}_model", type="model")
        artifact.add_dir(model_dir)

        # Add the config.yaml file to the artifact
        artifact.add_file(config_path)

        # Log the artifact
        wandb.log_artifact(artifact)

        return history

    def load_model_from_artifact(self, artifact_name, custom_objects=None, run_id=None):
        """
        Load a model along with its parameters from a previous run artifact.
        Initializes a wandb run if there's none active.

        :param artifact_name: The name of the artifact to load, in the form "entity/project/artifact_name:version".
        :param custom_objects: Optional. A dictionary mapping names (strings) to custom classes or functions to be considered during deserialization.
        :param run_id: Optional. The ID of the wandb run to resume. If not provided, a new run will be started.
        """
        # Check if there's an active run, if not, initialize wandb
        if wandb.run is None:
            if run_id:
                wandb.init(
                    project=self.project_name,
                    entity=self.entity,
                    config=self.config,
                    id=run_id,
                    resume="allow",
                )
            else:
                wandb.init(
                    project=self.project_name, entity=self.entity, config=self.config
                )

        artifact = wandb.use_artifact(artifact_name, type="model")
        artifact_dir = artifact.download()

        # Load the model
        self.model = tf.keras.models.load_model(
            artifact_dir, custom_objects=custom_objects
        )

    def save(self, artifact_name):
        """
        Save the WandbKerasModel instance as a W&B artifact.

        :param artifact_name: The name for the artifact.
        """
        # Serialize the WandbKerasModel instance
        with open("WandbKerasModel.pkl", "wb") as file:
            pickle.dump(self, file)

        # Create a W&B artifact and add the serialized object
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file("WandbKerasModel.pkl")

        # Log the artifact
        wandb.log_artifact(artifact)

    @classmethod
    def from_artifact(cls, artifact_name, project_name, entity):
        """
        Load a WandbKerasModel instance from a W&B artifact.

        :param artifact_name: The name of the artifact to load.
        :param run_id: Run id.
        :return: A WandbKerasModel instance.
        """

        wandb.init(project=project_name, entity=entity)

        # Use the artifact
        artifact = wandb.use_artifact(artifact_name)
        artifact_dir = artifact.download()

        # Deserialize the WandbKerasModel instance
        with open(f"{artifact_dir}/WandbKerasModel.pkl", "rb") as file:
            instance = pickle.load(file)

        return instance

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying Keras model if not found in this class.

        :param name: The name of the attribute being accessed.
        :return: The value of the attribute from the underlying Keras model.
        """
        # Check if the model is set, if not, raise an AttributeError
        if self.model is None:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'model' or the model is not set"
            )

        # Check if the model has the attribute, if not, raise an AttributeError
        if not hasattr(self.model, name):
            raise AttributeError(
                f"'{self.__class__.__name__}' object and its 'model' do not have the attribute '{name}'"
            )

        # Return the attribute from the model
        return getattr(self.model, name)


if __name__ == "__main__":
    # Load and preprocess the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data
    y_train, y_test = to_categorical(y_train, 10), to_categorical(
        y_test, 10
    )  # One-hot encoding

    # Define a simple Keras model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.7),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # Wrap the model with WandbKerasModel
    wandb_model = WandbKerasModel(
        model=model, project_name="mnist_project", config={}, entity="leonvillapun"
    )

    # Compile and fit the model
    wandb_model.compile_and_fit(
        compile_args={
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"],
        },
        fit_args={
            "x": x_train,
            "y": y_train,
            "epochs": 3,
            "batch_size": 32,
            "validation_data": (x_test, y_test),
            "experiment_name": "mnist_simple_nn_3epochs",
        },
        config_path='config.yaml'
    )
