import argparse
import tensorflow as tf
import wandb

from hda.models.base_model import WandbKerasModel
from hda.models.model import TimeSeriesModelSimple, TimeSeriesModelCfC, TimeSeriesModelCfCWithNCP, TimeSeriesModelGRU, TimeSeriesModelLSTM, TimeSeriesModelVanillaRNN
from hda.preprocessor import Preprocessor
from hda.utils import get_dataset_shape, load_config, predict_and_plot

parser = argparse.ArgumentParser(description="Process model name.")

# Add the arguments to the parser
parser.add_argument("--model", type=str, help="The name of the model", required=True)
parser.add_argument("--version", type=str, help="The name of the model", required=True)

# Parse the arguments
args = parser.parse_args()

def main():
    model_name = args.model
    versions = args.version
    if model_name not in ["Simple", "GRU", "VanillaRNN", "LSTM", "CfC", "CfCWithNCP"]:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Initialize artifact variables
    artifact_name = f'bdma/hda-big-3/{model_name}_weights:{versions}'
    artifact_type = 'model-weights'
    config_path = '/colab_config.yaml' 
    model_weights_path = f'/{model_name}_weights.h5'
    
    
    # Get the artifact
    run = wandb.init()
    artifact = run.use_artifact(artifact_name, type=artifact_type)
    artifact_dir = artifact.download()

    # Get the config
    config = load_config(artifact_dir + config_path)
    config['preprocess']['data_directory'] = 'datasets/'
    batch_size, versions, time_steps, diodes = get_dataset_shape(config)
    
    # Get the testing data
    preprocessor = Preprocessor(artifact_dir + config_path, mode="demo")
    _, _, test_dataset = preprocessor.run()
    
    # Inititalize the models
    models = {
        "Simple": TimeSeriesModelSimple(diodes, config['convolutions_conf'], config['dense_conf']),
        "GRU": TimeSeriesModelGRU(diodes, config['convolutions_conf'], config['dense_conf'], gru_conf=config['gru']),
        "VanillaRNN": TimeSeriesModelVanillaRNN(diodes, config['convolutions_conf'], config['dense_conf'], rnn_conf=config['simple_rnn']),
        "LSTM": TimeSeriesModelLSTM(diodes, config['convolutions_conf'], config['dense_conf'], lstm_conf=config['lstm']),
        "CfC": TimeSeriesModelCfC(diodes, config['convolutions_conf'], config['dense_conf'], cfc_conf=config['cfc']),
        "CfCWithNCP": TimeSeriesModelCfCWithNCP(diodes, config['convolutions_conf'], config['dense_conf'], cfc_conf=config['cfc'], wiring_conf=config['wiring']),
    }
    
    # Initialize wandb model and predict
    model_class= models[model_name]
    model_class.build(input_shape=(batch_size, versions, time_steps, diodes))
    model_class.load_weights(artifact_dir + model_weights_path)        
    wandb_model = WandbKerasModel(
            model=model_class, project_name="hda-demo", config={}, entity="bdma"
    )
    
    predict_and_plot(wandb_model, test_dataset, config)
    
if __name__ == "__main__":
    main()
