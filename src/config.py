import yaml
import argparse
import torch as t
import numpy as np
import random
import os

def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load base config from yaml
config_data = load_config("./src/config.yaml")

# Parse command line arguments to override config
parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument('--config', type=str, default='config.yaml', help="Path to config file")
parser.add_argument('--exp_name', type=str, default=config_data['experiment']['name'], help="Experiment Name")
parser.add_argument('--model', type=str, default=config_data['experiment']['model'], help="Model Name")
parser.add_argument('--gpu_id', type=str, default=config_data['experiment']['gpu_id'], help="GPU ID")

args_cmd, unknown = parser.parse_known_args()

# Update config with command line args
config_data['experiment']['exp_name'] = args_cmd.exp_name
config_data['experiment']['model'] = args_cmd.model
config_data['experiment']['gpu_id'] = args_cmd.gpu_id

# Flatten config for easier access in code (optional, but helps minimize changes in other files)
args = {}
args.update(config_data['experiment'])
args.update(config_data['model'])
args.update(config_data['training'])
args.update(config_data['data'])
args.update(config_data['paths'])

# Add derived paths
args['path'] = os.path.join(args['checkpoint_dir'], args['exp_name'], args['model'], f"op_steps_{args['out_length']}")

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
args['device'] = device

# Set seeds
seed = args['seed']
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False

print(f"Experiment Name: {args['exp_name']}")
print(f"Device: {device}")

# Export commonly used variables for compatibility
learning_rate = args['learning_rate']
dataset = args['dataset']
