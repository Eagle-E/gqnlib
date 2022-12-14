# Run training

# Settings
$MODEL = 'gqn'
$CUDA = 0
$MAX_STEPS = 100000
$TEST_INTERVAL = 2500
$SAVE_INTERVAL = 10000
$DATASET = 'shepard_metzler_5_parts'

# Dataset path
$env:LOGDIR = 'D:/path/to/your/log/dir/'
$env:EXPERIMENT_NAME = 'gqn-experiment'

# Dataset path
$env:DATASET_DIR = 'D:/path/to/your/dataset/dir'
$env:DATASET_NAME = $DATASET + "_10_percent_torch" # actual name of the folder in the dataset_dir you want to load

# Config for training
$env:CONFIG_PATH = 'D:/path/to/the/config.json'

#python "./examples/train.py" `
python "D:/path/to/the/training/script/train.py" `
            "--cuda" "$CUDA" `
            "--model" "$MODEL" `
            "--max-steps" "$MAX_STEPS" `
            "--test-interval" "$TEST_INTERVAL" `
            "--save-interval" "$SAVE_INTERVAL"
