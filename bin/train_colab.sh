#########################################################################
#
#	Script to run training code on Google Colab. This script assumes the 
#	data has been uploaded to a drive folder, from where it will be copied
#	to the colab compute instance.
#
#	Adjust paths and strings to make it work in your own environment
#
#   Run this script in a notebook cell.
#########################################################################


# mount googlde drive to colab instance
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# install gqnlib
%cd /content/drive/MyDrive/path/to/gqnlib/folder/gqnlib/
!python ./setup.py install
!pip install tensorboardX

# add cwd to path so gqnlib is accessible for imports
import sys, os
sys.path.append(os.getcwd())

# go back to the root folder
%cd /content



# move data from drive to instance storage
!mkdir /content/data
# copy from drive to instance 
#	/content/ is the root of the instance
#	/content/drive/MyDrive is a mounting of the drive to the instance, but read speeds are slow 
!cp -r /content/drive/MyDrive/path/to/dataset/in/drive /content/path/to/folder/in/instance


# go to the gqn directory
%cd /content/drive/MyDrive/path/to/gqnlib/folder/

# RUN TRAINING

# Settings
MODEL="gqn"
CUDA=0
MAX_STEPS=2000000
TEST_INTERVAL=2500
SAVE_INTERVAL=10000
DATASET="shepard_metzler_5_parts"

# Log path
%env LOGDIR=./logs/
%env EXPERIMENT_NAME=gqn_training_0

# Dataset path
%env DATASET_DIR=/content/data/
%env DATASET_NAME=dataset_folder_name

# Config for training
%env CONFIG_PATH=/content/path/togqnlib/examples/config.json

!python3 /content/path/to/gqnlib/examples/train.py \
            --cuda $CUDA \
            --model $MODEL \
            --max-steps $MAX_STEPS \
            --test-interval $TEST_INTERVAL \
            --save-interval $SAVE_INTERVAL \
            --batch-size 30