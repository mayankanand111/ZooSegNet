#!/bin/bash

# Replace "pytorch_p310" with the desired name of your new Conda environment
CONDA_ENV_NAME="pytorch_p310"

# Activate the Conda environment
source activate $CONDA_ENV_NAME

#uncomment this line on first run
# conda install -c anaconda tensorboard

# Run your Python script with the required arguments
#chkpoint- location where trained model is saved
#image_folder - Location where dataset is located
#save_dir - location where you want to save the predictions
#tensorboard_dir - location where you want to save tensorboard directory

python /home/ec2-user/SageMaker/Inference/Prediction_script.py \
    --chkpoint "/home/ec2-user/SageMaker/Inference/Model_v2_binary_best.pth.tar" \
    --image_folder "/home/ec2-user/SageMaker/Inference/2022-09-11" \
    --save_dir "/home/ec2-user/SageMaker/Inference/" \
    --tensorboard_dir "/home/ec2-user/SageMaker/Inference/tensorboard"
