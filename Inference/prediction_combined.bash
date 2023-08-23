#!/bin/bash

# Replace "pytorch_p310" with the desired name of your new Conda environment
CONDA_ENV_NAME="pytorch_p310"

# Activate the Conda environment
source activate $CONDA_ENV_NAME

#uncomment this line on first run
# conda install -c anaconda tensorboard

# Run your Python script with the required arguments
#chkpoint1- location where model trained with balanced dataset is saved
#chkpoint2- location where model trained with all dataset is saved
#image_folder - Location where dataset is located
#save_dir - location where you want to save the predictions
#tensorboard_dir - location where you want to save tensorboard directory
#is_onlysegmentation - Pass only if you want segmentation
python /home/ec2-user/SageMaker/Inference/PredEnsembleSegmentationClassifier.py \
    --chkpoint1 "/home/ec2-user/SageMaker/Inference/Model_v2_balanced_binary_best.pth.tar" \
    --chkpoint2 "/home/ec2-user/SageMaker/Inference/Model_v2_binary_best.pth.tar" \
    --chkpoint_segmentation "/home/ec2-user/SageMaker/Inference/checkpoint_2023-08-03_01-35-29.pt" \
    --image_folder "/home/ec2-user/SageMaker/Inference/Dataset" \
    --save_dir "/home/ec2-user/SageMaker/Inference/" \
    --segmented_save_dir "/home/ec2-user/SageMaker/Inference/Detected/segmentedresults" \
    --segmented_data_dir "/home/ec2-user/SageMaker/Inference/Detected" \
    --tensorboard_dir "/home/ec2-user/SageMaker/Inference/tensorboard" \
    # --is_onlysegmentation