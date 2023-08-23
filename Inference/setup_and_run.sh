#!/bin/bash

# Replace "pytorch_p310" with the desired name of your new Conda environment
CONDA_ENV_NAME="pytorch_p310"

# Create the Conda environment (only if it doesn't exist)
if ! conda info --envs | grep -q $CONDA_ENV_NAME; then
    conda create --name $CONDA_ENV_NAME python=3.9 -y
fi

# Activate the Conda environment
source activate $CONDA_ENV_NAME

# Install packages from requirements.txt
pip install -r requirements.txt

# Optionally install tensorboard from anaconda (uncomment if needed)
# conda install -c anaconda tensorboard

# Run your Python script with the required arguments
python /Users/mayank/PycharmProjects/ZooSegNet/Inference/PredEnsembleSegmentationClassifier.py \
    --chkpoint1 "/Users/mayank/PycharmProjects/ZooSegNet/Inference/Model_v2_balanced_binary_best.pth.tar" \
    --chkpoint2 "/Users/mayank/PycharmProjects/ZooSegNet/Inference/Model_v2_binary_best.pth.tar" \
    --chkpoint_segmentation "/Users/mayank/PycharmProjects/ZooSegNet/Inference/checkpoint_2023-08-03_01-35-29.pt" \
    --image_folder "/Users/mayank/PycharmProjects/ZooSegNet/Inference/Dataset" \
    --save_dir "/Users/mayank/PycharmProjects/ZooSegNet/Inference/" \
    --segmented_save_dir "/Users/mayank/PycharmProjects/ZooSegNet/Inference/Detected/segmentedresults" \
    --segmented_data_dir "/Users/mayank/PycharmProjects/ZooSegNet/Inference/Detected" \
    --tensorboard_dir "/Users/mayank/PycharmProjects/ZooSegNet/Inference/tensorboard" \
    --with_cuda \
    # Uncomment the next line if you want to include the --is_onlysegmentation flag
    # --is_onlysegmentation
