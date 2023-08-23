# ZooSegNet
# BlogPost
BlogPost: [click here](https://mayankanand111.github.io/Mayank_Portfolio/post/project-2/)

# Run your Python script with the required arguments
 chkpoint1- the location where model trained with the balanced dataset is saved or Model_v2_balanced_binary_best
 chkpoint2- the location where model trained with all dataset is saved or Model_v2_binary_best
 image_folder - Location where dataset is located or checkpoint_2023-08-03_01-35-29
 save_dir - the location where you want to save the predictions
 tensorboard_dir - the location where you want to save tensor board directory
 is_onlysegmentation - Pass only if you want segmentation
 with_cuda - Pass only when training on GPU 

# Run your Python script with the required arguments
python /home/ec2-user/SageMaker/FinalDemo/PredEnsembleSegmentationClassifier.py \
    --chkpoint1 "/home/ec2-user/SageMaker/FinalDemo/Model_v2_balanced_binary_best.pth.tar" \
    --chkpoint2 "/home/ec2-user/SageMaker/FinalDemo/Model_v2_binary_best.pth.tar" \
    --chkpoint_segmentation "/home/ec2-user/SageMaker/FinalDemo/checkpoint_2023-08-03_01-35-29.pt" \
    --image_folder "/home/ec2-user/SageMaker/FinalDemo/Dataset" \
    --save_dir "/home/ec2-user/SageMaker/FinalDemo/" \
    --segmented_save_dir "/home/ec2-user/SageMaker/FinalDemo/Detected/segmentedresults" \
    --segmented_data_dir "/home/ec2-user/SageMaker/FinalDemo/Detected" \
    --tensorboard_dir "/home/ec2-user/SageMaker/FinalDemo/tensorboard" \
    --with_cuda \
     --is_onlysegmentation
    # Uncomment the next line if you want to include the --is_onlysegmentation flag
    # --is_onlysegmentation

