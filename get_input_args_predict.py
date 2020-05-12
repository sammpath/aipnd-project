#!/usr/bin/env python3

# Imports python modules
import argparse

# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument 
#       collection that you created with this function
# 
def get_input_args_predict():
    """
    Train a new network on a data set with train.py

    Basic usage: python train.py data_directory
    Prints out training loss, validation loss, and validation accuracy as the network trains
    Options:
    Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    Choose architecture: python train.py data_dir --arch "vgg13"
    Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    Use GPU for training: python train.py data_dir --gpu
    """
# python predict.py /home/workspace/ImageClassifier/flowers/test/11/image_03098.jpg checkpoints/ --gpu --category_names cat_to_name.json --top_k 3 



    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_file_with_path', type=str, 
                        help = 'Input image file with path') 
    parser.add_argument('checkpoint_dir', type=str, 
                        help = 'Directory to checkpoint file') 
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                        help = 'category to names json file') 
    parser.add_argument('--top_k', type = int,  default = '3', 
                        help = 'top_k value in integer') 
    parser.add_argument('--gpu', action='store_true',
                        help = 'Use GPU') 

    
    in_args = parser.parse_args()

    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return in_args

# Call to main function to run the program
if __name__ == "__main__":
    with active_session():        
        main()