#!/usr/bin/env python3

# by: Sammpath
# Imports python modules
import argparse



# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument 
#       collection that you created with this function
# 
def get_input_args():
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

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type=str, 
                        help = 'Directory to images') 
    parser.add_argument('--save_dir', type = str, default = 'checkpoints/', 
                        help = 'path to the folder of images') 
    parser.add_argument('--arch', type = str, default = 'vgg13', 
                        help = 'CNN Model Architecture') 
    parser.add_argument('--learning_rate', type = float, default = '0.01', 
                        help = 'Learning rate') 
    parser.add_argument('--hidden_units', type = int, default = '512', 
                        help = 'Hidden units') 
    parser.add_argument('--epochs', type = int, default = '20', 
                        help = 'epochs') 
    parser.add_argument('--gpu', action='store_true',
                        help = 'Use GPU') 

    
    in_args = parser.parse_args()

    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return in_args
