import customDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from logUtils import printCustom

"""
Load the dataset from the hdf5 file
Parameters:
    file_path (str): path to the hdf5 file
    seconds (int): duration of the window in samples (options: 1500 (15 sec), 3000 (30 sec), 6000 (60 sec), 10000 (100 sec))
    window_size (int): size of the window in samples
    hopping_size (int): size of the hop in samples
    verbose (bool): print information about the dataset
Returns:
    dataset (SeismicDataset): seismic dataset
"""

# if dataset is not downloaded, download dataset
import os
current_path = os.getcwd()
printCustom("info", "Checking if the dataset is downloaded with the name afad_hfd5.zip")
if(not os.path.exists(current_path+"/afad_hdf5.zip")):
    printCustom("warning", "Dataset is not found, downloading the dataset")
    # share link is: https://drive.google.com/file/d/1isvI3lELKocPuF2mIjMiMpXrcaoWTaIQ/view?usp=sharing file_id = '1isvI3lELKocPuF2mIjMiMpXrcaoWTaIQ' 
    import gdown
    file_id = '1isvI3lELKocPuF2mIjMiMpXrcaoWTaIQ'# file id is the /d/ part in the share link
    output = 'afad_hdf5.zip'
    gdown.download(f'https://drive.google.com/uc?id={file_id}',output,quiet=False)

printCustom('success','Dataset is downloaded, loading the dataset')

dataset = customDataset.get_dataset(file_path=current_path+"/afad_hdf5.zip", 
                                    seconds = 1500 , window_size=900, hopping_size=300, verbose=True)

# torch print options are set in order not to use scientific notation, and to show only 2 decimal points, and show 100 characters in a line
torch.set_printoptions(sci_mode=False, precision=2, linewidth=100)
printCustom("info","Some label examples in the dataset:")
for i, (data, label) in enumerate(dataset):
    if i == 0 or i == 3 or i == 28 or i == 35:
        printCustom("info","Label: "+str(label))
    elif i == 36:
        break

# Split the dataset into train, validation, and test sets
# First, define sizes of the splits
TRAIN_RATIO, VALIDATION_RATIO = 0.8, 0.1
# check if the sum of the ratios is not more than 1, if it is, raise an exception
assert TRAIN_RATIO + VALIDATION_RATIO <= 1, "The sum of the ratios should be less than or equal to 1"
train_size = int(TRAIN_RATIO * len(dataset))
val_size = int(VALIDATION_RATIO * len(dataset))
test_size = len(dataset) - train_size - val_size
# Set the seed for reproducibility
torch.manual_seed(0)
# Then, create train, validation and test dataloaders
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

printCustom("success","--------------------------\nDataloaders are created.")
printCustom("success","Train dataloader size:"+str(len(train_dataloader)*TRAIN_BATCH_SIZE))
printCustom("success","Validation dataloader size:"+str(len(val_dataloader)*TRAIN_BATCH_SIZE))
printCustom("success","Test dataloader size:"+str(len(test_dataloader)*TEST_BATCH_SIZE))
printCustom("success","--------------------------\n")



       