import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from logUtils import printCustom

# Custom dataset class
class SeismicDataset(Dataset):
    def __init__(self, data, labels, verbose=False):
        self.data = data
        self.labels = labels
        self.verbose = verbose
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label.clone().detach()
    
def read_hdf5(file_path, seconds):
    samples_dict = {'sample':[], 'label':[]}
    sampling_rate = 100
    with h5py.File(file_path, 'r') as hdf5_file:
        dataset_group = hdf5_file['data']
        for key in dataset_group.keys():
            dataset = dataset_group[key]
            data = dataset[:]
            if(len(data[0]) != seconds):
                continue
            samples_dict['sample'].append(data)
            p_ts = dataset.attrs['p_arrival_sample']
            s_ts = dataset.attrs['s_arrival']
            if(p_ts == 'None'):
                p_idx = -1
                p_confidence = 0
            else:
                p_idx = int(p_ts*sampling_rate)
                p_confidence = 1
            if(s_ts == 'None'):
                s_idx = -1
                s_confidence = 0
            else:
                s_idx = int(s_ts*sampling_rate)
                s_confidence = 1
            # our label format will be [p_idx, s_idx,p_confidence, s_confidence]
            samples_dict['label'].append([p_idx, s_idx,p_confidence, s_confidence])
    return samples_dict

def applyWindowing(data, labels, window_size=900, hopping_size=300):
    global_indices = True
    global_confidences = False
    windowed_data = []
    windowed_labels = []
    for i in range(len(data)):
        sample = data[i] # sample is shape [3,1500]-->[channel, data points]
        label = labels[i] # label is shape [4]-->[output]
        if(type(label)!=torch.Tensor):
            printCustom("info","Label initial type:"+str(type(label)))
        # split the data into windows of window_size timesteps, with a hop of hopping_size timesteps
        windows = [sample[:, j:j+window_size] for j in range(0, sample.shape[1]-window_size+1, hopping_size)]
        # Each window has shape [3,900], we unsqueeze it to [3,1,900] for model input compatibility
        windows = [window.unsqueeze(1) for window in windows]
        windowed_data.append(torch.stack(windows)) # windowed data shape is [3,3,1,900]-->[number of windows, channel, 1, data points]
        # Labeling the Data for Each Frame 
        # You divide the data into frames of window_size points with a hopping_size-point overlap. 
        # For each frame, the labels (p_index, s_index, p_existence, s_existence) should reflect what happens in that frame:
        # p_existence and s_existence: These are straightforward. 
        # If P exists within the frame, p_existence = 1; otherwise, p_existence = 0. The same logic applies for s_existence.
        # p_index and s_index: These indices should be relative to the frame, not the global sequence. 
        # If P is at the 200th point globally, and the frame covers 0-900, the P index will be 200 in that frame. 
        # However, if the frame is 300-1200, P would now be at the 200th point relative to that frame. Similarly for S.
        # Now prepare the labels for each frame
        p_idx, s_idx, p_confidence, s_confidence = label.tolist()
        for j in range(len(windows)):
            new_labels = list()
            # Get the start and end indices of the frame
            start_idx = j * hopping_size
            end_idx = start_idx + window_size
            # Check if P exists in the frame
            p_existence = 1 if p_idx >= start_idx and p_idx < end_idx else 0
            # Check if S exists in the frame
            s_existence = 1 if s_idx >= start_idx and s_idx < end_idx else 0
            # Get the relative P index if P exists in that frame, otherwise set it to 0
            p_index = p_idx - start_idx if p_existence else 0
            # Get the relative S index if S exists in that frame, otherwise set it to 0
            s_index = s_idx - start_idx if s_existence else 0
            if(global_indices == True and global_confidences == True):
                new_labels.extend([p_idx, s_idx, p_confidence, s_confidence])
            elif(global_indices == True and global_confidences == False):
                new_labels.extend([p_idx, s_idx, p_existence, s_existence])
            elif(global_indices == False and global_confidences == True):
                new_labels.extend([p_index, s_index, p_confidence, s_confidence])
            elif(global_indices == False and global_confidences == False):
                new_labels.extend([p_index, s_index, p_existence, s_existence])
            windowed_labels.append(new_labels)
    return torch.stack(windowed_data), torch.tensor(windowed_labels).float()
    

def get_dataset(file_path, seconds, window_size, hopping_size, verbose=False):
    samples_dict = read_hdf5(file_path, seconds)

    # convert the list of numpy arrays to a single numpy array
    samples_array = np.array(samples_dict['sample'])
    labels_array = np.array(samples_dict['label'])

    # Convert sample and label numpy arrays to a PyTorch tensor   
    data = torch.tensor(samples_array).float() # Data shape: torch.Size([1575, 3, 1500])
    labels = torch.tensor(labels_array).float() # Labels shape: torch.Size([1575, 4])

    # apply windowing to the dataset
    windowed_data, windowed_labels = applyWindowing(data, labels, window_size, hopping_size)
    
    if(verbose):
        #print the dataset content info to user
        printCustom("header","---------------- Dataset information: ----------------")
        printCustom("info","Sample rate: 100 Hz")
        printCustom("info","Chosen window size:"+str(window_size))
        printCustom("info","Hopping size:"+str(hopping_size))
        printCustom("info","Each sample is window_size/sample rate seconds long. Each sample has 3 channels.")
        printCustom("info","Dataset values shape is: [num_samples, num_window, num_channel, (height) 1, (width) num_sample_points]")
        printCustom("info","Dataset value shape: "+str(windowed_data.shape))
        printCustom("info","Dataset labels shape is: [num_samples, 4], where the 4 values are [p_idx, s_idx, p_confidence, s_confidence]")
        printCustom("info","Dataset label shape:"+str(windowed_labels.shape))
        printCustom("info","One sample shape: "+str(windowed_data[0].shape))
        printCustom("info","One label shape: "+str(windowed_labels[0].shape))
        printCustom("info","An example sample's label: "+str(windowed_labels[12]))
        printCustom("info","----------------------------------------------------")
    # Create dataset
    dataset = SeismicDataset(windowed_data, windowed_labels, verbose=True)
    return dataset