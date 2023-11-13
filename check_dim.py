import h5py
import torch

# Reopening the HDF5 file to check other datasets
file_path = '/media/hdd2/one_slide_mae/23.CFNA.9 A1 H&E _154610-patch_features.h5'

# Dictionary to hold tensor data from each dataset
tensors_from_datasets = {}

with h5py.File(file_path, 'r') as file:
    for item in file.keys():
        if isinstance(file[item], h5py.Dataset):
            # Extracting the tensor from each dataset
            tensors_from_datasets[item] = torch.from_numpy(file[item][:])

# Checking the dimensions of each extracted tensor
tensor_dimensions = {key: tensor.shape for key, tensor in tensors_from_datasets.items()}
tensor_dimensions
 