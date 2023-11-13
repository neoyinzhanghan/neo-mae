import h5py
import torch

def extract_tensor_and_check_dimensions(file_path):
    with h5py.File(file_path, 'r') as file:
        # Iterating through top-level items to find the first dataset
        for item in file.keys():
            if isinstance(file[item], h5py.Dataset):
                # Extracting the tensor from the first dataset
                tensor_data = torch.from_numpy(file[item][:])
                break

        # Checking the dimensions of the extracted tensor
        tensor_dimensions = tensor_data.shape
        return tensor_dimensions

# Path to your HDF5 file
file_path = "/media/hdd2/one_slide_mae/23.CFNA.9 A1 H&E _154610-patch_features.h5"

# Using the function to get the dimensions of the tensor
dimensions = extract_tensor_and_check_dimensions(file_path)
print(f"The dimensions of the tensor are: {dimensions}")
