import os
import numpy as np
import matplotlib.pyplot as plt
import fastmri
from fastmri.data import transforms as T
import warnings
from torch.utils.data import Dataset
from fastmri.data.subsample import EquiSpacedMaskFunc
from fastmri.data.transforms import apply_mask
import torch

class file_loader:
    def __init__(self,file_path):
        self.file_path = file_path
        
    def get_file_items(file_path):
        """
        Get the items in the specified directory.
        Args:
            file_path (str): The path to the directory.
        Returns:
            list: A list of items in the directory.
        """
        folder_path = file_path
        items = os.listdir(folder_path)
        print(items[-1])
        mcoil_pthon_file = items[-1]
        items.pop()
        return items
    
    
    
def absolute_path_items(file_items, file_path):    
    file_paths = []
    for file in file_items:
        file_paths.append(os.path.join(file_path, file))    
    return file_paths

# To get size of a full dataset
def get_size(fs_data):
    size = []
    x_size = []
    y_size = []
    for i in range(len(fs_data)):
        size.append(np.shape(fs_data[i])[0])
        x_size.append(np.shape(fs_data[i])[1])
        y_size.append(np.shape(fs_data[i])[2])
    return size, x_size, y_size


# To get size of a full dataset
def filter_coils(fs_data, coil_count):
    fs_data_filtered= []
    for i in range(len(fs_data)):
        if np.shape(fs_data[i])[0] == coil_count :
            fs_data_filtered.append(fs_data[i])
    return fs_data_filtered

def filter_shape(fs_data, x_size, y_size):
    fs_data_filtered= []
    for i in range(len(fs_data)):
        if np.shape(fs_data[i])[1] == x_size and np.shape(fs_data[i])[2] == y_size  :
            fs_data_filtered.append(fs_data[i])
    return fs_data_filtered


def show_coils(data, slice_nums, cmap='gray'):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.tight_layout()
        plt.imshow(data[num], cmap=cmap)


def show_multicoil_K_I(data_slice, K=True, I=False, show_image=True, show_K=True, show_RSS = True):
    if I==True:
        K = False
    if K==False and I==False:
        warnings.warn("Either K or I need to be False")
    else:    
        if K == True:
            Kdata_slice = data_slice
            slice_image = convert_K_to_I(data_slice)           # Apply Inverse Fourier Transform to get the complex image
        elif I == True:
            slice_image = data_slice
            Kdata_slice = convert_I_to_K(slice_image)    
        slice_image_abs = np.abs(slice_image)
        count_coils = Kdata_slice.shape[0]
        l = [i for i in range(0,count_coils,1)]
        if show_K==True:
            show_coils(np.log(np.abs(Kdata_slice) + 1e-9),l)  # Showing Logarithm
        if show_image==True:   # To exclude this part when showing Mask
            show_coils(slice_image_abs, l, cmap='gray')
            if show_RSS == True:
                fig = plt.figure()
                plt.imshow(rss_combine(slice_image), cmap='gray')
    
def convert_K_to_I(Kdata_slice):
    Kdata_slice_tensor = T.to_tensor(Kdata_slice)      # Convert from numpy array to pytorch tensor
    slice_image = fastmri.ifft2c(Kdata_slice_tensor)   # Apply Inverse Fourier Transform to get the complex image
    return T.tensor_to_complex_np(slice_image)

def convert_I_to_K(Idata_slice):
    Idata_slice_tensor = T.to_tensor(Idata_slice)      # Convert from numpy array to pytorch tensor
    slice_K = fastmri.fft2c(Idata_slice_tensor)           # Apply Inverse Fourier Transform to get the complex image
    return T.tensor_to_complex_np(slice_K)

def rss_combine(imgs):
    """Root Sum of Squares (RSS) across coil dimension."""
    return np.sqrt(np.sum(np.abs(imgs)**2, axis=0))


class Custom_FastMRIDataset(Dataset):
    def __init__(self,data_path,
                 mask_func=EquiSpacedMaskFunc(center_fractions=[0.08], accelerations=[2]),
                 transform=None, K=False, I=False, rss_combine=False):
        self.dataset = np.load(data_path, mmap_mode='r')
        self.length = len(self.dataset)
        self.mask_func = mask_func
        self.transform = transform
        self.K = K
        self.I = I
        self.rss_combine = rss_combine
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # Load the data
        slice_kspace = self.dataset[index,:,:,:]
        masked_kspace, mask, _ = apply_mask(torch.tensor(slice_kspace), self.mask_func)
        maskfull, _, _ = apply_mask(torch.ones_like(torch.tensor(slice_kspace)), self.mask_func)
        slice_ispace = convert_K_to_I(slice_kspace)
        masked_ispace = convert_K_to_I(masked_kspace)
        rss_combined = rss_combine(slice_ispace)
        masked_rss_combined = rss_combine(masked_ispace)
        
        if self.K==True:  # To save RAM space (by loading only what is required)
            return {'kspace': slice_kspace,
                'masked_kspace':masked_kspace,
                'mask_used': maskfull
                }
        elif self.I==True:
            return {'mask_used': maskfull,
                'ispace': slice_ispace,
                'masked_ispace':masked_ispace,
                }
        elif self.rss_combine==True:
            return {'rss_combined' : rss_combined,
                'masked_rss_combined':masked_rss_combined 
                }
        else:
            return {'kspace': slice_kspace,
                'masked_kspace':masked_kspace,
                'mask_used': maskfull,
                'ispace': slice_ispace,
                'masked_ispace':masked_ispace,
                'rss_combined' : rss_combined,
                'masked_rss_combined':masked_rss_combined 
                }
                

  