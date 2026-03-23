import sys
import os
import h5py

from load_mri_data import file_loader

class vol_sampler():
    def __init__(self,file_path):
        self.file_path = file_path
    
    def __slice_data__(self):
        """
        Slice the k-space data into individual slices.
        Returns:
            list: A list of k-space data slices.
        """
        slice_kspace_data = []
        for i in range(0,len(self.volume_kspace)):
            slice_kspace_data.append(self.volume_kspace[i])
        return slice_kspace_data
        
    def load_h5py(self):
        """
        Load the HDF5 file and extract k-space data.
        Returns:
            list: A list of k-space data slices.
        """
        file_items = file_loader.get_file_items(self.file_path)
        fs_mri_data=[]
        for i in range(len(file_items)):
            file_items[i] = os.path.join(self.file_path, file_items[i])
            hf = h5py.File(file_items[i])
            self.volume_kspace = hf['kspace'][()]
            fs_mri_data.extend(self.__slice_data__())
        return fs_mri_data
            
# obj = vol_sampler(file)
# items = obj.load_h5py()

# for i in range(0,len(items),1):
#     print(f"file number {i+1} : {items[i].shape} ")

class Individual_vol_sampler():
    def __init__(self,file_item):
        self.file_item = file_item

    def __slice_data__(self):
        """
        Slice the k-space data into individual slices.
        Returns:
            list: A list of k-space data slices.
        """
        slice_kspace_data = []
        for i in range(0,len(self.volume_kspace)):
            slice_kspace_data.append(self.volume_kspace[i])
        return slice_kspace_data
        
    def load_h5py(self):
        """
        Load the HDF5 file and extract k-space data individually.
        Returns:
            list: A list of k-space data slices for an individual K-space.
        """
        hf = h5py.File(self.file_item)
        fs_mri_data=[]
        self.volume_kspace = hf['kspace'][()]
        fs_mri_data.extend(self.__slice_data__())
        return fs_mri_data
                