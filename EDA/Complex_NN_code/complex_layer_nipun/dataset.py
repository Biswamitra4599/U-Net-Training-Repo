
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from complex_utils import complex_modulate

def complex_haar(x):
    ll = (x[:, ::2, ::2] + x[:, 1::2, ::2] +
          x[:, ::2, 1::2] + x[:, 1::2, 1::2]) / 2
    lh = (x[:, ::2, ::2] - x[:, 1::2, ::2] +
          x[:, ::2, 1::2] - x[:, 1::2, 1::2]) / 2
    hl = (x[:, ::2, ::2] + x[:, 1::2, ::2] -
          x[:, ::2, 1::2] - x[:, 1::2, 1::2]) / 2
    hh = (x[:, ::2, ::2] - x[:, 1::2, ::2] -
          x[:, ::2, 1::2] + x[:, 1::2, 1::2]) / 2
    return torch.cat([ll, lh, hl, hh], dim=0)

class S1SLCDataset(Dataset):
    def __init__(self, paths_h=None, paths_v=None, path_label=None, scale=1, normal=False):
        self.path_counts = len(paths_h)
        self.normal = normal
        self.scale = scale
        self.h_datasets = [np.load(path, mmap_mode='r') for path in paths_h]
        self.h_lengths = [len(dataset) for dataset in self.h_datasets]      
        self.cumulative_lengths = np.cumsum([0] + self.h_lengths)
        if paths_v is not None:
            self.paths_v = paths_v
            self.v_datasets = [np.load(path, mmap_mode='r') for path in paths_v]
        else:
            self.paths_v = None

        if path_label is not None:
            self.path_label = path_label
            self.labels = [np.load(path, mmap_mode='r') for path in path_label]
        else:
            self.path_label = None
            self.labels = None

    @classmethod
    def from_root(cls, root_dir, scale=1, normal=False):
        """Build an S1SLCDataset by scanning city sub-directories under *root_dir*.

        Expected layout:
            root_dir/
                CityA/
                    HH_Complex_Patches.npy
                    HV_Complex_Patches.npy   (optional)
                    Labels.npy               (optional)
                CityB/
                    ...
        """
        subdirs = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )

        paths_h = []
        paths_v = []
        paths_label = []

        for sd in subdirs:
            sd_path = os.path.join(root_dir, sd)
            hh = os.path.join(sd_path, "HH_Complex_Patches.npy")
            hv = os.path.join(sd_path, "HV_Complex_Patches.npy")
            lbl = os.path.join(sd_path, "Labels.npy")

            if os.path.isfile(hh):
                paths_h.append(hh)
            if os.path.isfile(hv):
                paths_v.append(hv)
            if os.path.isfile(lbl):
                paths_label.append(lbl)

        return cls(
            paths_h=paths_h,
            paths_v=paths_v if paths_v else None,
            path_label=paths_label if paths_label else None,
            scale=scale,
            normal=normal,
        )

    def __len__(self):
        return self.cumulative_lengths[-1]


    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img_copy=np.copy(pil_img)
        pil_resized = np.resize(pil_img_copy,(newW, newH))

        img_nd = np.array(pil_resized)
        return img_nd
    
    def __getitem__(self, idx):
        # Locate correct dataset
        ds_idx = 0
        local_idx = idx
        for i in range(len(self.cumulative_lengths)-1):
            if self.cumulative_lengths[i] <= idx < self.cumulative_lengths[i+1]:
                ds_idx = i
                local_idx = idx - self.cumulative_lengths[i]
                break
        
        data_h = self.h_datasets[ds_idx]
        img_h = data_h[local_idx]
        
        img_v = None
        if self.paths_v is not None:
            data_v = self.v_datasets[ds_idx]
            img_v = data_v[local_idx]
            
        label_current = None
        if self.path_label is not None:
            label = self.labels[ds_idx]
            label_current = label[local_idx]


        if self.normal == True:
            # Note: This logic seems flawed in user snippet (img_h overwritten by img_down)
            # But I follow "exact dataset.py" request logic closely.
            img_highres_h = img_h
            img_down_h = self.preprocess(img_h, self.scale)
            
            if self.paths_v is not None:
                img_highres_v = img_v
                img_down_v = self.preprocess(img_v, self.scale)
                
                img_highres = np.stack((img_highres_h, img_highres_v), axis=0)
                img_down = np.stack((img_down_h, img_down_v), axis=0) # use downsampled
            else:
                img_highres = img_highres_h[np.newaxis, ...]
                img_down = img_down_h[np.newaxis, ...]

        if not self.normal:
            if self.paths_v is not None:
                img_down = np.stack((img_h, img_v), axis=0)
            else:
                img_down = img_h[np.newaxis, ...]
            img_highres = img_down

        # Convert to torch and compute transforms
        img_tensor = torch.from_numpy(img_down).type(torch.complex64)
        raw_c = img_tensor
        fourier_c = torch.fft.fft2(raw_c, norm="ortho")
        wavelet_c = complex_haar(raw_c)

        # Convert to Simulated Real-Valued Complex format
        raw = complex_modulate(raw_c.real, raw_c.imag).float()
        fourier = complex_modulate(fourier_c.real, fourier_c.imag).float()
        wavelet = complex_modulate(wavelet_c.real, wavelet_c.imag).float()
        
        # High-res target for super-resolution (if needed, user included target)
        target_c = torch.from_numpy(img_highres).type(torch.complex64)
        target = complex_modulate(target_c.real, target_c.imag).float()

       
        sample = {
            'raw': raw,
            'fourier': fourier,
            'wavelet': wavelet,
            'target': target,
        }
        
        if label_current is not None:
            sample['label'] = torch.tensor(label_current - 1).long()
            
        return sample
