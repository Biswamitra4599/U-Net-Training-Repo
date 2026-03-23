from torch.utils.data import Dataset
import torch
import numpy as np
import sys
import importlib
import fastmri
from numpy import random
import matplotlib.pyplot as plt
from load_mri_data import show_coils, show_multicoil_K_I, convert_K_to_I, convert_I_to_K, rss_combine
from fastmri.data.subsample import EquiSpacedMaskFunc
from fastmri.data.transforms import apply_mask
import time
import itertools
from torch.utils.data import DataLoader
import glob
import h5py
import sigpy as sp
import sigpy.mri as mr
from pygrappa import grappa
from fastmri.data import transforms as T


class Custom_FMRI_DataLoader_nil(Dataset):
    def __init__(self, data_paths,
                 mask_func=EquiSpacedMaskFunc(center_fractions=[0.08], accelerations=[20]),
                 transform=None,
                 input_req=[1, 1, 1, 1, 1],
                 output_req=[1, 1, 1, 1],
                 methods_flags=[1,1],
                 espirit_params = [24, 0.02, 6, 0.95,True],
                 espirit_device = -1,
                 grappa_params = [184, 20]
                ):
        
        """
            
            methods_flags are used for grappa and ESPIRiT
            they are binary flags , just change them to 0/1 for enabling them

            ESPRIRiT parameters
            calib_width=24,    # Center region used for auto-calibration
            thresh=0.02,       # Background threshold
            kernel_width=6,    # Standard kernel size
            crop=0.95,         # 0.95 is standard for data with no oversampling
            show_pbar=True

            ESPIRiT device selection
            espirit_device=-1 -> CPU
            espirit_device=0  -> First GPU (requires CuPy-enabled SigPy)
            show_pbar=True

            Grappa Params
            first param = the ACS line starting location
            second param = half width of your calibration signal
            the second parameter chooses the width of the ACS line
        """


        if len(input_req) != 5:
            print("Wrong ip parameters, Assigning Brute force!")
            input_req = [1, 1, 1, 1, 1]
        
        if len(output_req) != 4:
            print("Wrong output parameters, Assigning Brute force!")
            output_req = [1, 1, 1, 1]
        
        if len(methods_flags) !=2:
            print("Wrong method parameters, Assigning Brute force!, flags default to false for both")
            methods_flags = [0,0]

        self.paths = data_paths
        self.length = len(self.paths)
        self.mask_func = mask_func
        self.transform = transform

        # 5 input flags
        self.K = input_req[0]
        self.I = input_req[1]
        self.rss_combine = input_req[2]
        self.rss_fft = input_req[3]
        self.mask = input_req[4]

        # 4 output flags
        self.K_full = output_req[0]
        self.I_full = output_req[1]
        self.rss_combine_full = output_req[2]
        self.rss_fft_full = output_req[3]

        # 5 methods flags
        self.enable_grappa = methods_flags[0]
        self.enable_espirit = methods_flags[1]

        # 6 Espirit params
        self.espirit_params = espirit_params
        self.espirit_device = espirit_device
        self.grapp_params = grappa_params


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        dd = {}

        kspace_cmplx = np.load(self.paths[index]) #loads the data -> (coils, 640, 368)
        kspace = T.to_tensor(kspace_cmplx)  # converts to tensor -> (coils, 640, 368, 2) *introduces the last dimension
        
        # ─── Cached intermediates (None = not yet computed) ───
        _ispace = None
        _rss = None
        _masked_kspace = None
        _mask = None
        _masked_ispace = None
        _masked_rss = None

        # ─── Lazy getters: compute once, cache, reuse ───
        def get_ispace():
            nonlocal _ispace
            if _ispace is None:
                _ispace = fastmri.ifft2c(kspace)
            return _ispace

        def get_rss():
            nonlocal _rss
            if _rss is None:
                _rss = fastmri.rss(fastmri.complex_abs(get_ispace()), dim=0)
            return _rss

        def get_masked():
            nonlocal _masked_kspace, _mask
            if _masked_kspace is None:
                _masked_kspace, _mask, _ = apply_mask(kspace, self.mask_func)
            return _masked_kspace, _mask

        def get_masked_ispace():
            nonlocal _masked_ispace
            if _masked_ispace is None:
                mk, _ = get_masked()
                _masked_ispace = fastmri.ifft2c(mk)
            return _masked_ispace

        def get_masked_rss():
            nonlocal _masked_rss
            if _masked_rss is None:
                _masked_rss = fastmri.rss(fastmri.complex_abs(get_masked_ispace()), dim=0)
            return _masked_rss
        
        def get_espirit_sensitivity_maps():
            espirit_dev = -1
            if self.espirit_device >= 0 and sp.config.cupy_enabled:
                espirit_dev = self.espirit_device

            app = mr.app.EspiritCalib(
                kspace_cmplx, 
                calib_width= self.espirit_params[0],    # Center region used for auto-calibration
                thresh= self.espirit_params[1],       # Background threshold
                kernel_width= self.espirit_params[2],    # Standard kernel size
                crop= self.espirit_params[3],         # 0.95 is standard for data with no oversampling
                device=sp.Device(espirit_dev),
                show_pbar= self.espirit_params[4]
            )

            sensitivity_maps = app.run() #running Espirit
            sensitivity_maps_cpu = sp.to_device(sensitivity_maps, sp.cpu_device)

            return T.to_tensor(np.asarray(sensitivity_maps_cpu))

        def get_grappa_op():
            """
                this returns the k-space reconstrcuted data for each coil of the sample
            """
            if _masked_kspace is None:
                get_masked()
            ctr, pd = self.grapp_params[0], self.grapp_params[1]
            # Calibration region: Center of readout (sx/Height)
            _masked_kspace_np = fastmri.tensor_to_complex_np(_masked_kspace) #this is needed as the retuned data is in format
            # (coils, H, W, 2) and grappa only takes (coils, H, W) as input so we need to convert it
            calib = _masked_kspace_np[:, :, ctr-pd:ctr+pd].copy() 
            result = grappa(_masked_kspace_np, calib=calib, kernel_size=(5,5), coil_axis=0)

            return T.to_tensor(result) #k-space data

        # ═══════════════════════════════════════
        # FULLY SAMPLED OUTPUTS
        # ═══════════════════════════════════════
        if self.K_full:
            dd["full_k_space"] = kspace

        if self.I_full:
            dd["full_i_space"] = get_ispace()

        if self.rss_combine_full:
            dd["full_rss_combined"] = get_rss()

        if self.rss_fft_full:
            dd["full_rss_fft"] = torch.fft.fftshift(
                torch.fft.fft2(get_rss(), norm='ortho')
            )

        # ═══════════════════════════════════════
        # MASKED INPUTS
        # ═══════════════════════════════════════
        if self.K:
            mk, _ = get_masked()
            dd["masked_k_space"] = mk

        if self.I:
            dd["masked_i_space"] = get_masked_ispace()

        if self.rss_combine:
            dd["masked_rss_combined"] = get_masked_rss()

        if self.rss_fft:
            dd["masked_rss_combined_fft"] = torch.fft.fftshift(
                torch.fft.fft2(get_masked_rss(), norm='ortho')
            )

        if self.mask:
            _, m = get_masked()
            dd["mask"] = m
        
        if self.enable_espirit:
            dd['sensitivity_maps'] = get_espirit_sensitivity_maps()

        if self.enable_grappa:
            dd['grappa_output'] = get_grappa_op()

        return dd