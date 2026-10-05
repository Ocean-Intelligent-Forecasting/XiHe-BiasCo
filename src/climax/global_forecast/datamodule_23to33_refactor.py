# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import random
from climax.pretrain.datamodule_refactor import collate_fn
from climax.pretrain.dataset_refactor import NpyReader



class GlobalForecastDataModule(LightningDataModule):
    """DataModule for global forecast data.

    Args:
        root_dir (str): Root directory for sharded data.
        variables (list): List of input variables.
        buffer_size (int): Buffer size for shuffling.
        out_variables (list, optional): List of output variables.
        predict_range (int, optional): Predict range.
        hrs_each_step (int, optional): Hours each step.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of workers.
        pin_memory (bool, optional): Whether to pin memory.
    """

    def __init__(
            self,
            root_dir,
            in_variables,
            variables,
            buffer_size,
            out_variables=None,
            predict_range: int = 6,
            hrs_each_step: int = 1,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            num_nodes: int = 1
    ):
        super().__init__()
        # if num_workers > 1:
        #     raise NotImplementedError(
        #         "num_workers > 1 is not supported yet. Performance will likely degrage too with larger num_workers."
        #     )

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        if isinstance(out_variables, str):
            out_variables = [out_variables]
            self.hparams.out_variables = out_variables

        # 返回各文件的路径
        self.lister_train = list(dp.iter.FileLister("/public/home/acct230421094230/901/LZP/XiHe_DC_data_deep/1day/train"))
        self.lister_train_label = sorted([os.path.join("/public/home/acct230421094230/data_23to33/train", os.path.basename(f)[5:]) for f in self.lister_train])
        
        self.lister_val = list(dp.iter.FileLister("/public/home/acct230421094230/901/LZP/XiHe_DC_data_deep/1day/val"))
        self.lister_val_label = sorted([os.path.join("/public/home/acct230421094230/data_23to33/train", os.path.basename(f)[5:]) for f in self.lister_val])

        self.lister_test = sorted(list(dp.iter.FileLister("/public/home/acct230421094230/901/LZP/XiHe_DC_data_deep/1day/test")))
        self.lister_test_label = sorted(list(dp.iter.FileLister("/public/home/acct230421094230/901/LZP/XiHe_DC_data_deep/1day/test")))
        
        random.seed(42)
          
        train_data_files = self.lister_train
        val_data_files = self.lister_val
                                                        
        train_label_files = self.lister_train_label
        val_label_files = self.lister_val_label
        
        print(f"val data files length {len(val_data_files)}  val label length {len(val_label_files)}")
        print(f"test data files length{len(self.lister_test)}  test lable length {len(self.lister_test_label)}")

        print(f"##########check data path for train {train_data_files[0]}##############check data path for val {val_data_files[0]}######################")

        print(f"##########check label path for train {train_label_files[0]}##############check label path for val {val_label_files[0]}######################")

        self.train_data_dp = train_data_files
        self.val_data_dp = val_data_files
        self.train_label_dp = train_label_files
        self.val_label_dp = val_label_files
        #  /public/home/xujianbo/all_data/data_23to33
        self.transforms = self.get_normalize(in_variables)
        self.output_transforms = self.get_normalize(out_variables)
        self.var_map = self.create_var_map()
        # self.val_clim = self.get_climatology("val", out_variables)
        # self.test_clim = self.get_climatology("test", out_variables)

    def get_label_pth(self, data_path):
        filename = os.path.splitext(os.path.basename(data_path))[0]
        return os.path.join("/public/home/xujianbo/all_data/lzp_data/truevalue/deep_true", f"{filename}.npy")

    def create_var_map(self, variables=None):
        if variables is None:
            variables = self.hparams.variables
        var_map = {}
        idx = 0
        for var in variables:
            var_map[var] = idx
            idx += 1
        return var_map

    def get_normalize(self, variables=None):
        if variables is None:
            variables = self.hparams.variables
        normalize_mean = dict(np.load(os.path.join(self.hparams.root_dir, "normalize_mean_50.npz")))
        mean = []
        for var in variables:
            if var != "sst":
               mean.append(normalize_mean[var])
            else:
               mean.append(normalize_mean[var]-273.15)
            # mean.append(normalize_mean[var])
        # 所有变量的均值 拼接到一起
        normalize_mean = np.concatenate(mean)
        #print('normalize_mean:', normalize_mean)
        normalize_std = dict(np.load(os.path.join(self.hparams.root_dir, "normalize_std_50.npz")))
        normalize_std = np.concatenate([normalize_std[var] for var in variables])
        return transforms.Normalize(normalize_mean, normalize_std)

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npz"))['arr_0']
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npz"))['arr_0']
        return lat, lon

    def get_climatology(self, partition="val", variables=None):
        path = os.path.join(self.hparams.root_dir, partition, "climatology.npz")
        clim_dict = np.load(path)
        if variables is None:
            variables = self.hparams.variables
        clim = np.concatenate([clim_dict[var] for var in variables])
        clim = torch.from_numpy(clim)
        return clim

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        # setup data for each machine

        self.data_train = NpyReader(file_list=self.train_data_dp,
                                    file_list_label=self.train_label_dp,
                                    variables=self.hparams.in_variables,
                                    out_variables=self.hparams.out_variables,
                                    var_map=self.var_map,
                                    predict_range=self.hparams.predict_range,
			        	hrs_each_step=self.hparams.hrs_each_step,
                                    transforms=self.transforms,
                                    output_transforms=self.output_transforms)

        self.data_val = NpyReader(file_list=self.val_data_dp,
                                  file_list_label=self.val_label_dp,
                                  variables=self.hparams.in_variables,
                                  out_variables=self.hparams.out_variables,
                                  var_map=self.var_map,
                                  predict_range=self.hparams.predict_range,
								  hrs_each_step=self.hparams.hrs_each_step,
                                  transforms=self.transforms,
                                  output_transforms=self.output_transforms)
        self.data_test = NpyReader(file_list=self.lister_test,
                                   file_list_label=self.lister_test_label,
                                   variables=self.hparams.in_variables,
                                   out_variables=self.hparams.out_variables,
                                   var_map=self.var_map,
                                   predict_range=self.hparams.predict_range,
								   hrs_each_step=self.hparams.hrs_each_step,
                                   transforms=self.transforms,
                                   output_transforms=self.output_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )
