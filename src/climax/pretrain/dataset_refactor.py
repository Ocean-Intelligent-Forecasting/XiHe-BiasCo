# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
import random

import numpy as np
import torch
import logging
from torch.utils.data import Dataset
FORMAT = '[%(levelname)s: %(filename)s:%(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT,filename='../train.log',filemode='a')
logger = logging.getLogger(__name__)
class NpyReader(Dataset):
    def __init__(self, file_list, file_list_label, variables, out_variables, var_map, predict_range, hrs_each_step,
                transforms:torch.nn.Module,
                output_transforms:torch.nn.Module):
        self.file_list = file_list
        self.file_list_label = file_list_label
        self.variables = variables
        self.out_variables = out_variables
        self.var_map = var_map
        self.predict_range = predict_range
        self.hrs_each_step=hrs_each_step
        self.transforms = transforms
        self.output_transforms = output_transforms
    def __len__(self):

        #return len(self.file_list) - self.predict_range
        return len(self.file_list)

    def __getitem__(self,idx):

        path_x = self.file_list[idx]
        data_x = np.load(path_x)
        data_x[:, [1,2], :, :] = 0
        path_y = self.file_list_label[idx]
        data_y = np.load(path_y)

        # path_x = self.file_list[idx]
        # data_x = np.load(path_x)
        # path_y = self.file_list[idx+self.predict_range]
        # data_y = np.load(path_y)
        
        index_in = []
        index_out = []
        for var in self.variables:
            index_in.append(self.var_map[var])
        data_x = data_x[0, index_in, :, :]
        for var in self.out_variables:
            index_out.append(self.var_map[var])
        data_y = data_y[0, index_out, :, :]
        
		
        inp = torch.from_numpy(data_x)
        out = torch.from_numpy(data_y)

        inp = inp.float()
        out = out.float()
        
        predict_range = torch.ones(1).to(torch.long) * self.predict_range
        lead_time = self.hrs_each_step * predict_range / 100
        lead_time = lead_time.to(inp.dtype)
		
        inp[np.isnan(inp).bool()] = -32767
        inp[torch.isinf(inp)] = -32767
        mask_inp = inp < -30000 
        x = self.transforms(inp)
        #x = inp
        x[mask_inp] = 0
        out[np.isnan(out).bool()] = -32767
        out[torch.isinf(out)] = -32767
        mask_out = out < -30000
        y = self.output_transforms(out)
        #y = out
        y[mask_out] = 0
		
        
        
        mask = out > -30000

        file_name = path_y.split('/')[-1]
        
        return x, y, lead_time, self.variables, self.out_variables, mask, file_name
