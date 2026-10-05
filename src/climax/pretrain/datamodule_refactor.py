# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Dict, Optional

import numpy as np
import torch
import torchdata.datapipes as dp
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms




def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])

    lead_times = torch.stack([batch[i][2] for i in range(len(batch))])
    variables = batch[0][3]
    out_variables = batch[0][4]
    mask = torch.stack([batch[i][5] for i in range(len(batch))])
    
    file_name =[batch[i][6] for i in range(len(batch))]
    
    return (
        inp,
        out,
        lead_times,
        [v for v in variables],
        [v for v in out_variables],
        mask,
        file_name
    )