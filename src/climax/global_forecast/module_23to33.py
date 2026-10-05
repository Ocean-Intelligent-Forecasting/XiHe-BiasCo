# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any

import torch
torch.set_float32_matmul_precision('high')
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms
from pytorch_lightning.utilities.grads import grad_norm
from climax.arch import ClimaX, ClimaX_FC
from climax.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from climax.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
    lat_weighted_bias
)
from climax.utils.pos_embed import interpolate_pos_embed


class GlobalForecastModule(LightningModule):
    """Lightning module for global forecasting with the ClimaX model.

    Args:
        net (ClimaX_FC): ClimaX_FC model.
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """

    def __init__(
        self,
        net: ClimaX_FC,
        pretrained_path: str = "",
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10000,
        max_epochs: int = 200000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"]) #保存超参数,net部分的超参数不保存
        self.net = net
        self.lr = lr
        self.max_epochs = max_epochs
        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        # if pretrained_path.startswith('http'):
        #     checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        # else:
        checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        # interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        state_dict = self.state_dict()
        # checkpoint_keys = list(checkpoint_model.keys())
        '''for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                #print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]'''

        # load pre-trained model

        msg = self.load_state_dict(checkpoint_model, strict=False)
        print("!!!pretrained model loaded!!!")

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon): # 经纬度
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_val_clim(self, clim):
        self.val_clim = clim

    def set_test_clim(self, clim):
        self.test_clim = clim

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.net,norm_type=2)
        self.log_dict(norms)

    def training_step(self, batch: Any, batch_idx: int):
        '''
        x:B*48*32*64 输入的变量
        y:B*n*32*64 ground truth
        lead_times: B
        variables: 48个变量名
        out_variables: 需要预测的变量名 n
        '''
        
        
        #x, y, lead_times, variables, out_variables, mask = batch
        x, y, lead_times, variables, out_variables, mask, _ = batch
        # x     [1, 52, 2041, 4320]
        # preds [1, 50, 2041, 4320]
        loss_dict, preds = self.net.forward(x, y, lead_times, variables, out_variables, [lat_weighted_mse], lat=self.lat, mask=mask)
        #print(x.shape, y.shape, y2.shape, preds.shape)

        loss_dict = loss_dict[0]
        for var in loss_dict.keys():
            self.log(
                "train_step1/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            #print("train/" + var, loss_dict[var])
        #print('#'*30+' loss '+'#'*30,loss_dict)
        loss = loss_dict["loss"]
        

        

        

        
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        #x, y, lead_times, variables, out_variables, mask = batch
        x, y, lead_times, variables, out_variables, mask, _ = batch
        #print(x.shape, y.shape, y2.shape, mask.shape, mask2.shape)
        #print(mask.shape)
        #print(mask)

#        if self.pred_range < 24:
#            log_postfix = f"{self.pred_range}_days"
#        else:
#            days = int(self.pred_range / 24)
#            log_postfix = f"{days}_days"
        log_postfix = f"{self.pred_range}_days"

        preds, all_loss_dicts = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_rmse],
            lat=self.lat,
            # clim=self.val_clim,
            log_postfix=log_postfix,
            mask=mask
        )
        #print(preds)
        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        #print(loss_dict)
        for var in loss_dict.keys():
            self.log(
                "val_step1/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            #print(var, loss_dict[var])
        

        
            
        

    def test_step(self, batch: Any, batch_idx: int):
        #x, y, lead_times, variables, out_variables, mask = batch
        x, y, lead_times, variables, out_variables,mask, file_name = batch
        x = torch.nan_to_num(x, nan=-32767.0)

#        if self.pred_range < 24:
#            log_postfix = f"{self.pred_range}_days"
#        else:
#            days = int(self.pred_range / 24)
#            log_postfix = f"{days}_days"
        log_postfix = f"{self.pred_range}_days"
        preds, all_loss_dicts = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_rmse, lat_weighted_bias],
            lat=self.lat,
            log_postfix=log_postfix,
            mask=mask,
            file_name = file_name
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "test_step1/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0,
                },
            ]
        )
        #print("warmup_epoch", self.hparams.warmup_epochs, self.hparams.max_epochs)

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        # self.on_before_optimizer_step(optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}