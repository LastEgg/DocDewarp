import torch
from torchmetrics import ConfusionMatrix, F1Score
import lightning.pytorch as pl
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim

torch.set_float32_matmul_precision("high")


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader, len_valloader):
        super().__init__()
        self.learning_rate = opt.learning_rate  # 学习率
        self.len_trainloader = len_trainloader  # 训练数据加载器长度
        self.len_valloader = len_valloader 
        self.opt = opt  # 配置参数
        self.model = model  # 模型
        self.l1_loss_fn = torch.nn.L1Loss()
        self.mse_loss_fn = torch.nn.MSELoss()
        self.dev = torch.device(f'cuda:{opt.devices}')
        self.avg_ssim = torch.zeros([]).to(self.dev)
        self.avg_ms_ssim = torch.zeros([]).to(self.dev)


    def forward(self, x):
        """前向传播"""
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        """配置优化器和学习率 Scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.opt.epochs,
            steps_per_epoch=self.len_trainloader,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        image, label = (batch["image"], batch["label"])
        prediction = self(image)  # 前向传播
        pred_nhwc = prediction.permute([0, 2, 3, 1])
        l1_loss = self.l1_loss_fn(pred_nhwc, label)
        loss = l1_loss
        self.log("loss/train_l1_loss", l1_loss)  # 记录训练交叉熵损失
        self.log("loss/train_loss", loss)  # 记录训练损失
        self.log("trainer/learning_rate", self.optimizer.param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        image, label = (batch["image"], batch["label"])
        prediction = self(image)  # 前向传播
        pred_nhwc = prediction.permute([0, 2, 3, 1])
        out = F.grid_sample(image, (pred_nhwc / self.opt.image_size - 0.5) * 2)
        out_gt = F.grid_sample(image, (label / self.opt.image_size - 0.5) * 2)
        ssim_val = ssim(out, out_gt, data_range=1.0)
        ms_ssim_val = ms_ssim(out, out_gt, data_range=1.0)
        l1_loss = self.l1_loss_fn(pred_nhwc, label)
        loss = l1_loss
        self.avg_ssim += ssim_val
        self.avg_ms_ssim += ms_ssim_val
        self.log("loss/valid_l1_loss", l1_loss)  # 记录验证交叉熵损失
        self.log("loss/valid_loss", loss)  # 记录验证损失

    def on_train_epoch_end(self):
        """训练周期结束时执行"""
        pass

    def on_validation_epoch_end(self):
        """验证周期结束时执行"""
        self.avg_ssim /= self.len_valloader
        self.avg_ms_ssim /= self.len_valloader

        self.log("metric/avg_ssim",self.avg_ssim)
        self.log("metric/avg_ms_ssim", self.avg_ms_ssim)  # 记录整体F1分数
        # 清空存储
  
        self.avg_ssim = torch.zeros([]).to(self.dev)
        self.avg_ms_ssim = torch.zeros([]).to(self.dev)
