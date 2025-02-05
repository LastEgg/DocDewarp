import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
from torchmetrics import TotalVariation
from torchvision.utils import make_grid
import itertools
import wandb

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

        self.gamma1 = 0.1



    def forward(self, data, **kwargs):
        image = data["image"]

        return image

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
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.3)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        image, label, label_2x, label_4x, label_8x = (batch["image"], batch["label"], batch["label_2x"], batch["label_4x"], batch["label_8x"])
        out, out2, out4, out8 = self.model(image)  # 前向传播
        
        ssim_loss = self.gamma1 * (1 - ssim(out, label, data_range=1.0))
        l1_loss = self.l1_loss_fn(out, label)


        ssim_loss_2x = self.gamma1 * (1 - ssim(out2, label_2x, data_range=1.0))
        l1_loss_2x = self.l1_loss_fn(out2, label_2x)

        ssim_loss_4x = self.gamma1 * (1 - ssim(out4, label_4x, data_range=1.0))
        l1_loss_4x = self.l1_loss_fn(out4, label_4x)

        l1_loss_8x = self.l1_loss_fn(out8, label_8x)

        loss = l1_loss + l1_loss_2x + l1_loss_4x + l1_loss_8x + ssim_loss + ssim_loss_2x + ssim_loss_4x
        self.log("loss/train_l1_loss", l1_loss + l1_loss_2x + l1_loss_4x + l1_loss_8x) 
        self.log("loss/train_ssim_loss", ssim_loss + ssim_loss_2x + ssim_loss_4x)
        self.log("loss/train_loss", loss)  # 记录训练损失
        self.log("trainer/learning_rate", self.optimizer.param_groups[0]["lr"])
        if batch_idx == 0:
            self.log_images(image, out, label, tag="visual/train", limit_samples=4)
        return loss
    

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        image, label, label_2x, label_4x, label_8x = (batch["image"], batch["label"], batch["label_2x"], batch["label_4x"], batch["label_8x"])
        out, out2, out4, out8 = self.model(image)  # 前向传播
        
        ssim_loss = self.gamma1 * (1 - ssim(out, label, data_range=1.0))
        l1_loss = self.l1_loss_fn(out, label)

        ssim_loss_2x = self.gamma1 * (1 - ssim(out2, label_2x, data_range=1.0))
        l1_loss_2x = self.l1_loss_fn(out2, label_2x)

        ssim_loss_4x = self.gamma1 * (1 - ssim(out4, label_4x, data_range=1.0))
        l1_loss_4x = self.l1_loss_fn(out4, label_4x)

        l1_loss_8x = self.l1_loss_fn(out8, label_8x)

        loss = l1_loss + l1_loss_2x + l1_loss_4x + l1_loss_8x + ssim_loss + ssim_loss_2x + ssim_loss_4x
        self.log("loss/val_l1_loss", l1_loss + l1_loss_2x + l1_loss_4x + l1_loss_8x) 
        self.log("loss/val_ssim_loss", ssim_loss + ssim_loss_2x + ssim_loss_4x)
        self.log("loss/val_loss", loss)  # 记录训练损失
        if batch_idx == 0:
            self.log_images(image, out, label, tag="visual/val", limit_samples=4)
        return loss

    def on_train_epoch_end(self):
        """训练周期结束时执行"""
        pass

    def on_validation_epoch_end(self):
        """验证周期结束时执行"""
        pass

    def log_images(
        self,
        images_input: torch.Tensor,
        images_output: torch.Tensor,
        albedo: torch.Tensor,
        tag: str,
        limit_samples,
    ):

        if limit_samples:
            images_input = images_input[:limit_samples][:,0:3,:,:]
            images_output = images_output[:limit_samples]
            albedo = albedo[:limit_samples]

        data = torch.stack(
            list(
                itertools.chain.from_iterable(zip(images_input, images_output, albedo))
            )
        )

        grid = make_grid(data, nrow=3)

        self.logger.experiment.log({tag: [wandb.Image(grid)]}) 