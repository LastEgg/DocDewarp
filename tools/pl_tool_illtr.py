import torch

from torchmetrics import ConfusionMatrix, F1Score
import lightning.pytorch as pl
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
from .losses.vgg_pretrained_loss import VGGPerceptualLoss
from .utils import load_model

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
        self.vgg = VGGPerceptualLoss()
        self.alpha = 1e-5

        if opt.pretrained:
            print("-" * 30)
            msg = load_model(self.model, opt.pretrained)
            print(f"pretrained model loading: {msg}")
            print("-" * 30)


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
        image, label = (batch["image"], batch["label"])
        prediction = self.model(image)  # 前向传播

        l1_loss = self.l1_loss_fn(prediction, label)
        vgg_loss = self.vgg(prediction, label)
        loss = l1_loss + self.alpha * vgg_loss
        self.log("loss/train_l1_loss", l1_loss) 
        self.log("loss/train_vgg_loss", vgg_loss)
        self.log("loss/train_loss", loss)  # 记录训练损失
        self.log("trainer/learning_rate", self.optimizer.param_groups[0]["lr"])
        if batch_idx == 0:
            self.log_images(image, prediction, label, tag="visual/train", limit_samples=8)
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        image, label = (batch["image"], batch["label"])
        prediction = self.model(image)  # 前向传播

        l1_loss = self.l1_loss_fn(prediction, label)
        vgg_loss = self.vgg(prediction, label)
        loss = l1_loss + self.alpha * vgg_loss
        self.log("loss/val_l1_loss", l1_loss) 
        self.log("loss/val_vgg_loss", vgg_loss)
        self.log("loss/valid_loss", loss)  # 记录训练损失
        if batch_idx == 0:
            self.log_images(image, prediction, label, tag="visual/val", limit_samples=8)

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
            images_input = images_input[:limit_samples]
            images_output = images_output[:limit_samples]
            albedo = albedo[:limit_samples]

        data = torch.stack(
            list(
                itertools.chain.from_iterable(zip(images_input, images_output, albedo))
            )
        )

        grid = make_grid(data, nrow=3)

        self.logger.experiment.log({tag: [wandb.Image(grid)]}) 