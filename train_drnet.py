'''
drnet + docres数据预处理 训练代码
'''
import torch
from configs.option import get_option
from tools.datasets.datasets_doc3dshade.datasets_docres import *
from tools.pl_tools.pl_tool_drnet import * 
from models import UNext_full_resolution_padding_L_py_L
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
import os

torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    opt = get_option("config_drnet_doc3dshade.yaml")
    """定义网络"""
    model = UNext_full_resolution_padding_L_py_L(num_classes=3, input_channels=6,img_size=opt.image_size)
    """模型编译"""
    # model = torch.compile(model)
    """导入数据集"""
    train_dataloader, valid_dataloader = get_dataloader(opt, prompt=True)
    """Lightning 模块定义"""
    wandb_logger = WandbLogger(
        project=opt.project,
        name=opt.exp_name,
        offline=not opt.save_wandb,
        config=opt,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=[opt.devices],
        strategy="auto",
        max_epochs=opt.epochs,
        precision=opt.precision,
        default_root_dir="./",
        logger=wandb_logger,
        val_check_interval=opt.val_check,
        log_every_n_steps=opt.log_step,
        accumulate_grad_batches=opt.accumulate_grad_batches,
        gradient_clip_val=opt.gradient_clip_val,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join("./checkpoints", opt.exp_name),
                monitor="loss/valid_loss",
                mode="min",
                save_top_k=1,
                save_last=False,
                filename="epoch_{epoch}-loss_{loss/valid_loss:.3f}",
                auto_insert_metric_name=False,  # 使用 f-string 和 replace
            ),
        ],
    )

    # Start training
    trainer.fit(
        LightningModule(opt, model, len(train_dataloader), len(valid_dataloader)),
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=opt.resume,
    )
    wandb.finish()
