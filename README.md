## 简介
这是一个基于PyTorch Lightning构建的文档矫正与文档增强的训练框架，用于复现各种文档矫正与增强的论文，以及一些属于自己的创新

## 主要特性
- 使用 PyTorch Lightning 构建，代码结构清晰

- 支持 Weights & Biases (wandb) 实验跟踪

- 灵活的配置系统，支持 YAML 配置和命令行参数

- 支持模型断点保存和恢复

- 内置数据可视化工具

- 支持混合精度训练

- 自动记录训练指标

## 配置说明
主要配置参数在 configs/config_*.yaml 中设置：

###  实验环境配置
- seed: 随机种子

- exp_name: 实验名称

- project: wandb 项目名称

### 数据集配置
- data_path: 数据集路径

- image_size: 输入图像大小

- num_classes: 分类类别数

### 模型配置
- model_name: 使用的模型架构

- pretrained: 是否使用预训练权重

### 训练配置
- learning_rate: 学习率

- batch_size: 批次大小

- epochs: 训练轮数

- precision: 训练精度模式

## 已有方法
| **方法名称** | **配置文件** | **数据集** |
| --- | --- | --- |
| DocTr++ | config_geotr_Doc3D.yaml | Doc3D |
| IllTr from DocTr | config_illtr_Inv3D.yaml | Inv3D |


