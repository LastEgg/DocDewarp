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