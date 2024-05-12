import platform
import torch
from torch.utils.tensorboard import SummaryWriter
import timm
from timm import optim, scheduler
import os
import time
import json
from types import SimpleNamespace
import numpy as np

from cnn_utils import create_model, train
from cnn_models import focal_loss


import warnings
warnings.filterwarnings('ignore')


class Arguments:
    def __init__(self):
        # 基础设置
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.basic_args = SimpleNamespace()
        # log 路径  
        self.basic_args.date = "1210"  # time.strftime('%m%d')
        self.basic_args.log_index = '0'
        # 数据设置
        self.data_args = SimpleNamespace()
        self.data_args.root_dir = './'  # 数据存放位置 ./
        self.data_args.qp = '32'
        self.data_args.shape = '32x32'
        # 数据预处理设置
        self.data_args.use_data_normal = False

        # dataloader 相关设置
        if platform.system() == 'Linux':
            self.data_args.batch_size = 512
            self.data_args.num_workers = 16
        else:
            self.data_args.batch_size = 32
            self.data_args.num_workers = 1

        # 模型设置
        self.model_args = SimpleNamespace()
        self.model_args.model_name = 'MyNet'
        self.model_args.in_channels = 1
        self.model_args.num_classes = 6
        self.model_args.save_model = True
        self.model = create_model(args=self.model_args, from_timm=False)

        # 损失函数
        self.loss_args = SimpleNamespace()
        self.loss_args.label_smoothing = 0  # 标签平滑
        self.loss_args.use_weights = True
        # self.loss_function = torch.nn.CrossEntropyLoss(label_smoothing=self.loss_args.label_smoothing)
        self.loss_function = focal_loss(gamma=2, device=self.device)

        # 优化器设置
        self.optimizer_args = SimpleNamespace()
        self.optimizer_args.opt = 'adamw'  #
        self.optimizer_args.lr = 4.53e-3
        self.optimizer_args.weight_decay = 0
        self.optimizer_args.momentum = 0.9

        self.optimizer = optim.create_optimizer(self.optimizer_args, self.model)

        # 早停机制
        self.optimizer_args.early_stopping_patience = 10

        # scheduler 设置
        self.scheduler_args = SimpleNamespace()
        self.scheduler_args.sched = 'step'  #
        self.scheduler_args.num_epochs = 100  #
        self.scheduler_args.decay_epochs = 6  #
        self.scheduler_args.decay_rate = 0.1  #
        self.scheduler_args.min_lr = 0  #
        # warm-up 设置
        self.scheduler_args.warmup_lr = 5e-6  #
        self.scheduler_args.warmup_epochs = 5  #

        self.scheduler, _ = scheduler.create_scheduler(self.scheduler_args, self.optimizer)
        # return lr_scheduler, num_epochs

    def get_dict(self):
        config_dict = dict(
            date=str(time.strftime('%Y-%m-%d')),
            system=platform.system(),
            device=str(self.device),
            basic_cfg=self.basic_args.__dict__,
            data_cfg=self.data_args.__dict__,
            model_cfg=self.model_args.__dict__,
            loss_cfg=str(self.loss_args.__dict__),
            loss_function=str(self.loss_function),
            optimizer_cfg=self.optimizer_args.__dict__,
            scheduler_cfg=self.scheduler_args.__dict__,
        )
        return config_dict


if __name__ == "__main__":
    QPs = ['37', '32', '27', '22']  # '37', '32', '27', '22'
    Shapes = ['32x32']  # , '32x16', '16x32', '16x16', '8x32', '32x8', '32x32'

    # ----------------- 模型定义 --------------------------
    args = Arguments()

    # 循环对所有QP 和 Shape 进行训练
    for Shape in Shapes:
        for QP in QPs:
            args.data_args.qp = QP
            args.data_args.shape = Shape

            # 记录 time.strftime('%m%d')
            log_path_prefix = os.path.join('log', args.basic_args.date, args.model.model_name, args.basic_args.log_index,
                                           f'QP{args.data_args.qp}_{args.data_args.shape}')
            train_writer = (SummaryWriter(log_path_prefix + '/train_accuracy'),
                            SummaryWriter(log_path_prefix + '/train_loss'))

            val_writer = (SummaryWriter(log_path_prefix + '/val_accuracy'),
                          SummaryWriter(log_path_prefix + '/val_loss'))

            # 记录配置信息：
            output_file_path = os.path.join(log_path_prefix, 'config.json')
            json_data = json.dumps(args.get_dict(), indent=4)
            with open(output_file_path, 'w') as f:
                f.write(json_data)
            
            print(f"Starting training model: QP{QP}_{Shape} ....")
            train(
                model=args.model,
                criterion=args.loss_function,
                optimizer=args.optimizer,
                scheduler=args.scheduler,
                device=args.device,
                epochs=args.scheduler_args.num_epochs,
                batch_size=args.data_args.batch_size,
                root_dir=args.data_args.root_dir,
                qp=args.data_args.qp,
                shape=args.data_args.shape,
                train_writer=train_writer,
                val_writer=val_writer,
                log_path_prefix=log_path_prefix,
                num_workers=args.data_args.num_workers,
                use_data_normal=args.data_args.use_data_normal,
                save_model=args.model_args.save_model,
                early_stopping_patience=args.optimizer_args.early_stopping_patience,
            )
