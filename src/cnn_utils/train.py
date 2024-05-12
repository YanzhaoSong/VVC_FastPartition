"""
功能：多标签分类模型的寻来你，包含二分类

"""

import torch
from torch import nn
from typing import Optional, Iterable, List
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from sklearn import metrics

from data import load_data_cnn
from cnn_models import focal_loss


def evaluate(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        device: torch.device,
        batch_size: int,
        val_data_loader: Iterable,
        log_path_prefix: str,
        epoch: int,
        qp: str,
        shape: str,
        save_model=False,
        best_loss=10,  #
):
    model.eval()
    with torch.no_grad():
        loss_one_epoch = []  #
        predicts_one_epoch = []
        score_one_epoch = []
        targets_one_epoch = []

        for i, samples in enumerate(val_data_loader):
            images, targets = samples
            images, targets = images.to(device=device), targets.to(device=device)
            outputs = model(images)
            loss = criterion(outputs, targets)  #

            # 记录
            score_one_epoch.extend(torch.softmax(outputs, dim=1).cpu().detach().numpy())
            predicts = torch.argmax(outputs, dim=1)  #
            predicts_one_epoch.extend(predicts.cpu().detach().numpy())
            loss_one_epoch.append(loss.item())
            targets_one_epoch.extend(targets.cpu().detach().numpy())

        # 计算指标，并且保存数据、模型
        val_loss = np.mean(loss_one_epoch)
        val_accuracy = metrics.accuracy_score(y_true=targets_one_epoch, y_pred=predicts_one_epoch)
        val_top2_accuracy = metrics.top_k_accuracy_score(y_true=targets_one_epoch, y_score=score_one_epoch, k=2, labels=[0, 1, 2, 3, 4, 5])
        val_confusion_matrix = metrics.confusion_matrix(y_true=targets_one_epoch, y_pred=predicts_one_epoch, normalize='pred')
        val_classification_report = metrics.classification_report(y_true=targets_one_epoch, y_pred=predicts_one_epoch)
        # 写入文件,路径：log/1202/MyNet/0/QP32_32x32
        with open(os.path.join(log_path_prefix, f'validate_report.txt'), mode='a', newline='\n') as f:
            f.write(f"Validate_{epoch}: \n"
                    f"Loss: {val_loss}, Accuracy: {val_accuracy}, Top-2 Accuracy: {val_top2_accuracy}\n")  # , Top-2 Accuracy: {val_top2_accuracy}
            np.savetxt(f, val_confusion_matrix, fmt='%6f', delimiter=', ', header='Confusion_matrix: ')
            f.write(f'\nClassification_Report: \n')
            f.write(val_classification_report)
            f.write('\n')

        # 模型保存,路径：log/1202/MyNet/0/QP32_32x32/QP32_32x32.pth
        if save_model is True and best_loss - val_loss > 0.001:  #
            model_path = os.path.join(log_path_prefix, f'{model.model_name}_QP{qp}_{shape}.pth')
            torch.save(model.state_dict(), model_path)

    return val_loss, val_accuracy


def train(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: torch.device,
        epochs: int,
        batch_size: int,
        root_dir: str,
        qp: str,
        shape: str,
        log_path_prefix: str,
        train_writer=None,
        val_writer=None,
        use_data_normal=True,
        val_ratio=0.2,
        num_workers=1,
        save_model=False,
        early_stopping_patience=None,
        random_state=None,
):
    if random_state is not None:
        torch.manual_seed(random_state)
    model = model.to(device=device)

    train_data_loader, val_data_loader, class_weights = load_data_cnn(
        root_dir=root_dir,
        qp=qp,
        shape=shape,
        batch_size=batch_size,
        num_workers=num_workers,
        val_ratio=val_ratio,
        data_normal=use_data_normal,
        random_state=random_state,
    )

    criterion = focal_loss(alpha=class_weights, gamma=2.0, device=device)

    if train_writer is not None:
        train_writer_accuracy, train_writer_loss = train_writer
    if val_writer is not None:
        val_writer_accuracy, val_writer_loss = val_writer

    train_length = len(train_data_loader.dataset)
    train_num_batches = len(train_data_loader)
    best_val_loss = np.inf
    current_patience = 0  #

    # train
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_accu_sum = 0.0
        with tqdm(total=train_length, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for i, batch in enumerate(train_data_loader):
                images, targets = batch
                images, targets = images.to(device=device), targets.to(device=device)
                output = model(images)

                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                accuracy = (torch.sum((torch.argmax(output, dim=1) == targets)) / batch_size).item()
                loss = loss.item()

                train_loss_sum += loss
                train_accu_sum += accuracy

                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss': loss, 'accuracy': accuracy})

        scheduler.step(epoch)

        train_loss = train_loss_sum / train_num_batches
        train_accuracy = train_accu_sum / train_num_batches

        val_loss, val_accuracy = evaluate(
            model=model,
            criterion=criterion,
            device=device,
            batch_size=batch_size,
            val_data_loader=val_data_loader,
            log_path_prefix=log_path_prefix,
            epoch=epoch,
            save_model=save_model,
            qp=qp,
            shape=shape,
            best_loss=best_val_loss,
        )
        # 记录数据
        if val_writer is not None and train_writer is not None:
            train_writer_accuracy.add_scalar('accuracy', train_accuracy, epoch)
            train_writer_loss.add_scalar('loss', train_loss, epoch)
            val_writer_accuracy.add_scalar('accuracy', val_accuracy, epoch)
            val_writer_loss.add_scalar('loss', val_loss, epoch)

        # 更新loss
        if best_val_loss - val_loss > 0.001:
            best_val_loss = val_loss
            current_patience = 0
        else:
            current_patience += 1

        # early stop
        if early_stopping_patience is not None:
            if current_patience >= early_stopping_patience:
                print("Early Stopping!")
                break

    train_writer_accuracy.close()
    train_writer_loss.close()
    val_writer_accuracy.close()
    val_writer_loss.close()
