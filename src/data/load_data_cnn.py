import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import sklearn
import os
from PIL import Image


class MyDataSet(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            img_dir: str,
            transform=None,
    ):
        super().__init__()

        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, item):
        seq_name, poc, ctu_id, x, y, w, h = self.df.loc[
            item, ['SequenceName', 'poc', 'ctuIdx', 'xRelativeToCTU', 'yRelativeToCTU', 'width', 'height']]

        img = self.load_image(str(seq_name), int(poc), int(ctu_id), int(x), int(y), int(w), int(h))

        if self.transform is not None:
            img = self.transform(img)  #

        label = self.df.loc[item, 'splitMode']
        label = torch.tensor(label, dtype=torch.long)
        sample = (img, label)

        return sample

    def __len__(self):
        return self.df.shape[0]

    def load_image(self, seq_name, poc, ctu_id, x, y, w, h):
        img_filedir = os.path.join(self.img_dir, seq_name, str(poc), str(ctu_id) + '.jpg')
        image = Image.open(img_filedir)
        image = image.crop((x, y, x + w, y + h))

        return image


def load_data_cnn(
        root_dir: str,
        qp: str,
        shape: str,
        batch_size: int,
        num_workers=1,
        val_ratio=0.2,
        data_normal=False,
        random_state=None,
):
    img_dir = os.path.join(root_dir, 'images')
    pickle_dir = os.path.join(root_dir, 'pickles')

    df = pd.read_pickle(os.path.join(pickle_dir, shape, f'QP{qp}_{shape}.pkl'))

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if data_normal:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.4062, std=0.0688),
        ])

    # 训练、验证集划分
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=random_state,
                                        shuffle=True, stratify=df['splitMode'])

    # 注意
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # 计算 class_weights
    class_weights = sklearn.utils.class_weight.compute_class_weight(
        class_weight='balanced', classes=[0, 1, 2, 3, 4, 5], y=train_df['splitMode'])

    # 构造 Dataset
    train_dataset = MyDataSet(df=train_df, img_dir=img_dir, transform=data_transform)
    val_dataset = MyDataSet(df=val_df, img_dir=img_dir, transform=data_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=True)
    data_loader = (train_dataloader, val_dataloader, class_weights)

    return data_loader


