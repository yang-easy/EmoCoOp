from dassl.data.datasets import DatasetBase
from dassl.data.datasets.base_dataset import Datum
import pandas as pd
import numpy as np
import torch
import os

class EmotionDataset(DatasetBase):
    dataset_dir = "emotion_dataset"

    def __init__(self, cfg):
        # 所有情感标签
        self.all_emotions = ['approval', 'bloom', 'calm', 'devil', 'fly', 'guard', 'negative',
        'neutral', 'positive', 'rest', 'wilt', 'work', 'worried']
        self.emotion2idx = {emotion: idx for idx, emotion in enumerate(self.all_emotions)}
        self._num_classes = self.get_num_classes()

        # 根路径
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        csv_path = os.path.abspath(os.path.expanduser(cfg.DATASET.CSV_PATH))

        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 设置随机种子
        try:
            seed = cfg.SEED if hasattr(cfg, 'SEED') else 42
            if isinstance(seed, int) and 0 <= seed < 2**32:
                np.random.seed(seed)
            else:
                print(f"警告: 种子 {seed} 非法，使用默认值 42")
                np.random.seed(42)
        except Exception as e:
            print(f"设置种子出错: {e}，使用默认值 42")
            np.random.seed(42)

        # 打乱并划分数据
        train_ratio = cfg.DATASET.get("TRAIN_RATIO", 0.7)
        val_ratio = cfg.DATASET.get("VAL_RATIO", 0.15)
        shuffled_indices = np.random.permutation(len(df))
        train_size = int(len(df) * train_ratio)
        val_size = int(len(df) * val_ratio)
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:train_size + val_size]
        test_indices = shuffled_indices[train_size + val_size:]

        train, val = [], []

        for i, row in df.iterrows():
            img_path = os.path.join(root, row["filename"])
            emotions = str(row["label"]).split('_')
            # 确保创建的是一个浮点型张量，形状为 [num_classes]
            multi_hot = torch.zeros(len(self.all_emotions), dtype=torch.float)
            for emotion in emotions:
                if emotion in self.emotion2idx:
                    multi_hot[self.emotion2idx[emotion]] = 1.0
            
            # Store label as tensor explicitly
            datum = Datum(impath=img_path, label=multi_hot, domain=0)

            if i in train_indices:
                train.append(datum)
            elif i in val_indices:
                val.append(datum)

        # ===== ✅ 固定读取测试集（非 train.csv 的一部分） =====
        test_csv = os.path.abspath(os.path.expanduser(cfg.DATASET.get("TEST_CSV", "/home/user_3505_11/data/data_csv/test_data_coco.csv")))
        test_root = os.path.abspath(os.path.expanduser(cfg.DATASET.get("TEST_ROOT", "/home/user_3505_11/data/test_data")))

        df_test = pd.read_csv(test_csv)

        test = []
        for _, row in df_test.iterrows():
            img_path = os.path.join(test_root, row["filename"])
            emotions = str(row["label"]).split('_')
            # 同样确保测试集标签也是浮点型张量
            multi_hot = torch.zeros(len(self.all_emotions), dtype=torch.float)
            for emotion in emotions:
                if emotion in self.emotion2idx:
                    multi_hot[self.emotion2idx[emotion]] = 1.0
            
            datum = Datum(impath=img_path, label=multi_hot, domain=0)
            test.append(datum)

        print(f"数据集划分完成：训练 {len(train)}，验证 {len(val)}，测试 {len(test)}")
        print(f"情感类别数：{len(self.all_emotions)}")

        super().__init__(train_x=train, val=val, test=test)

    @property
    def classnames(self):
        return self.all_emotions

    @property
    def num_classes(self):
        return len(self.all_emotions)


    def get_num_classes(self, data_source=None):
        return len(self.all_emotions)  # 或 len(self.emotion2idx)
    def get_lab2cname(self, data_source=None):
        lab2cname = {i: cname for i, cname in enumerate(self.all_emotions)}
        return lab2cname, self.all_emotions


