#!/usr/bin/env python
# _*_coding:utf-8 _*_
import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import transforms as T
random.seed(1)
rmb_label = {"1": 0, "100": 1}

class bddDataset(Dataset):
    def __init__(self, data_dir, transforms, flag):

        self.data_dir = data_dir
        self.transforms = transforms
        if flag == 'train':
            self.img_dir = os.path.join(data_dir, "images/", '10k/', 'train/')
            self.mask_dir = os.path.join(data_dir, "labels/", 'sem_seg/','masks/','train/')
        if flag == 'val':
            self.img_dir = os.path.join(data_dir, "images/", '10k/', 'val/')
            self.mask_dir = os.path.join(data_dir, "labels/", 'sem_seg/','masks/','val/')

        self.names = [name[:-4] for name in list(filter(lambda x: x.endswith(".jpg"), os.listdir(self.img_dir)))]
        #self.label_data = json.load(open(self.json_dir, 'r', encoding='UTF-8'))

    def __getitem__(self, index):
        """
        返回img和target
        :param idx:
        :return:
        """

        name = self.names[index]
        path_img = os.path.join(self.img_dir, name + ".jpg")
        path_labels=os.path.join(self.mask_dir, name + ".png")
        # path_json = self.json_dir

        # load img
        img = Image.open(path_img).convert("RGB")
        target = Image.open(path_labels).convert("L")

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        if len(self.names) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.names)

def collate_fn(batch):
    return tuple(zip(*batch))
def build_train_transform2(train_min_size, train_max_size, train_crop_size, aug_mode,ignore_value):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    fill = tuple([int(v * 255) for v in mean])
    #ignore_value = 255
    edge_aware_crop=False
    resize_mode="uniform"
    transforms = []
    transforms.append(
        T.RandomResize(train_min_size, train_max_size, resize_mode)
    )
    if isinstance(train_crop_size,int):
        crop_h,crop_w=train_crop_size,train_crop_size
    else:
        crop_h,crop_w=train_crop_size
    transforms.append(
        T.RandomCrop2(crop_h,crop_w,edge_aware=edge_aware_crop)
    )
    transforms.append(T.RandomHorizontalFlip(0.5))
    if aug_mode == "baseline":
        pass
    elif aug_mode == "randaug":
        transforms.append(T.RandAugment(2, 0.2, "full",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
    elif aug_mode=="randaug_reduced":
        transforms.append(T.RandAugment(2, 0.2, "reduced",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
    elif aug_mode== "colour_jitter":
        transforms.append(T.ColorJitter(0.3, 0.3,0.3, 0,prob=1))
    elif aug_mode=="rotate":
        transforms.append(T.RandomRotation((-10,10), mean=fill, ignore_value=ignore_value,prob=1.0,expand=False))
    elif aug_mode=="noise":
        transforms.append(T.AddNoise(15,prob=1.0))
    elif aug_mode=="noise2":
        transforms.append(T.AddNoise2(10,prob=1.0))
    elif aug_mode=="noise3":
        transforms.append(T.AddNoise3(10,prob=1.0))
    elif aug_mode == "custom1":
        transforms.append(T.RandAugment(2, 0.2, "reduced",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
        transforms.append(T.AddNoise(10,prob=0.2))
    elif aug_mode == "custom2":
        transforms.append(T.RandAugment(2, 0.2, "reduced2",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
        transforms.append(T.AddNoise(10,prob=0.1))
    elif aug_mode=="custom3":
        transforms.append(T.ColorJitter(0.3, 0.4,0.5, 0,prob=1))
    else:
        raise NotImplementedError()
    transforms.append(T.RandomPad(crop_h,crop_w,fill,ignore_value,random_pad=True))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(
        mean,
        std
    ))
    return T.Compose(transforms)
def build_val_transform(val_input_size,val_label_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms=[]
    transforms.append(
        T.ValResize(val_input_size,val_label_size)
    )
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(
        mean,
        std
    ))
    return T.Compose(transforms)
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
def get_BDD100K_dataset(root,batch_size,train_min_size,train_max_size,train_crop_size,val_input_size,val_label_size,
                        aug_mode,num_workers,ignore_value):
    train_transform = build_train_transform2(train_min_size, train_max_size, train_crop_size, aug_mode, ignore_value)
    train_set = bddDataset(data_dir=root, transforms=train_transform, flag='train')
    train_sampler = torch.utils.data.RandomSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=num_workers,
                              sampler=train_sampler,
                              worker_init_fn=worker_init_fn)

    val_transform = build_val_transform(val_input_size, val_label_size)
    val_set = bddDataset(data_dir=root, transforms=val_transform, flag='val')
    test_sampler = torch.utils.data.SequentialSampler(val_set)
    val_loader = DataLoader(val_set, batch_size=1, sampler=test_sampler, num_workers=num_workers, collate_fn=collate_fn,
                            worker_init_fn=worker_init_fn)
    return train_loader,val_loader

if __name__ =="__main__":
    # import transforms as T
    # train_transform = build_train_transform2(400,1600,[768,768],"randaug_reduced",255)
    # train_set = bddDataset(data_dir="./dataset/bdd100k/", transforms=train_transform, flag='train')
    # train_sampler = torch.utils.data.RandomSampler(train_set)
    # train_loader = DataLoader(train_set, batch_size=8, collate_fn=collate_fn,drop_last=True, num_workers=4,sampler=train_sampler,
    #                           worker_init_fn=worker_init_fn)
    #
    # val_transform = build_val_transform(768, 768)
    # val_set = bddDataset(data_dir="./dataset/bdd100k/", transforms=train_transform,flag='val')
    # test_sampler = torch.utils.data.SequentialSampler(val_set)
    # val_loader = DataLoader(val_set, batch_size=1,sampler=test_sampler, num_workers=4,collate_fn=collate_fn,worker_init_fn=worker_init_fn)
    #
    #
    #
    train_loader ,val_loader = get_BDD100K_dataset(
        root = "D:/ZHBG/CLReg/CLReg_prog/bdd100k_dataset",  # 数据地址root
        batch_size = 8,  # 批量
        train_min_size = 400,  # 训练最小尺寸
        train_max_size = 1600,  # 最大尺寸
        train_crop_size = [768,768],  # 训练图大小
        val_input_size = 1024,
        val_label_size = 1024,
        aug_mode = "randaug_reduced",
        num_workers = 6,
        ignore_value = 255)

    print(len(train_loader))
    for image,label in train_loader:
        img = torch.stack(image)
        trg = torch.stack(label)
        print(img.shape)