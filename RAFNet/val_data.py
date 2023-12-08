import transforms as T
from data_utils import *
from datasets.cityscapes import Cityscapes2
from datasets.camvid import Camvid
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


def get_cityscapes(root, val_input_size,val_label_size,class_uniform_pct,val_split,num_workers):
    #assert(boost_rare in [True,False])
    val_transform=build_val_transform(val_input_size,val_label_size)
    val = Cityscapes2(root, split=val_split, target_type="semantic",
                      transforms=val_transform,class_uniform_pct=class_uniform_pct)
    val_loader = get_dataloader_val(val, num_workers)
    return val_loader
def get_camvid(root, val_input_size,val_label_size,class_uniform_pct,val_split,num_workers):
    val_transform=build_val_transform(val_input_size,val_label_size)
    val=Camvid(root,val_split,transforms=val_transform)
    val_loader = get_dataloader_val(val, num_workers)
    return val_loader
    
def count_class_nums(data_loader,num_classes):
    class_counts=[0 for _ in range(num_classes)]
    for t,(image,target) in enumerate(data_loader):
        for i in range(num_classes):
            if i in target:
                class_counts[i]+=1
        if (t+1)%100==0:
            print(f"{t+1} done.")
    print(class_counts)