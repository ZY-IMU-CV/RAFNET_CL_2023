from losses import *
from lr_schedulers import poly_lr_scheduler,cosine_lr_scheduler,step_lr_scheduler,exp_lr_scheduler
from data import get_cityscapes, get_camvid,build_val_transform,Cityscapes2
from model import RegSeg
import torch

def get_lr_function(config,total_iterations):
    # get the learning rate multiplier function for LambdaLR
    name=config["lr_scheduler"]
    warmup_iters=config["warmup_iters"]
    warmup_factor=config["warmup_factor"]
    if "poly"==name:
        p=config["poly_power"]
        return lambda x : poly_lr_scheduler(x,total_iterations,warmup_iters,warmup_factor,p)
    elif "polydul"==name:
        p=config["poly_power"]
        segitr = config["dul_poly_power"]
        return lambda x : poly_lr_scheduler_Dul(x,total_iterations,warmup_iters,warmup_factor,segitr,p)
    elif "step"==name:
        return lambda x : step_lr_scheduler(x,total_iterations,warmup_iters,warmup_factor)
    elif "exp"==name:
        beta=config["exp_beta"]
        return lambda x : exp_lr_scheduler(x,total_iterations,warmup_iters,warmup_factor,beta)
    else:
        raise NotImplementedError()

def get_loss_fun(config):
    train_crop_size=config["train_crop_size"]
    ignore_value=config["ignore_value"]
    if isinstance(train_crop_size,int):
        crop_h,crop_w=train_crop_size,train_crop_size
    else:
        crop_h,crop_w=train_crop_size
    loss_type="cross_entropy"
    if "loss_type" in config:
        loss_type=config["loss_type"]
    if loss_type=="cross_entropy":
        loss_fun=torch.nn.CrossEntropyLoss(ignore_index=ignore_value)
    elif loss_type=="bootstrapped":
        # 8*768*768/16
        minK=int(config["batch_size"]*crop_h*crop_w/16)
        print(f"bootstrapped minK: {minK}")
        loss_fun=BootstrappedCE(minK,0.3,ignore_index=ignore_value)
    elif loss_type == "OHEM":
        Min = config["batch_size"]/2
        loss_fun = OhemCrossEntropy2d(thresh=0.7,n_min=100000,ignore_lb=ignore_value)
    else:
        raise NotImplementedError()
    return loss_fun

def get_optimizer(model,config):
    if not config["bn_weight_decay"]:
        p_bn = [p for n, p in model.named_parameters() if "bn" in n]
        p_non_bn = [p for n, p in model.named_parameters() if "bn" not in n]
        optim_params = [
            {"params": p_bn, "weight_decay": 0},
            {"params": p_non_bn, "weight_decay": config["weight_decay"]},
        ]
    else:
        optim_params = model.parameters()
    return torch.optim.SGD(
        optim_params,
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )

def get_val_dataset(config):
    val_input_size=config["val_input_size"]
    val_label_size=config["val_label_size"]

    root=config["dataset_dir"]
    name=config["dataset_name"]
    val_split=config["val_split"]
    if name=="cityscapes":
        val_transform=build_val_transform(val_input_size,val_label_size)
        val = Cityscapes2(root, split=val_split, target_type="semantic",
                      transforms=val_transform,class_uniform_pct=0)
    else:
        raise NotImplementedError()
    return val

def get_dataset_loaders(config):
    name=config["dataset_name"]
    if name=="cityscapes":
        train_loader, val_loader,train_set=get_cityscapes(
            config["dataset_dir"],#数据地址root
            config["batch_size"],#批量
            config["train_min_size"],#训练最小尺寸
            config["train_max_size"],#最大尺寸
            config["train_crop_size"],#训练图大小
            config["val_input_size"],
            config["val_label_size"],
            config["aug_mode"],
            config["class_uniform_pct"],
            config["train_split"],
            config["val_split"],
            config["num_workers"],
            config["ignore_value"]
        )
    elif name=="camvid":
        train_loader, val_loader,train_set=get_camvid(
            config["dataset_dir"],
            config["batch_size"],
            config["train_min_size"],
            config["train_max_size"],
            config["train_crop_size"],
            config["val_input_size"],
            config["val_label_size"],
            config["aug_mode"],
            config["train_split"],
            config["val_split"],
            config["num_workers"],
            config["ignore_value"]
        )
    else:
        raise NotImplementedError()
    print("train size:", len(train_loader))
    print("val size:", len(val_loader))
    return train_loader, val_loader,train_set

def get_model(config):
    pretrained_backbone=config["pretrained_backbone"]
    if config["resume"]:
        pretrained_backbone=False
    model_type=config["model_type"]
    if model_type=="experimental2" or model_type=="regseg":
        ablate_decoder=False
        if "ablate_decoder" in config:
            ablate_decoder=config["ablate_decoder"]
        change_num_classes=False
        if "change_num_classes" in config:
            change_num_classes=config["change_num_classes"]
        return RegSeg(
            name=config["model_name"],
            num_classes=config["num_classes"],
            pretrained=config["pretrained_path"],
            ablate_decoder=ablate_decoder,
            change_num_classes=change_num_classes
        )
    else:
        raise NotImplementedError()
