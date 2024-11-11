import os
import math
import logging
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import yaml
import models
from models import get_model
from os.path import join


class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)


def debatchify(out, squeeze):
    """
    debatchify a tensor by squeezing and/or transposing its dimensions,
    supporting multiple tensor format transforms (BCHW -> CHW | CHW -> HWC | HWC -> HW)
    
    :param out: The output tensor that needs to be transformed
    :param squeeze: A boolean parameter that determines whether to convert tensor format with C = 1 to HW format
    :return: the tensor `out` with simplified format.
    """
    if len(out.shape) == 4:
        out = out.squeeze(0)  # BCHW -> CHW
    if len(out.shape) == 3:
        if out.shape[0] == 3 or out.shape[0] == 1:
            out = out.transpose(1, 2, 0)  # CHW -> HWC
        if out.shape[2] == 1 and squeeze:
            out = out.squeeze(2)  # HWC -> HW
    return out


def to_ndarray(x, debatch=False, squeeze=False):
    """
    convert a given input into a numpy array and optionally remove any batch dimensions.
    
    :param x: The input data that needs to be converted to a numpy array
    :param debatch: A boolean parameter that specifies whether to remove the batch dimension from the
    input tensor or not. If set to True, the function will call the `debatchify` function to remove the
    batch dimension. If set to False, the function will return the input tensor as is, defaults to False
    (optional)
    :param squeeze: the `squeeze` boolean parameter in `debatchify`, that determines whether to convert 
    tensor format with C = 1 to HW format 
    :return: a numpy array. If `debatch` is True, the output is passed through the `debatchify` function
    with `squeeze` before being returned. 
    """
    if isinstance(x, torch.Tensor):
        out = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        out = x.astype('float32')
    else:
        out = np.array(x)
    if debatch:
        out = debatchify(out, squeeze)
    return out


def imshow(*imgs, maxcol=3, gray=False, titles=None, off_axis=False) -> None:
    """
    display one or more images in a grid with customizable parameters such as
    maximum number of columns, grayscale, and titles.
    
    :param maxcol: The maximum number of columns to display the images in. If there are more images than
    maxcol, they will be displayed in multiple rows. The default value is 3, defaults to 3 (optional)
    :param gray: A boolean parameter that determines whether the image(s) should be displayed in
    grayscale or in color. If set to True, the images will be displayed in grayscale. If set to False,
    the images will be displayed in color, defaults to False (optional)
    :param titles: titles is a list of strings that contains the titles for each image being displayed.
    If titles is None, then no titles will be displayed
    """
    import matplotlib.pyplot as plt
    if len(imgs) != 1:
        plt.figure(figsize=(10, 5))
    row = (len(imgs) - 1) // maxcol + 1
    col = maxcol if len(imgs) >= maxcol else len(imgs)
    for idx, img in enumerate(imgs):
        img = to_ndarray(img, debatch=True)
        if img.max() > 2: img = img / 255
        img = img.clip(0, 1)
        if gray: plt.gray()
        plt.subplot(row, col, idx + 1)
        plt.imshow(img)
        if titles is not None: plt.title(titles[idx])
        if off_axis: plt.axis('off')
    plt.show()


class Logger:
    def __init__(self, log_file_name, log_level=logging.DEBUG, logger_name=None):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] - [%(filename)s line:%(lineno)3d] : %(message)s"
        )
        if log_file_name is not None:
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setFormatter(formatter)
            self.__logger.addHandler(file_handler)        
        console_handler.setFormatter(formatter)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def data_augmentation(config, is_train=True, grayscale=True):
    aug = []
    if is_train:
        # random crop
        if config.augmentation.random_crop:
            aug.append(transforms.RandomCrop(config.input_size, padding=4))
        # horizontal filp
        if config.augmentation.random_horizontal_filp:
            aug.append(transforms.RandomHorizontalFlip())
        
    aug.append(transforms.ToTensor())
    if grayscale:
        aug.append(transforms.Grayscale(1))

    return aug


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + ".pth")
    if is_best:
        shutil.copyfile(filename + ".pth", filename + "_best.pth")


def load_checkpoint(path, model, optimizer=None, logger=None, retrain_head=False):
    if logger is not None:
        logger.info("=== loading checkpoint '{}' ===".format(path))

    checkpoint = torch.load(path)
    checkpoint_model = checkpoint["state_dict"]
    
    if retrain_head:        
        for k in ['head.weight', 'head.bias']:
            del checkpoint_model[k]
        if logger is not None:
            logger.info('[i] remove head parameters from the checkpoint to retrain the head...')

    model.load_state_dict(checkpoint_model, strict=False)

    if optimizer is not None:
        best_prec = checkpoint["best_prec"]
        last_epoch = checkpoint["last_epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        if logger is not None:
            logger.info(
                "=== done. also loaded optimizer from "
                + "checkpoint '{}' (epoch {}) ===".format(path, last_epoch + 1)
            )
        return best_prec, last_epoch


def get_data_loader(transform_train, transform_test, config):
    if config.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=config.data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=transform_test)
    elif config.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=config.data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=config.data_path, train=False, download=True, transform=transform_test)
    elif config.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root=config.data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root=config.data_path, train=False, download=True, transform=transform_test)
    elif config.dataset == 'fashion_mnist':
        trainset = torchvision.datasets.FashionMNIST(root=config.data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.FashionMNIST(root=config.data_path, train=False, download=True, transform=transform_test)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch, shuffle=False, num_workers=config.workers)
    return train_loader, test_loader


def adjust_learning_rate(optimizer, epoch, config):
    def get_current_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]
    lr = get_current_lr(optimizer)
    min_lr = min_lr
    base_lr = config.lr_scheduler.base_lr
    lr_scheduler = config.lr_scheduler
    if lr_scheduler.type == "STEP":
        if epoch in lr_scheduler.lr_epochs:
            lr *= lr_scheduler.lr_mults
    elif lr_scheduler.type == "COSINE":
        ratio = epoch / config.epochs
        lr = min_lr + (base_lr - min_lr) * (1.0 + math.cos(math.pi * ratio)) / 2.0        
    elif lr_scheduler.type == "COSINE-V2":
        ratio = min(epoch / (config.epochs - 10), 1)
        lr = min_lr + (base_lr - min_lr) * (1.0 + math.cos(math.pi * ratio)) / 2.0
    elif lr_scheduler.type == "HTD":
        ratio = epoch / config.epochs
        lr = min_lr + (base_lr - min_lr) * (1.0 - math.tanh(lr_scheduler.lower_bound + (lr_scheduler.upper_bound - lr_scheduler.lower_bound) * ratio)) / 2.0        
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def get_config(path):    
    with open(path) as f:
        config = yaml.safe_load(f)
        config = EasyDict(config)
    return config


def get_model_from_workpath(work_path, load_checkpoint=False, is_imagenet=False, ckpt_path=None):
    if not is_imagenet: # cifar10 model
        config = get_config(join(work_path, 'config.yaml'))
        model = get_model(config)
        ckpt_path = join(work_path, config.ckpt_name+".pth") if ckpt_path is None else ckpt_path
    else: # imagenet model
        config_path = join(work_path, 'config.yaml')    
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        model_params = EasyDict(cfg['model_params'])
        model = models.__dict__[cfg['model']](cfg['num_classes'], **model_params)
    if load_checkpoint:
        checkpoint = torch.load(ckpt_path)        
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def gen_model_param_config(model_type):
    if model_type == 'SKSI':
        model_params = {
            'opto_layer_info': {
                'type': 'conv',
                'kernel_size': 3,
            },
            'first_norm': False,
            'last_norm': False,
            'pooling_type': 'avg',
            'elec_layer_type': 'dws-a1-infer',
            'channels': [25, 50],
            'elec_layers': [0, 0],
        }    
    elif model_type == 'LKSI':
        model_params = {
            'opto_layer_info': {
                'type': 'conv',
                'kernel_size': 15,
            },
            'first_norm': False,
            'last_norm': False,
            'pooling_type': 'avg',
            'elec_layer_type': 'dws-a1-infer',
            'channels': [25, 50],
            'elec_layers': [0, 0],
        }
    elif model_type == 'SKSV':
        model_params = {
            'opto_layer_info': {
                'type': 'lrsvconv',
                'basis_conv_type': 'conv',
                'kernel_size': 3,
                'kernel_rank': 6,       
            },
            'first_norm': False,
            'last_norm': False,
            'pooling_type': 'avg',
            'elec_layer_type': 'dws-a1-infer',
            'channels': [25, 50],
            'elec_layers': [0, 0],
        }            
    elif model_type == 'LKSV':
        model_params = {
            'opto_layer_info': {
                'type': 'lrsvconv',
                'basis_conv_type': 'conv',                
                'kernel_size': 15,
                'kernel_rank': 6,                       
            },
            'first_norm': False,
            'last_norm': False,
            'pooling_type': 'avg',
            'elec_layer_type': 'dws-a1-infer',
            'channels': [25, 50],
            'elec_layers': [0, 0],
        }
    else:
        raise NotImplementedError
    return model_params


def get_class_dict(dataset, classes=None):
    if classes is None:
        classes = dataset.classes
    class_dict = {i: [] for i in range(10)}
    for i, (img, label) in enumerate(dataset):
        img = (np.array(img) * 255).astype(np.uint8)
        if img.ndim == 3:
            img = img.transpose((1, 2, 0))
        class_dict[label].append((i, img))
    return class_dict


def dataset_vis_prob(dataset, predicts_prob=None, classes=None, savepath=None):
    if classes is None:
        classes = dataset.classes
        classes[1] = 'car'
    class_dict = get_class_dict(dataset, classes)    
    fig, axes = plt.subplots(10, 10, figsize=(14, 16))

    for i in range(10):
        ks = np.random.permutation(1000)
        for j in range(10):
            # k = np.random.randint(0, 100)
            k = ks[j]
            idx = class_dict[i][k][0]
            img = class_dict[i][k][1]
            axes[i, j].imshow(img, cmap='gray')
            # axes[i, j].imshow(class_dict[i][k])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            
            if predicts_prob is not None:
                predict_prob = predicts_prob[idx]
                top_prob_indices = np.argsort(predict_prob)[::-1]
                                
                fontsize = 10
                topk = 2
                labels = list(map(lambda ind: classes[ind].capitalize(), top_prob_indices[:topk]))
                probs = predict_prob[top_prob_indices[:topk]]
                label_list = [r'\textcolor{Orange}' + '{' + label + '}' + r' \textcolor{Orange}' + '{' + f'{prob:.3f}' + '}' if ind != i 
                              else r'\textcolor{ForestGreen}' + '{' + label + '}' + r' \textcolor{ForestGreen}' + '{' + f'{prob:.3f}' + '}'
                              for label, prob, ind in zip(labels, probs, top_prob_indices[:topk])]                
                
                xlabel = '\n'.join(label_list)

                axes[i, j].set_xlabel(xlabel, fontsize=fontsize, weight='bold')
                
    fig.subplots_adjust(hspace=0.2, wspace=0.4)

    for i in range(10):
        axes[i, 0].set_ylabel(r'\textbf{' + classes[i].capitalize() + '}', rotation=0, weight='bold', fontsize=14, labelpad=60)
        axes[i, 0].yaxis.set_label_position('left')
        
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.clf()
    else:
        plt.show()


def get_dataset_labels(dataset):
    labels = []
    for _, label in dataset:
        labels.append(label)
    return np.array(labels)


def visualize_cm(labels, predicts, classes):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm_exp = confusion_matrix(labels, predicts, normalize='true')
    acc_exp = np.equal(predicts, labels).sum() / len(labels)

    cmap = 'Blues'
    cm = confusion_matrix(labels, predicts, normalize='true')
    cm_display = ConfusionMatrixDisplay(cm, display_labels=classes)
    cm_display.plot(cmap=cmap, text_kw={'fontsize': 16}, values_format='.2f')
    cm_display.figure_.set_size_inches(10, 8)
    ax = cm_display.figure_.gca()
    ax.images[-1].colorbar.ax.tick_params(axis='both', labelsize=18)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='both', labelsize=18)
    plt.xticks(rotation = 60)
    plt.yticks(rotation = 25)

    return ax


def generate_chip_pattern(pattern_type):
    chip_pattern = {}
    for kernel_id in range(25):
        if kernel_id in [0, 1]:
            x = (kernel_id * 2) - 2 + 1
            y = - 4
        elif kernel_id in [23, 24]:
            x = ((kernel_id - 23) * 2) - 2 + 1
            y = 4
        else:
            x = ((kernel_id - 2) % 3) * 2 - 2
            y = ((kernel_id - 2) // 3) + 1 - 4
        chip_pattern[f'pos_{kernel_id}'] = (-x, -y)
        chip_pattern[f'neg_{kernel_id}'] = (-(x+1), -y)
    return chip_pattern


def ftr2arr(images_pos, images_neg, img_size=128, chip_pattern_type='6x9', draw_figure=True, show=True, clip=False, suptitle=None, **kwargs):    
    chip_pattern = generate_chip_pattern(chip_pattern_type)
    if draw_figure:
        fig, axes = plt.subplots(6, 9, figsize=(10, 5))
        if clip:
            vmin, vmax = np.percentile(np.concatenate([images_pos, images_neg], axis=0), [0.5, 99.5])
        else:
            vmin, vmax = 0, 1
    
    image_arr = np.zeros((6, 9, img_size, img_size), dtype=np.float32)
    for i in range(25):
        cord_p = chip_pattern[f'pos_{i}']
        cord_n = chip_pattern[f'neg_{i}']
        image_arr[-cord_p[0]+2, -cord_p[1]+4] = images_pos[i]
        image_arr[-cord_n[0]-1+3, -cord_n[1]+4] = images_neg[i]
        if draw_figure:
            axes[-cord_p[0]+2, -cord_p[1]+4].imshow(images_pos[i], vmin=vmin, vmax=vmax, **kwargs)
            axes[-cord_n[0]-1+3, -cord_n[1]+4].imshow(images_neg[i], vmin=vmin, vmax=vmax, **kwargs)
    
    if draw_figure:
        for i in range(6):
            for j in range(9):
                axes[i, j].axis('off')
        
        ax = axes[4, 8]

        cax = fig.add_axes([ax.get_position().x1+0.03, ax.get_position().y0+ax.get_position().height*0.5, 0.01, ax.get_position().height * 3.5])        
        cbar = plt.colorbar(axes[0, 1].get_images()[0], aspect=50, cax=cax)        
        
        cbar.ax.tick_params(labelsize=10, width=0.1)  # Adjust font size and colorbar width
        cbar.outline.set_linewidth(0.2)  # Set colorbar outline width                                
    
    if draw_figure and suptitle: plt.suptitle(suptitle, fontsize=16)
    if draw_figure and show: plt.show()
    return image_arr


def visualize_cls_prob(predict_prob, class_names):
    num_classes = len(class_names)
    x_pos = np.arange(num_classes)
    
    # Creating a horizontal bar plot for each class
    plt.figure(figsize=(8, 6))
    plt.bar(x_pos, predict_prob, color='skyblue')
    plt.ylim([0, 1])
    
    # Adding class labels
    plt.xticks(x_pos, class_names)
    plt.ylabel('Probability')
    plt.xlabel('Classes')

    plt.tight_layout()
    plt.show()
