# -*-coding:utf-8-*-
import argparse
import logging
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import yaml
from os.path import join
from models import get_model
from utils import Logger, data_augmentation, EasyDict, FeatureSet


parser = argparse.ArgumentParser(description="CIFAR-10 Test")
parser.add_argument("--work-path", required=True, type=str)
parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
parser.add_argument("--resume_path", '-rp', type=str, default=None, help='checkpoint to use.')
parser.add_argument("--no-log", action="store_true", help="disable log function")
parser.add_argument("--real", action="store_true", help="test real measurements")
parser.add_argument("--save_result", '-sr', action="store_true", help="save results into .npy")

args = parser.parse_args()

log_file_name = join(args.work_path, 'log_test.txt') if not args.no_log else None
logger = Logger(
    log_file_name=log_file_name,
    log_level=logging.DEBUG,
    logger_name="CLSTest",
).get_log()

config = None


@torch.no_grad()
def eval(test_loader, net, device):
    net.eval()
    correct = 0
    total = 0
    logger.info(" === Model Evaluation on CIFAR-10 testset ===")
    
    predicts = []
    for _, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.squeeze()
        outputs = net(inputs)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if args.save_result:
            probs = torch.softmax(outputs, dim=1)
            predicts.append(probs.detach().cpu().numpy())
            
    if args.save_result:
        predicts = np.concatenate(predicts, axis=0)
        np.save(join(args.work_path, 'predicts.npy'), predicts)

    logger.info(f"   == test acc: {correct / total:6.3%}")


@torch.no_grad()
def eval_real(test_loader, net, device):
    net.eval()

    correct = 0
    total = 0

    logger.info(" === Validate ===")
    
    predicts = []
    for _, ((ftrs_p, ftrs_n), targets) in enumerate(test_loader):
        ftrs_p, ftrs_n, targets = ftrs_p.to(device), ftrs_n.to(device), targets.to(device)
        outputs = net((ftrs_p, ftrs_n))
        targets = targets.squeeze()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if args.save_result:
            probs = torch.softmax(outputs, dim=1)
            predicts.append(probs.detach().cpu().numpy())
            
    if args.save_result:
        predicts = np.concatenate(predicts, axis=0)
        np.save(join(args.work_path, 'predicts.npy'), predicts)

    logger.info(f"   == test acc: {correct / total:6.3%}")


def main():
    global args, config
    # read config from yaml file
    with open(args.work_path + "/config.yaml") as f:
        config = yaml.safe_load(f)
    # convert to dict
    config = EasyDict(config)
    
    # define network
    net = get_model(config)
    ckpt_file_name = join(args.work_path, config.ckpt_name+".pth")
    if args.resume_path is not None:
        ckpt_file_name = args.resume_path
    checkpoint = torch.load(ckpt_file_name, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint["state_dict"], strict=True)  # torch > 1.9.0
    
    # CPU or GPU
    device = "cuda" if config.use_gpu else "cpu"
    if device == "cuda":
        cudnn.benchmark = True

    net.to(device)
    transform_train = transforms.Compose(data_augmentation(config))
    transform_test = transforms.Compose(data_augmentation(config, is_train=False))
    
    config.workers = 1
    # if not args.real:
    testset = torchvision.datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch, shuffle=False, num_workers=config.workers)
    eval(test_loader, net, device)
    # else:
        # basedir = '../nanophotonics/designs/cifar10/20230601/SV/oenet_o3x7sv_t1/r6x4/z4.0'
        # testset = FeatureSet(torch.load(join(basedir, f'cifar10_ftrs_x1', 'testset.pt')), augment=False, hflip_prob=0, crop_pad=0)
        # test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch, shuffle=False, num_workers=4)        
        # eval_real(test_loader, net, device)
    

if __name__ == "__main__":
    main()
