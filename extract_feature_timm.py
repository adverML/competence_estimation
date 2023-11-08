#!/usr/bin/env python
import os
import argparse
import torch
import torch.nn as nn
from list_dataset import ImageFilelist
import numpy as np
import pickle
from tqdm import tqdm
# import mmcv
import torchvision as tv
from torch.cuda.amp import autocast

# import detectors
import timm
from torchvision import transforms

# from attack import attack_pgd_restart, ctx_noparamgrad
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import foolbox
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import  L2DeepFoolAttack, LinfBasicIterativeAttack, FGSM, L2CarliniWagnerAttack, LinfPGD, LinfDeepFoolAttack

import config as cf

DEEPFOOL = ['fgsm', 'l2df', 'linfdf', 'linfpgd']
AUTOATTACK = ['aa', 'apgd-ce']

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--data_root', default='/home/DATA/ITWM/lorenzp/cifar10', help='Path to data')
    # parser.add_argument('--out_file', default='/home/lorenzp/workspace/competence_estimation/features/cifar10/resnet18_test.npy', help='Path to output file')
    parser.add_argument('--out_file', default='/home/lorenzp/workspace/competence_estimation/features/cifar10/spatial/resnet18_train.npy', help='Path to output file')
    parser.add_argument('--model', default='resnet18', help='Path to config')
    parser.add_argument('--mode', default='spatial', type=str, choices=['benign', 'spatial','|phase|', 'phase', 'magnitude'], help='')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'svhn', 'imagenet', 'imagenet-a', 'imagenet-c', 'phase_cifar10', 'magnitude_cifar10'], help='dataset = [cifar10/cifar100]')
    # parser.add_argument('--checkpoint', default='checkpoints/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth', help='Path to checkpoint')
    parser.add_argument('--preprocess', default='', choices=['MFS', 'PFS'], help='apply FFT?')
    parser.add_argument('--checkpoint', default='', help='Path to checkpoint')
    parser.add_argument('--img_list', default=None, help='Path to image list')
    parser.add_argument('--batch', type=int, default=256, help='Path to data')
    parser.add_argument('--workers', type=int, default=4, help='Path to data')
    parser.add_argument('--attack', default=None, choices=[None, 'aa', 'pgd', 'fgsm', 'l2df', 'linfdf', 'linfpgd'], help='')
    parser.add_argument('--ε', type=float, default=8./255)
    parser.add_argument('--fc_save_path', default=None, help='Path to save fc')
    # parser.add_argument('--fc_save_path', default="/home/lorenzp/workspace/competence_estimation/features/cifar10/", help='Path to save fc')

    # AutoAttack
    parser.add_argument("--norm",  default="Linf", choices=['Linf', 'L2', 'L1'], help="")
    parser.add_argument("--version",  default="standard", help="")
    parser.add_argument("--verbose",  default=False, type=str2bool, help="")

    return parser.parse_args()


def normalize(images, mean, std):
    images[:,0,:,:] = (images[:,0,:,:] - mean[0]) / std[0]
    images[:,1,:,:] = (images[:,1,:,:] - mean[1]) / std[1]
    images[:,2,:,:] = (images[:,2,:,:] - mean[2]) / std[2]    
    return images


def create_dir(path):
    is_existing = os.path.exists(path)
    if not is_existing:
        os.makedirs(path)
        print("The new directory is created!", path)

def load_model_timm(args):
    if args.dataset == 'cifar10':
        # https://huggingface.co/edadaltocg/resnet18_cifar10
        model = timm.create_model("resnet18", num_classes=10, exportable=True, pretrained=False)
        # override model
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()  # type: ignore
        # model.fc = nn.Linear(512,  10)

        if args.mode in ['benign', 'spatial']:
            model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
                map_location="cpu", 
                file_name="resnet18_cifar10.pth",
                )
            )
        elif args.mode == '|phase|':
            checkpt = torch.load('/home/lorenzp/wide-resnet.pytorch/checkpoint_wrn/cifar10/resnet18_timm_|phase|_2023-10-17_08:36:00.pt')
            model.load_state_dict(checkpt['model_state_dict'])
        elif args.mode == 'phase':
            checkpt = torch.load('/home/lorenzp/wide-resnet.pytorch/checkpoint_wrn/cifar10/resnet18_timm_phase_2023-10-17_08:36:12.pt')
            model.load_state_dict(checkpt['model_state_dict'])
        elif args.mode == 'magnitude':
            checkpt = torch.load('/home/lorenzp/wide-resnet.pytorch/checkpoint_wrn/cifar10/resnet18_timm_magnitude_2023-10-16_20:05:23.pt')
            model.load_state_dict(checkpt['model_state_dict'])

    elif args.dataset == 'svhn':
        model = timm.create_model("hf_hub:edadaltocg/resnet18_svhn", num_classes=10, pretrained=False)
        # override model
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()  # type: ignore
        # model.fc = nn.Linear(512, 10)

        if args.mode in ['benign', 'spatial']:
            model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                "https://huggingface.co/edadaltocg/resnet18_svhn/resolve/main/pytorch_model.bin",
                map_location="cpu", 
                file_name="resnet18_cifar10.pth",
                )
            )

    elif args.dataset in ['imagenet', 'imagenet-a', 'imagenet-c']:
        model = timm.create_model("resnet18", pretrained=True)

    model.eval()
    model.cuda()
    # cudnn.benchmark = True
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])

    return model 


def main():
    args = parse_args()
    print(args)

    mean = cf.mean[args.dataset]
    std = cf.std[args.dataset]

    if args.mode in [ '|phase|', 'phase', 'magnitude' ]:
        print("dataset: ", args.mode + '_' + args.dataset)
        mean = cf.mean[args.mode + '_' + args.dataset]
        std  = cf.std[ args.mode + '_' + args.dataset]

    normalization = transforms.Normalize(mean, std)
    print("Load mean and std", mean, std)
    
    torch.backends.cudnn.benchmark = True

    if args.fc_save_path is not None:
        if not os.path.exists(args.fc_save_path):
            os.makedirs(args.fc_save_path)
        
        model =  load_model_timm(args)
       
        create_dir(os.path.dirname(args.fc_save_path))
        if args.model in ['repvgg_b3']:
            w = model.head.fc.weight.cpu().detach().numpy()
            b = model.head.fc.bias.cpu().detach().numpy()
        elif args.model in ['swin_base_patch4_window7_224', 'deit_base_patch16_224']:
            w = model.head.weight.cpu().detach().numpy()
            b = model.head.bias.cpu().detach().numpy()
        else:
            w = model.fc.weight.cpu().detach().numpy()
            b = model.fc.bias.cpu().detach().numpy()
        
        W_path = os.path.join(args.fc_save_path, args.model + '_W.npy')
        torch.save(w, W_path.replace('npy', 'pt'))
        
        b_path = os.path.join(args.fc_save_path, args.model + '_b.npy')
        torch.save(b, b_path.replace('npy', 'pt'))
          
#         W_path = os.path.join(args.fc_save_path, args.model + '_W.npy')
#         with open(W_path, 'wb') as f:
#             np.save(f, w)
        
#         b_path = os.path.join(args.fc_save_path, args.model + '_b.npy')
#         with open(b_path, 'wb') as f:
#             np.save(f, b)
#         print("Save W: ", W_path)
#         print("Save b: ", b_path)
        return

    model =  load_model_timm(args)
    train_nodes, eval_nodes = get_graph_node_names(model)

    if not args.attack == None and not args.attack == 'pgd':
        preprocessing = dict(mean=mean, std=std, axis=-3)
        fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    if args.attack == 'pgd':
        print("Not implemented")
        pass
        # def test_attacker(x, y):
        #     with ctx_noparamgrad(model):
        #         adv_delta = attack_pgd_restart(
        #             model=model,
        #             X=x,
        #             y=y,
        #             eps=args.ε,
        #             alpha=args.ε / 4,
        #             attack_iters=40,
        #             n_restarts=10,
        #             rs=True,
        #             verbose=True,
        #             linf_proj=True,
        #             l2_proj=False,
        #             l2_grad_update=False,
        #             cuda=torch.cuda.is_available()
        #         )
        #     return x + adv_delta
        
    elif args.attack == 'fgsm':
        attack = FGSM()
    elif args.attack == 'l2df':
        args.ε = None
        attack = L2DeepFoolAttack()
    elif args.attack == 'linfdf':
        args.ε = None
        attack = LinfDeepFoolAttack()
    elif args.attack == 'linfpgd':
        attack = LinfPGD()
    elif args.attack in AUTOATTACK:
        from submodules.autoattack.autoattack import AutoAttack as AutoAttack_mod
        adversary = AutoAttack_mod(fmodel, norm=args.norm.capitalize(), eps=args.ε, 
                                    log_path=None, # os.path.join(log_pth, args.load_json.split('/')[-1]).replace("json", "log"),  
                                    verbose=args.verbose, 
                                    version=args.version)

        if args.version == 'individual':
            adversary.attacks_to_run = [ args.attack ]

    train=True
    if 'test' in args.out_file:
        train=False

    if args.img_list is not None:
        dataset = ImageFilelist(args.data_root, args.img_list, transform)
    else:
        if args.dataset == 'cifar10':
            transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                # tv.transforms.Normalize(mean, std),
            ])
            dataset = tv.datasets.CIFAR10(root=args.data_root, train=train, download=True, transform=transform)
        elif  args.dataset == 'cifar100':
            transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
            ])
            dataset = tv.datasets.CIFAR100(root=args.data_root, train=train, download=True, transform=transform)
        elif  args.dataset == 'svhn':
            transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
            ])

            if train: 
                split = 'train'
            else:
                split = 'test'

            dataset = tv.datasets.SVHN(root=args.data_root, split=split, download=True, transform=transform)

        elif  args.dataset == 'imagenet':
            transform = tv.transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
            max_samples = 10000
            if train: 
                path = os.path.join(args.data_root, 'train')
            else:
                max_samples = 10000
                path = os.path.join(args.data_root, 'val')
            max_iterations = (max_samples + args.batch - 1) // args.batch
            dataset = tv.datasets.ImageFolder(path, transform)

        elif  args.dataset in ['imagenet-a', 'imagenet-c']:
            transform = tv.transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
            max_samples = 10000
            max_iterations = (max_samples + args.batch - 1) // args.batch
            dataset = tv.datasets.ImageFolder(args.data_root, transform)
        
        elif  args.dataset == 'dtd':
            # from dataset import DTD 
            # https://github.com/louity/patches/blob/master/dtd.py
            transform = tv.transforms.Compose([ 
                transforms.Resize(9*spatial_size//8),
                transforms.CenterCrop(spatial_size),
                transforms.ToTensor(),
                # normalize,
            ])

            if train: 
                split = 'train'
            else:
                split = 'test'

            dataset = tv.datasets.DTD(args.data_root, train, transform=transform)
            # dataset = DTD(root=args.data_root, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    feature_extractor = create_feature_extractor(model, return_nodes={'global_pool.flatten': 'features', 'fc': 'logits'})
    # feature_extractor = create_feature_extractor(model, return_nodes={'global_pool.pool': 'features', 'fc': 'logits'})

    features = []
    logits = []
    labels = []

    # with autocast():
    for epoch, (x, y) in tqdm(enumerate(dataloader)):
        x = x.cuda()
        y = y.cuda()
        
        if args.mode == '|phase|':
            print("err: 1")
            x = torch.abs(torch.angle(torch.fft.fft2(x, dim=(-2, -1))))
        elif args.mode == 'phase':
            print("err: 2")
            x = torch.angle(torch.fft.fft2(x, dim=(-2, -1)))
        elif args.mode == 'magnitude':
            print("err: 3")
            x =  torch.abs(torch.fft.fft2(x, dim=(-2, -1)))

        if not args.attack is None:
            print("err: 4")
            if args.attack == 'pgd':
                x = test_attacker(x,y)
            elif args.attack in DEEPFOOL: 
                raw_x, x, success = attack(fmodel, x, criterion=foolbox.criteria.Misclassification(y), epsilons=args.ε)
            elif args.attack in AUTOATTACK:
                with torch.no_grad():
                    if args.batch == 1:
                        x = torch.unsqueeze(x, 0)
                        y = torch.unsqueeze(y, 0)

                    if args.version == 'standard':
                        x, y_, max_nr, success = adversary.run_standard_evaluation(x, y, bs=args.batch, return_labels=True)
                    else: 
                        adv_complete = adversary.run_standard_evaluation_individual(x, y, bs=args.batch, return_labels=True)
                        x, y_, max_nr, success = adv_complete[ args.att ]
                    suc = success.float().mean().item() * 100
                    print(suc)
                    # nor = x[torch.where(success.int().cpu().squeeze() == 1)[0]].cpu()
                    # adv = x_[torch.where(success.int().cpu().squeeze() == 1)[0]].cpu()
        
        # x = normalize(x, mean, std)
        x = normalization(x)
        with torch.no_grad():
            feature = feature_extractor(x)
        features.append(feature['features'].cpu().detach().numpy())
        logits.append(feature['logits'].cpu().detach().numpy())

        if not args.attack is None:
            y_adv = model(x)
            y_adv = torch.argmax(y_adv, axis=1)

        labels.append(y.cpu().detach().numpy())

        if args.dataset == 'imagenet' and epoch >= max_iterations:
            break

    features = np.concatenate(features, axis=0)
    logits   = np.concatenate(logits, axis=0)
    labels   = np.concatenate(labels, axis=0)

    create_dir(os.path.dirname(args.out_file))
    dirname = os.path.dirname(args.out_file)
    basename = os.path.basename(args.out_file)
    preprocess = ''
    
    if not args.attack is None:
        basename =  args.attack + '_' + args.mode + '_' + basename
    else:
        basename = args.mode + '_' + basename

    out = os.path.join(dirname, "features_" +  basename)
    print("save as: ", out.replace("npy", "pt"))
    torch.save(features, out.replace("npy", "pt"))

    out = os.path.join(dirname, "logits_" + basename)
    torch.save(logits, out.replace("npy", "pt"))

    out = os.path.join(dirname, "labels_" + basename)
    torch.save(labels, out.replace("npy", "pt"))

#     out = os.path.join(dirname, "features_" +  basename)
#     with open(out, 'wb') as f:
#         np.save(out, features)
#     out = os.path.join(dirname, "logits_" + basename)
#     with open(out, 'wb') as f:
#         np.save(out, logits)
#     out = os.path.join(dirname, "labels_" + basename)
#     with open(out, 'wb') as f:
#         np.save(out, labels)
    
    
if __name__ == '__main__':
    main()
