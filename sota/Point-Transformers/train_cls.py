"""
Author: Benny
Date: Nov 2019
"""
from dataset import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import hydra
import omegaconf

def test(model, loader, num_class=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc

@hydra.main(config_path='config', config_name='cls', version_base='1.1')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    # print(args.pretty())
    '''DATA LOADING'''
    logger.info('Load dataset ...')

    DATA_PATH = hydra.utils.to_absolute_path('/home/hice1/madewolu9/scratch/madewolu9/LLM_PointNet/LLM-Guided-PointCloud-Class/sota/Point-Transformers/modelnet40_normal_resampled/')
    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal)

    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # Default Implementation
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.num_class = 40
    # args.input_dim = 6 if args.normal else 3

    # shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    # # classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args).cuda()
    # classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args).to(device)
    # criterion = torch.nn.CrossEntropyLoss()

    # Specific Model Implementation #1
    # args.num_class = 40
    # args.input_dim = 6 if args.normal else 3
    # model_folder = args.model.name
    # model_file = args.model.file
    # model_file_path = hydra.utils.to_absolute_path(f'models/{model_folder}/{model_file}')
    # # Error Handling if the file does not exist
    # if not os.path.exists(model_file_path):
    #     raise FileNotFoundError(f"Model file {model_file_path} not found!")
    # shutil.copy(model_file_path, '.')
    # model_module = importlib.import_module(f'models.{model_folder}.{model_file.replace(".py", "")}')
    # model_class_name = None
    # for name, obj in inspect.getmembers(model_module):
    #     if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
    #         model_class_name = name
    #         break
    # if model_class_name is None:
    #     raise ValueError(f"No subclass of torch.nn.Module found in {model_file}")
    # classifier = getattr(model_module, model_class_name)(args).cuda()
    # criterion = torch.nn.CrossEntropyLoss()

    # Specific Model Implementation #2
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.num_class = 40
    # args.input_dim = 6 if args.normal else 3
    # model_file = args.model.file
    # model_name = args.model.name 
    # # model_module_path = f'models.{model_name}.model' 
    # shutil.copy(hydra.utils.to_absolute_path('models/Menghao/{}.py}'.format(model_file)), '.')
    # # model_module = importlib.import_module(f'models.{model_folder}.{model_file.replace(".py", "")}')
    # # model_class = getattr(model_module, 'PointTransformerCls') 
    # # classifier = model_class(args).to(device)
    # classifier = getattr(importlib.import_module('models.Menghao.{}'.format(model_file)), 'PointTransformerCls')(args).to(device)
    # criterion = torch.nn.CrossEntropyLoss()

    # Specific Model Implementation #3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_class = 40
    args.input_dim = 6 if args.normal else 3

    model_file = args.model.file

    model_name = args.model.name 

    # Updated the model_file_path it was incorrect before
    model_file_path = hydra.utils.to_absolute_path('/home/hice1/madewolu9/scratch/madewolu9/LLM_PointNet/LLM-Guided-PointCloud-Class/sota/Point-Transformers/models/Menghao/{}.py'.format(model_file))

    # shutil.copy(hydra.utils.to_absolute_path('models/Menghao/{}.py'.format(model_file)), '.')

    try:
        shutil.copy(model_file_path, '.')
    except FileNotFoundError:
        logger.warning(f"Model file {model_file_path} does not exist. Skipping this model.")
        return 

    classifier = getattr(importlib.import_module('models.Menghao.{}'.format(model_file)), 'PointTransformerCls')(args).to(device)

    criterion = torch.nn.CrossEntropyLoss()
  
    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        classifier.train()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()

            pred = classifier(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
            
        scheduler.step()

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = 'best_model.pth'
                logger.info('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    gene_id = model_file.split("_")

    if len(gene_id) > 1:
        second_part = gene_id[1]
    else:
        second_part = None

    filename = f'/home/hice1/madewolu9/scratch/madewolu9/LLM_PointNet/LLM-Guided-PointCloud-Class/sota/Point-Transformers/results/{second_part}_results.txt'

    dir_path = os.path.dirname(filename)

    os.makedirs(dir_path, exist_ok=True)

    total_params = sum(p.numel() for p in classifier.parameters())

    results_text = f"{best_instance_acc},{total_params},{best_class_acc},{best_epoch}"

    with open(filename, 'w') as file:
        file.write(results_text)

    logger.info(f"Results have been written to {filename}")

    logger.info('End of training...')

if __name__ == '__main__':
    main()

# """
# Author: Benny
# Date: Nov 2019
# """
# from dataset import ModelNetDataLoader
# import argparse
# import numpy as np
# import os
# import torch
# import datetime
# import logging
# from pathlib import Path
# from tqdm import tqdm
# import sys
# import provider
# import importlib
# import shutil
# import hydra
# import omegaconf
# # This Method help turns turn the 
# def parse_args():
#     parser = argparse.ArgumentParser(description="Training Script for point cloud classification.")
#     parser.add_argument('--model_file', type=str, required=True, help="Model file name (e.g., model_38978342387423.py)")
#     # Add other arguments you need
#     return parser.parse_args()
# def test(model, loader, num_class=40):
#     mean_correct = []
#     class_acc = np.zeros((num_class,3))
#     for j, data in tqdm(enumerate(loader), total=len(loader)):
#         points, target = data
#         target = target[:, 0]
#         points, target = points.cuda(), target.cuda()
#         classifier = model.eval()
#         pred = classifier(points)
#         pred_choice = pred.data.max(1)[1]
#         for cat in np.unique(target.cpu()):
#             classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
#             class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
#             class_acc[cat,1]+=1
#         correct = pred_choice.eq(target.long().data).cpu().sum()
#         mean_correct.append(correct.item()/float(points.size()[0]))
#     class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
#     class_acc = np.mean(class_acc[:,2])
#     instance_acc = np.mean(mean_correct)
#     return instance_acc, class_acc

# """
# ░█─▄▀ ░█▀▀▀ ░█──░█ 　 ░█▀▀█ ░█▀▀▀█ ░█▀▀▄ ░█▀▀▀ 
# ░█▀▄─ ░█▀▀▀ ░█▄▄▄█ 　 ░█─── ░█──░█ ░█─░█ ░█▀▀▀ 
# ░█─░█ ░█▄▄▄ ──░█── 　 ░█▄▄█ ░█▄▄▄█ ░█▄▄▀ ░█▄▄▄
# """
# # Point Transformers works differently from the other files it uses .yaml files
# @hydra.main(config_path='config', config_name='cls')
# def main(args):
#     """
#     █▀▀ █──█ █▀▀█ █▀▀▄ █▀▀▀ █▀▀ █▀▀ 
#     █── █▀▀█ █▄▄█ █──█ █─▀█ █▀▀ ▀▀█ 
#     ▀▀▀ ▀──▀ ▀──▀ ▀──▀ ▀▀▀▀ ▀▀▀ ▀▀▀
#     """
#     cmd_args = parse_args()
#     model_file = cmd_args.model_file
#     model_dir = 'Menghao' 
#     logger = logging.getLogger(__name__)
#     logger.info(f"Using model: {model_file}")
#     omegaconf.OmegaConf.set_struct(args, False)
#     '''HYPER PARAMETER'''
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
#     logger = logging.getLogger(__name__)
#     print(args.pretty())
#     '''DATA LOADING'''
#     logger.info('Load dataset ...')
#     DATA_PATH = hydra.utils.to_absolute_path('modelnet40_normal_resampled/')
#     TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal)
#     TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
#     trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
#     testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
#     '''MODEL LOADING'''
#     args.num_class = 40
#     args.input_dim = 6 if args.normal else 3
#     model_path = hydra.utils.to_absolute_path(f'models/{model_dir}/{model_file}')
#     # shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')
#     # Check if the path exist in the model
#     if not os.path.exists(model_path):
#         logger.error(f"Model file {model_file} not found in {model_dir} directory.")
#         sys.exit(1)
#     model_module = importlib.import_module(f'models.{model_dir}.{model_file.replace(".py", "")}')
#     classifier = getattr(model_module, 'PointTransformerCls')(args).cuda()
#     # classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args).cuda()
#     criterion = torch.nn.CrossEntropyLoss()
#     try:
#         checkpoint = torch.load('best_model.pth')
#         start_epoch = checkpoint['epoch']
#         classifier.load_state_dict(checkpoint['model_state_dict'])
#         logger.info('Use pretrain model')
#     except:
#         logger.info('No existing model, starting training from scratch...')
#         start_epoch = 0
#     if args.optimizer == 'Adam':
#         optimizer = torch.optim.Adam(
#             classifier.parameters(),
#             lr=args.learning_rate,
#             betas=(0.9, 0.999),
#             eps=1e-08,
#             weight_decay=args.weight_decay
#         )
#     else:
#         optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
#     global_epoch = 0
#     global_step = 0
#     best_instance_acc = 0.0
#     best_class_acc = 0.0
#     best_epoch = 0
#     mean_correct = []
#     '''TRANING'''
#     logger.info('Start training...')
#     for epoch in range(start_epoch,args.epoch):
#         logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))1
#         classifier.train()
#         for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
#             points, target = data
#             points = points.data.numpy()
#             points = provider.random_point_dropout(points)
#             points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
#             points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
#             points = torch.Tensor(points)
#             target = target[:, 0]
#             points, target = points.cuda(), target.cuda()
#             optimizer.zero_grad()
#             pred = classifier(points)
#             loss = criterion(pred, target.long())
#             pred_choice = pred.data.max(1)[1]
#             correct = pred_choice.eq(target.long().data).cpu().sum()
#             mean_correct.append(correct.item() / float(points.size()[0]))
#             loss.backward()
#             optimizer.step()
#             global_step += 1
#         scheduler.step()
#         train_instance_acc = np.mean(mean_correct)
#         logger.info('Train Instance Accuracy: %f' % train_instance_acc)
#         with torch.no_grad():
#             instance_acc, class_acc = test(classifier.eval(), testDataLoader)
#             if (instance_acc >= best_instance_acc):
#                 best_instance_acc = instance_acc
#                 best_epoch = epoch + 1
#             if (class_acc >= best_class_acc):
#                 best_class_acc = class_acc
#             logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
#             logger.info('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))
#             if (instance_acc >= best_instance_acc):
#                 logger.info('Save model...')
#                 savepath = 'best_model.pth'
#                 logger.info('Saving at %s'% savepath)
#                 state = {
#                     'epoch': best_epoch,
#                     'instance_acc': instance_acc,
#                     'class_acc': class_acc,
#                     'model_state_dict': classifier.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                 }
#                 torch.save(state, savepath)
#             global_epoch += 1
#     logger.info('End of training...')

# if __name__ == '__main__':
#     main()
