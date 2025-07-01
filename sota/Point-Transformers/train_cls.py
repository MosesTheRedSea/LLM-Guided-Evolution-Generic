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
from src.cfg import constants


def test(model, loader, num_class=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc

@hydra.main(version_base="1.1", config_path='config', config_name='cls')
def main(args):
   
    omegaconf.OmegaConf.set_struct(args, False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = constants.DATA_PATH
    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=24)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=24)

    '''MODEL LOADING'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_class = 40
    args.input_dim = 6 if args.normal else 3

    model_file = args.model.file
    file_name = str(model_file)

    if "_" in file_name:
        model_file_path = hydra.utils.to_absolute_path(
            f'/home/hice1/madewolu9/scratch/madewolu9/LLM_PointNet/LLM-Guided-PointCloud-Class/sota/Point-Transformers/models/Menghao/llmge_models/{model_file}.py')
    else:
        model_file_path = hydra.utils.to_absolute_path(
            f'/home/hice1/madewolu9/scratch/madewolu9/LLM_PointNet/LLM-Guided-PointCloud-Class/sota/Point-Transformers/models/Menghao/{model_file}.py')

    classifier = getattr(importlib.import_module(f'models.{args.model.name}.model'), 'PointTransformerCls')(args).to(device)
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

    # === Early stopping settings ===
    end_lr = getattr(args, 'end_lr', 1e-5)
    patience = getattr(args, 'patience', 15)
    stale_epochs = 0

    '''TRAINING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        classifier.train()
        mean_correct = []

        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
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

            if instance_acc > best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1

            if class_acc > best_class_acc:
                best_class_acc = class_acc

            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if instance_acc >= best_instance_acc:
                logger.info('Save model...')
                savepath = 'best_model.pth'
                logger.info('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

        # === Early stopping condition ===
        current_lr = optimizer.param_groups[0]['lr']
        if stale_epochs >= patience:
            logger.info(f"Early stopping at epoch {epoch} due to no improvement for {patience} epochs.")
            break
        if current_lr < end_lr:
            logger.info(f"Early stopping at epoch {epoch} because learning rate {current_lr:.2e} < {end_lr:.2e}")
            break

        global_epoch += 1

    gene_id = model_file.split("_")

    second_part = gene_id[1].split(".")[0] if len(gene_id) > 1 else "unknown"

    filename = f'{constants.SOTA_ROOT}/results/{second_part}_results.txt'

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    total_params = sum(p.numel() for p in classifier.parameters())

    results_text = f"{best_instance_acc},{total_params},{best_class_acc},{best_epoch}"

    with open(filename, 'w') as file:
        file.write(results_text)

    logger.info('End of training...')

if __name__ == '__main__':
    main()
