"""
pytorch main to use pretrained models

Created on 04/13/2020

@author: RH
"""

import sys
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
import setsep
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


train_transformer = transforms.Compose([
    transforms.ColorJitter(brightness=0.35, contrast=0.5, saturation=0.5, hue=0.35),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class DataSet(Dataset):
    def __init__(self, datadir, transform=None):
        self.data_dir = datadir
        self.transform = transform
        self.imglist = pd.read_csv(self.data_dir, header=0).values.tolist()

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imglist[idx][1]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.imglist[idx][2]),
                  'slide': str(self.imglist[idx][0]),
                  'path': str(self.imglist[idx][1])}
        return sample


def modeldict(mmd):
    if mmd =='alexnet_pt': out_model = models.alexnet(pretrained=True)
    elif mmd == 'alexnet': out_model = models.alexnet(pretrained=False)
    elif mmd == 'vgg11_pt': out_model = models.vgg11(pretrained=True)
    elif mmd == 'vgg11': out_model = models.vgg11(pretrained=False)
    elif mmd == 'vgg11bn_pt': out_model = models.vgg11_bn(pretrained=True)
    elif mmd == 'vgg11bn': out_model = models.vgg11_bn(pretrained=False)
    elif mmd == 'vgg13_pt': out_model = models.vgg13(pretrained=True)
    elif mmd == 'vgg13': out_model = models.vgg13(pretrained=False)
    elif mmd == 'vgg13bn_pt': out_model = models.vgg13_bn(pretrained=True)
    elif mmd == 'vgg13bn': out_model = models.vgg13_bn(pretrained=False)
    elif mmd == 'vgg16_pt': out_model = models.vgg16(pretrained=True)
    elif mmd == 'vgg16': out_model = models.vgg16(pretrained=False)
    elif mmd == 'vgg16bn_pt': out_model = models.vgg16_bn(pretrained=True)
    elif mmd == 'vgg16bn': out_model = models.vgg16_bn(pretrained=False)
    elif mmd == 'vgg19_pt': out_model = models.vgg19(pretrained=True)
    elif mmd == 'vgg19': out_model = models.vgg19(pretrained=False)
    elif mmd == 'vgg19bn_pt': out_model = models.vgg19_bn(pretrained=True)
    elif mmd == 'vgg19bn': out_model = models.vgg19_bn(pretrained=False)
    elif mmd == 'resnet18': out_model = models.resnet18(pretrained=False)
    elif mmd == 'resnet18_pt': out_model = models.resnet18(pretrained=True)
    elif mmd == 'resnet34': out_model = models.resnet34(pretrained=False)
    elif mmd == 'resnet34_pt': out_model = models.resnet34(pretrained=True)
    elif mmd == 'resnet50': out_model = models.resnet50(pretrained=False)
    elif mmd == 'resnet50_pt': out_model = models.resnet50(pretrained=True)
    elif mmd == 'resnet101': out_model = models.resnet101(pretrained=False)
    elif mmd == 'resnet101_pt': out_model = models.resnet101(pretrained=True)
    elif mmd == 'resnet152': out_model = models.resnet152(pretrained=False)
    elif mmd == 'resnet152_pt': out_model = models.resnet152(pretrained=True)
    elif mmd == 'squeezenet10_pt': out_model = models.squeezenet1_0(pretrained=True)
    elif mmd == 'squeezenet10': out_model = models.squeezenet1_0(pretrained=False)
    elif mmd == 'squeezenet11_pt': out_model = models.squeezenet1_1(pretrained=True)
    elif mmd == 'squeezenet11': out_model = models.squeezenet1_1(pretrained=False)
    elif mmd == 'densenet121_pt': out_model = models.densenet121(pretrained=True)
    elif mmd == 'densenet121': out_model = models.densenet121(pretrained=False)
    elif mmd == 'densenet161_pt': out_model = models.densenet161(pretrained=True)
    elif mmd == 'densenet161': out_model = models.densenet161(pretrained=False)
    elif mmd == 'densenet169_pt': out_model = models.densenet169(pretrained=True)
    elif mmd == 'densenet169': out_model = models.densenet169(pretrained=False)
    elif mmd == 'densenet201_pt': out_model = models.densenet201(pretrained=True)
    elif mmd == 'densenet201': out_model = models.densenet201(pretrained=False)
    elif mmd == 'inception_pt': out_model = models.inception_v3(pretrained=True, aux_logits=False)
    elif mmd == 'inception': out_model = models.inception_v3(pretrained=False, aux_logits=False)
    elif mmd == 'googlenet_pt': out_model = models.googlenet(pretrained=True, aux_logits=False)
    elif mmd == 'googlenet': out_model = models.googlenet(pretrained=False, aux_logits=False)
    elif mmd == 'shufflenet05_pt': out_model = models.shufflenet_v2_x0_5(pretrained=True)
    elif mmd == 'shufflenet05': out_model = models.shufflenet_v2_x0_5(pretrained=False)
    elif mmd == 'shufflenet10_pt': out_model = models.shufflenet_v2_x1_0(pretrained=True)
    elif mmd == 'shufflenet10': out_model = models.shufflenet_v2_x1_0(pretrained=False)
    elif mmd == 'shufflenet20': out_model = models.shufflenet_v2_x2_0(pretrained=False)
    elif mmd == 'mobilenet_pt': out_model = models.mobilenet_v2(pretrained=True)
    elif mmd == 'mobilenet': out_model = models.mobilenet_v2(pretrained=False)
    elif mmd == 'resnext50_32x4d_pt': out_model = models.resnext50_32x4d(pretrained=True)
    elif mmd == 'resnext50_32x4d': out_model = models.resnext50_32x4d(pretrained=False)
    elif mmd == 'resnext101_32x8d_pt': out_model = models.resnext101_32x8d(pretrained=True)
    elif mmd == 'resnext101_32x8d': out_model = models.resnext101_32x8d(pretrained=False)
    elif mmd == 'wide_resnet50_2_pt': out_model = models.wide_resnet50_2(pretrained=True)
    elif mmd == 'wide_resnet50_2': out_model = models.wide_resnet50_2(pretrained=False)
    elif mmd == 'wide_resnet101_2_pt': out_model = models.wide_resnet101_2(pretrained=True)
    elif mmd == 'wide_resnet101_2': out_model = models.wide_resnet101_2(pretrained=False)
    elif mmd == 'mnasnet05_pt': out_model = models.mnasnet0_5(pretrained=True)
    elif mmd == 'mnasnet05': out_model = models.mnasnet0_5(pretrained=False)
    elif mmd == 'mnasnet075': out_model = models.mnasnet0_75(pretrained=False)
    elif mmd == 'mnasnet10_pt': out_model = models.mnasnet1_0(pretrained=True)
    elif mmd == 'mnasnet10': out_model = models.mnasnet1_0(pretrained=False)
    elif mmd == 'mnasnet13': out_model = models.mnasnet1_3(pretrained=False)
    else:
        out_model = None
        print('Invalid model name. Terminated.')
        exit(0)
    return out_model


if __name__ == '__main__':
    dirr = sys.argv[1]  # output directory
    bs = sys.argv[2]  # batch size
    bs = int(bs)
    md = sys.argv[3]  # model to use

    try:
        ep = sys.argv[4]  # epochs to train
        ep = int(ep)
    except IndexError:
        ep = 3000

    # paths to directories
    METAGRAPH_DIR = "../Results/{}".format(dirr)
    data_dir = "../Results/{}/data".format(dirr)
    out_dir = "../Results/{}/out".format(dirr)

    # make directories if not exist
    for DIR in (METAGRAPH_DIR, data_dir, out_dir):
        try:
            os.mkdir(DIR)
        except FileExistsError:
            pass

    try:
        trs = DataSet(str(data_dir + '/tr_sample.csv'), transform=train_transformer)
        tes = DataSet(str(data_dir + '/te_sample.csv'), transform=val_transformer)
        vas = DataSet(str(data_dir + '/va_sample.csv'), transform=val_transformer)
    except FileNotFoundError:
        setsep.set_sep(path=data_dir)
        trs = DataSet(str(data_dir + '/tr_sample.csv'), transform=train_transformer)
        tes = DataSet(str(data_dir + '/te_sample.csv'), transform=val_transformer)
        vas = DataSet(str(data_dir + '/va_sample.csv'), transform=val_transformer)

    train_loader = DataLoader(trs, batch_size=bs, drop_last=False, shuffle=True)
    val_loader = DataLoader(vas, batch_size=bs, drop_last=False, shuffle=False)
    test_loader = DataLoader(tes, batch_size=bs, drop_last=False, shuffle=False)

    model = modeldict(md)
    if 'vgg' in md or 'alex' in md:
        number_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]  # Remove last layer
        features.extend([torch.nn.Linear(number_features, 2)])
        model.classifier = torch.nn.Sequential(*features)
    elif 'squeeze' in md:
        model.num_classes = 2
    elif 'dense' in md:
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif 'mobile' in md or 'mnas' in md:
        number_features = model.classifier[1].in_features
        features = list(model.classifier.children())[:-1]  # Remove last layer
        features.extend([torch.nn.Linear(number_features, 2)])
        model.classifier = torch.nn.Sequential(*features)
    else:
        model.fc = nn.Linear(model.fc.in_features, 2)

    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # train
    best_epoch = -1
    model.train()

    losslist = []

    summarylist = []
    if '_pt' in md:
        summarylist.append(md.split('_pt')[0])
        summarylist.append('pretrained')
    else:
        summarylist.append(md)
        summarylist.append('scratch')

    for epoch in range(ep):
        train_loss = 0
        train_correct = 0
        for batch_index, batch_samples in enumerate(train_loader):
            data, target = batch_samples['img'].to('cuda'), batch_samples['label'].to('cuda')
            optimizer.zero_grad()
            output = model(data)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, target.long())
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.long().view_as(pred)).sum().item()
        print('\nEpoch: {} \nTrain set: Average loss: {:.4f}, Tile Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
            100.0 * train_correct / len(train_loader.dataset)), flush=True)

        # validation
        val_loss = 0
        correct = 0
        model.eval()
        predlist = []
        scorelist = []
        targetlist = []
        slidelist = []
        pathlist = []
        with torch.no_grad():
            # Predict
            for batch_index, batch_samples in enumerate(val_loader):
                data, target, slide, impath = batch_samples['img'].to('cuda'), batch_samples['label'].to('cuda'), \
                                                batch_samples['slide'], batch_samples['path']
                output = model(data)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(output, target.long())
                val_loss += loss
                score = F.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.long().view_as(pred)).sum().item()
                targetcpu = target.long().cpu().numpy()
                predlist = np.append(predlist, pred.cpu().numpy())
                scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
                targetlist = np.append(targetlist, targetcpu)
                slidelist = np.append(slidelist, slide)
                pathlist = np.append(pathlist, impath)
            ave_val_loss = val_loss.cpu().numpy() / len(val_loader.dataset)
            losslist = np.append(losslist, ave_val_loss)

            if ave_val_loss == min(losslist):
                best_epoch = epoch
                print('Temporary best model found @ epoch {}! Saving...'.format(epoch))
                torch.save(model, '{}/model.pth'.format(METAGRAPH_DIR))

                best_joined = pd.DataFrame({
                    'prediction': predlist,
                    'target': targetlist,
                    'score': scorelist,
                    'slide': slidelist,
                    'path': pathlist
                })
                best_joined.to_csv('{}/best_validation_tile.csv'.format(out_dir), index=False)

            print('\nValidation set: Average loss: {:.4f}, Tile Accuracy: {}/{} ({:.0f}%)\n'.format(
                ave_val_loss, correct, len(val_loader.dataset),
                100.0 * correct / len(val_loader.dataset)), flush=True)

            TP = ((predlist == 1) & (targetlist == 1)).sum()
            TN = ((predlist == 0) & (targetlist == 0)).sum()
            FN = ((predlist == 0) & (targetlist == 1)).sum()
            FP = ((predlist == 1) & (targetlist == 0)).sum()
            print("\nPer tile metrics: ")
            print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
            print('TP+FP=', TP + FP)
            if (TP + FP) != 0:
                p = TP / (TP + FP)
                print('precision=', p)
                r = TP / (TP + FN)
                print('recall=', r)
                F1 = 2 * r * p / (r + p)
                print('F1=', F1)
            acc = (TP + TN) / (TP + TN + FP + FN)
            print('acc=', acc)
            AUC = roc_auc_score(targetlist, scorelist)
            print('AUC=', AUC)

            joined = pd.DataFrame({
                'prediction': predlist,
                'target': targetlist,
                'score': scorelist,
                'slide': slidelist
            })

            joined = joined.groupby(['slide']).mean()
            joined = joined.round({'prediction': 3, 'target': 3})
            if best_epoch == epoch:
                joined.to_csv('{}/best_validation_slide.csv'.format(out_dir), index=True)

            print("\nPer slide metrics: ")
            TP = joined.loc[(joined['prediction'] == 1) & (joined['target'] == 1)].shape[0]
            TN = joined.loc[(joined['prediction'] == 0) & (joined['target'] == 0)].shape[0]
            FN = joined.loc[(joined['prediction'] == 0) & (joined['target'] == 1)].shape[0]
            FP = joined.loc[(joined['prediction'] == 1) & (joined['target'] == 0)].shape[0]
            print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
            print('TP+FP=', TP + FP)
            if (TP+FP) != 0:
                p = TP / (TP + FP)
                print('precision=', p)
                r = TP / (TP + FN)
                print('recall=', r)
                F1 = 2 * r * p / (r + p)
                print('F1=', F1)
            acc = (TP + TN) / (TP + TN + FP + FN)
            print('acc=', acc)
            AUC = roc_auc_score(joined['target'].tolist(), joined['score'].tolist())
            print('AUC=', AUC)

            if epoch > 99 and ave_val_loss >= np.mean(losslist[-21:-1]):
                print("\nEarly stop criteria met @ epoch: ", epoch)
                break

    print('\nBest model @ epoch: ', best_epoch)
    summarylist.append(best_epoch)

    # test
    test_loss = 0
    correct = 0
    model = torch.load('{}/model.pth'.format(METAGRAPH_DIR))
    model.eval()
    predlist = []
    scorelist = []
    targetlist = []
    slidelist = []
    pathlist = []
    with torch.no_grad():
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target, slide, impath = batch_samples['img'].to('cuda'), batch_samples['label'].to('cuda'), \
                                            batch_samples['slide'], batch_samples['path']

            output = model(data)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, target.long())
            test_loss += loss
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)
            slidelist = np.append(slidelist, slide)
            pathlist = np.append(pathlist, impath)
        ave_test_loss = test_loss.cpu().numpy() / len(val_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Tile Accuracy: {}/{} ({:.0f}%)\n'.format(
            ave_test_loss, correct, len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset)), flush=True)

        summarylist.append(ave_test_loss)

        tile_joined = pd.DataFrame({
            'prediction': predlist,
            'target': targetlist,
            'score': scorelist,
            'slide': slidelist,
            'path': pathlist
        })
        tile_joined.to_csv('{}/test_tile.csv'.format(out_dir), index=False)

        print("\nPer tile metrics: ")
        TP = ((predlist == 1) & (targetlist == 1)).sum()
        TN = ((predlist == 0) & (targetlist == 0)).sum()
        FN = ((predlist == 0) & (targetlist == 1)).sum()
        FP = ((predlist == 1) & (targetlist == 0)).sum()
        print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
        print('TP+FP', TP + FP)
        if (TP + FP) != 0:
            p = TP / (TP + FP)
            print('precision', p)
            r = TP / (TP + FN)
            print('recall', r)
            F1 = 2 * r * p / (r + p)
            print('F1', F1)
        else:
            p = np.nan
            r = np.nan
            F1 = np.nan
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('acc', acc)
        AUC = roc_auc_score(targetlist, scorelist)
        print('AUC', AUC)

        fpr, tpr, _ = roc_curve(targetlist, scorelist)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.5f)' % AUC)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of COVID')
        plt.legend(loc="lower right")
        plt.savefig("../Results/{}/out/ROC_tile.png".format(dirr))

        average_precision = average_precision_score(targetlist, scorelist)
        print('Average precision-recall score: {0:0.5f}'.format(average_precision))
        plt.figure()
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        precision, recall, _ = precision_recall_curve(targetlist, scorelist)
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('COVID PRC: AP={:0.5f}; Accu={}'.format(average_precision, acc))
        plt.savefig("../Results/{}/out/PRC_tile.png".format(dirr))

        summarylist.extend([TP, TN, FN, FP, p, r, F1, acc, AUC, average_precision])


        joined = pd.DataFrame({
            'prediction': predlist,
            'target': targetlist,
            'score': scorelist,
            'slide': slidelist
        })

        joined = joined.groupby(['slide']).mean()
        joined = joined.round({'prediction': 3, 'target': 3})
        joined.to_csv('{}/test_slide.csv'.format(out_dir), index=True)

        print("\nPer slide metrics: ")
        TP = joined.loc[(joined['prediction'] == 1) & (joined['target'] == 1)].shape[0]
        TN = joined.loc[(joined['prediction'] == 0) & (joined['target'] == 0)].shape[0]
        FN = joined.loc[(joined['prediction'] == 0) & (joined['target'] == 1)].shape[0]
        FP = joined.loc[(joined['prediction'] == 1) & (joined['target'] == 0)].shape[0]
        print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
        print('TP+FP=', TP + FP)
        if (TP + FP) != 0:
            p = TP / (TP + FP)
            print('precision=', p)
            r = TP / (TP + FN)
            print('recall=', r)
            F1 = 2 * r * p / (r + p)
            print('F1=', F1)
        else:
            p = np.nan
            r = np.nan
            F1 = np.nan
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('acc=', acc)
        AUC = roc_auc_score(joined['target'].tolist(), joined['score'].tolist())
        print('AUC=', AUC)

        fpr, tpr, _ = roc_curve(joined['target'].tolist(), joined['score'].tolist())
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.5f)' % AUC)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of COVID')
        plt.legend(loc="lower right")
        plt.savefig("../Results/{}/out/ROC_slide.png".format(dirr))

        average_precision = average_precision_score(joined['target'].tolist(), joined['score'].tolist())
        print('Average precision-recall score: {0:0.5f}'.format(average_precision))
        plt.figure()
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        precision, recall, _ = precision_recall_curve(joined['target'].tolist(), joined['score'].tolist())
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('COVID PRC: AP={:0.5f}; Accu={}'.format(average_precision, acc))
        plt.savefig("../Results/{}/out/PRC_slide.png".format(dirr))

        summarylist.extend([TP, TN, FN, FP, p, r, F1, acc, AUC, average_precision])

    summarypd = pd.DataFrame([summarylist], columns=['model', 'state', 'best epoch', 'test loss', 'TP_tile',
                                                     'TN_tile', 'FN_tile', 'FP_tile', 'precision_tile',
                                                     'recall_tile', 'F1_tile', 'accuracy_tile', 'AUROC_tile',
                                                     'AUPRC_tile',	'TP_slide', 'TN_slide', 'FN_slide',
                                                     'FP_slide', 'precision_slide',	'recall_slide', 'F1_slide',
                                                     'accuracy_slide', 'AUROC_slide', 'AUPRC_slide'])

