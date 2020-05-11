import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import pandas as pd
import cv2


# Plot ROC and PRC plots
def ROC_PRC(outtl, pdx, path, name, dm, accur):
    tl = outtl.values[:, 0].ravel()
    y_score = np.asarray(pdx[:, 1]).ravel()
    auc = sklearn.metrics.roc_auc_score(tl, y_score)
    auc = round(auc, 5)
    print('{0} AUC = {1:0.5f}'.format(dm, auc))
    fpr, tpr, _ = sklearn.metrics.roc_curve(tl, y_score)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.5f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC'.format(name))
    plt.legend(loc="lower right")
    plt.savefig("../Results/{}/out/{}_{}_ROC.png".format(path, name, dm))

    average_precision = sklearn.metrics.average_precision_score(tl, y_score)
    print('Average precision-recall score: {0:0.5f}'.format(average_precision))
    plt.figure()
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    precision, recall, _ = sklearn.metrics.precision_recall_curve(tl, y_score)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('{} PRC: AP={:0.5f}; Accu={}'.format(name, average_precision, accur))
    plt.savefig("../Results/{}/out/{}_{}_PRC.png".format(path, name, dm))


# slide level; need prediction scores, true labels, output path, and name of the files for metrics;
# accuracy, AUROC; AUPRC.
def slide_metrics(inter_pd, path, name, fordict, pmd):
    inter_pd = inter_pd.drop(['path', 'tile_label', 'Prediction', 'True_label'], axis=1)
    inter_pd = inter_pd.groupby(['slide']).mean()
    inter_pd = inter_pd.round({'slide_label': 0})
    inter_pd['Prediction'] = inter_pd[['NEG_score', 'POS_score']].idxmax(axis=1)
    redict = {'NEG_score': int(0), 'POS_score': int(1)}
    inter_pd['Prediction'] = inter_pd['Prediction'].replace(redict)

    # accuracy calculations
    tott = inter_pd.shape[0]
    accout = inter_pd.loc[inter_pd['Prediction'] == inter_pd['slide_label']]
    accu = accout.shape[0]
    accurr = round(accu/tott, 5)
    print('Slide Total Accuracy: '+str(accurr))
    try:
        outtl_slide = inter_pd['slide_label'].to_frame(name='slide_lable')
        pdx_slide = inter_pd[['NEG_score', 'POS_score']].values
        ROC_PRC(outtl_slide, pdx_slide, path, name, 'slide', accurr)
    except ValueError:
        print('Not able to generate plots based on this set!')
    inter_pd['Prediction'] = inter_pd['Prediction'].replace(fordict)
    inter_pd['slide_label'] = inter_pd['slide_label'].replace(fordict)
    inter_pd.to_csv("../Results/{}/out/{}_slide.csv".format(path, name), index=True)


# for real image prediction, just output the prediction scores as csv
def realout(pdx, path, name, pmd):
    lbdict = {0: 'negative', 1: pmd}
    pdx = np.asmatrix(pdx)
    prl = pdx.argmax(axis=1).astype('uint8')
    prl = pd.DataFrame(prl, columns=['Prediction'])
    prl = prl.replace(lbdict)
    out = pd.DataFrame(pdx, columns=['NEG_score', 'POS_score'])
    out.reset_index(drop=True, inplace=True)
    prl.reset_index(drop=True, inplace=True)
    out = pd.concat([out, prl], axis=1)
    out.insert(loc=0, column='Num', value=out.index)
    out.to_csv("../Results/{}/out/{}.csv".format(path, name), index=False)


# tile level; need prediction scores, true labels, output path, and name of the files for metrics; accuracy, AUROC; PRC.
def metrics(pdx, tl, path, name, pmd, ori_test=None):
    # format clean up
    tl = np.asmatrix(tl)
    tl = tl.argmax(axis=1).astype('uint8')
    pdxt = np.asmatrix(pdx)
    prl = pdxt.argmax(axis=1).astype('uint8')
    prl = pd.DataFrame(prl, columns=['Prediction'])
    lbdict = {0: 'negative', 1: pmd}
    outt = pd.DataFrame(pdxt, columns=['NEG_score', 'POS_score'])
    outtlt = pd.DataFrame(tl, columns=['True_label'])
    if name == 'Validation' or name == 'Training':
        outtlt = outtlt.round(0)
    outt.reset_index(drop=True, inplace=True)
    prl.reset_index(drop=True, inplace=True)
    outtlt.reset_index(drop=True, inplace=True)
    out = pd.concat([outt, prl, outtlt], axis=1)
    if ori_test is not None:
        ori_test.reset_index(drop=True, inplace=True)
        out.reset_index(drop=True, inplace=True)
        out = pd.concat([ori_test, out], axis=1)
        slide_metrics(out, path, name, lbdict, pmd)

    stprl = prl.replace(lbdict)
    stouttl = outtlt.replace(lbdict)
    outt.reset_index(drop=True, inplace=True)
    stprl.reset_index(drop=True, inplace=True)
    stouttl.reset_index(drop=True, inplace=True)
    stout = pd.concat([outt, stprl, stouttl], axis=1)
    if ori_test is not None:
        ori_test.reset_index(drop=True, inplace=True)
        stout.reset_index(drop=True, inplace=True)
        stout = pd.concat([ori_test, stout], axis=1)
    stout.to_csv("../Results/{}/out/{}_tile.csv".format(path, name), index=False)

    # accuracy calculations
    tott = out.shape[0]
    accout = out.loc[out['Prediction'] == out['True_label']]
    accu = accout.shape[0]
    accurw = round(accu/tott, 5)
    print('Tile Total Accuracy: '+str(accurw))
    try:
        ROC_PRC(outtlt, pdxt, path, name, 'tile', accurw)
    except ValueError:
        print('Not able to generate plots based on this set!')


def py_returnCAMmap(activation, weights_LR):
    n_feat, w, h, n = activation.shape
    act_vec = np.reshape(activation, [n_feat, w*h])
    n_top = weights_LR.shape[0]
    out = np.zeros([w, h, n_top])

    for t in range(n_top):
        weights_vec = np.reshape(weights_LR[t], [1, weights_LR[t].shape[0]])
        heatmap_vec = np.dot(weights_vec,act_vec)
        heatmap = np.reshape(np.squeeze(heatmap_vec), [w, h])
        out[:,:,t] = heatmap

    return out


def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


def py_map2jpg(imgmap, rang, colorMap):
    if rang is None:
        rang = [np.min(imgmap), np.max(imgmap)]

    heatmap_x = np.round(imgmap*255).astype(np.uint8)

    return cv2.applyColorMap(heatmap_x, cv2.COLORMAP_JET)


def CAM(net, w, pred, x, y, path, name, bs, rd=0):
    DIRA = "../Results/{}/out/{}_NEG_img".format(path, name)
    DIRB = "../Results/{}/out/{}_POS_img".format(path, name)
    for DIR in (DIRA, DIRB):
        try:
            os.mkdir(DIR)
        except FileExistsError:
            pass
    catdict = {0: 'negative', 1: 'positive'}
    dirdict = {0: DIRA, 1: DIRB}
    rd = rd*bs

    y = np.asmatrix(y)
    y = y.argmax(axis=1).astype('uint8')
    pdx = np.asmatrix(pred)

    prl = (pdx[:,1] > 0.5).astype('uint8')

    for ij in range(len(y)):
        id = str(ij + rd)
        if prl[ij, 0] == y[ij]:
            ddt = 'Correct'
        else:
            ddt = 'Wrong'

        weights_LR = w
        activation_lastconv = np.array([net[ij]])
        weights_LR = weights_LR.T
        activation_lastconv = activation_lastconv.T

        topNum = 1  # generate heatmap for top X prediction results
        prdd = prl[ij, 0]
        curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[[prdd], :])
        DIRR = dirdict[prdd]
        catt = catdict[prdd]
        for kk in range(topNum):
            curCAMmap_crops = curCAMmapAll[:, :, kk]
            curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (299, 299))
            curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops), (299, 299))
            curHeatMap = im2double(curHeatMap)
            curHeatMap = py_map2jpg(curHeatMap)
            xim = x[ij].reshape(-1, 3)
            xim1 = xim[:, 0].reshape(-1, 299)
            xim2 = xim[:, 1].reshape(-1, 299)
            xim3 = xim[:, 2].reshape(-1, 299)
            image = np.empty([299, 299, 3])
            image[:, :, 0] = xim1
            image[:, :, 1] = xim2
            image[:, :, 2] = xim3
            a = im2double(image) * 255
            b = im2double(curHeatMap) * 255
            curHeatMap = a * 0.6 + b * 0.4
            ab = np.hstack((a, b))
            full = np.hstack((curHeatMap, ab))
            # imname = DIRR + '/' + id + ddt + '_' + catt + '.png'
            # imname1 = DIRR + '/' + id + ddt + '_' + catt + '_img.png'
            # imname2 = DIRR + '/' + id + ddt + '_' + catt + '_hm.png'
            imname3 = DIRR + '/' + id + ddt + '_' + catt + '_full.png'
            # cv2.imwrite(imname, curHeatMap)
            # cv2.imwrite(imname1, a)
            # cv2.imwrite(imname2, b)
            cv2.imwrite(imname3, full)


def CAM_R(net, w, pred, x, path, name, bs, rd=0):
    DIRR = "../Results/{}/out/{}_img".format(path, name)
    rd = rd * bs

    try:
        os.mkdir(DIRR)
    except(FileExistsError):
        pass

    pdx = np.asmatrix(pred)

    prl = pdx.argmax(axis=1).astype('uint8')

    for ij in range(len(prl)):
        id = str(ij + rd)
        weights_LR = w
        activation_lastconv = np.array([net[ij]])
        weights_LR = weights_LR.T
        activation_lastconv = activation_lastconv.T

        topNum = 1  # generate heatmap for top X prediction results
        curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[[1], :])
        for kk in range(topNum):
            curCAMmap_crops = curCAMmapAll[:, :, kk]
            curCAMmapLarge_crops = cv2.resize(curCAMmap_crops, (299, 299))
            curHeatMap = cv2.resize(im2double(curCAMmapLarge_crops), (299, 299))  # this line is not doing much
            curHeatMap = im2double(curHeatMap)
            curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
            xim = x[ij].reshape(-1, 3)
            xim1 = xim[:, 0].reshape(-1, 299)
            xim2 = xim[:, 1].reshape(-1, 299)
            xim3 = xim[:, 2].reshape(-1, 299)
            image = np.empty([299,299,3])
            image[:, :, 0] = xim1
            image[:, :, 1] = xim2
            image[:, :, 2] = xim3
            a = im2double(image) * 255
            b = im2double(curHeatMap) * 255
            curHeatMap = a * 0.6 + b * 0.4
            ab = np.hstack((a,b))
            full = np.hstack((curHeatMap, ab))
            # imname = DIRR + '/' + id + '.png'
            # imname1 = DIRR + '/' + id + '_img.png'
            # imname2 = DIRR + '/' + id +'_hm.png'
            imname3 = DIRR + '/' + id + '_full.png'
            # cv2.imwrite(imname, curHeatMap)
            # cv2.imwrite(imname1, a)
            # cv2.imwrite(imname2, b)
            cv2.imwrite(imname3, full)


# Output activation for tSNE
def tSNE_prep(flatnet, ori_test, y, pred, path):
    # format clean up
    tl = np.asmatrix(y)
    tl = tl.argmax(axis=1).astype('uint8')
    pdxt = np.asmatrix(pred)
    prl = pdxt.argmax(axis=1).astype('uint8')
    prl = pd.DataFrame(prl, columns=['Prediction'])
    print(np.shape(flatnet))
    act = pd.DataFrame(np.asmatrix(flatnet))
    outt = pd.DataFrame(pdxt, columns=['NEG_score', 'POS_score'])
    outtlt = pd.DataFrame(tl, columns=['True_label'])
    outt.reset_index(drop=True, inplace=True)
    prl.reset_index(drop=True, inplace=True)
    outtlt.reset_index(drop=True, inplace=True)
    out = pd.concat([outt, prl, outtlt], axis=1)
    ori_test.reset_index(drop=True, inplace=True)
    out.reset_index(drop=True, inplace=True)
    act.reset_index(drop=True, inplace=True)
    out = pd.concat([ori_test, out, act], axis=1)
    if out.shape[0] > 30000:
        out = out.sample(30000, replace=False)
    out.to_csv("../Results/{}/out/For_tSNE.csv".format(path), index=False)
