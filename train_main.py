import torch
from torch.autograd import Variable
import os
import argparse
import csv
# from lib.PraNet_Res2Net import PraNet
# from lib.PVT_FormerNet_rwkv import FormerNet
# from lib.PVT_RWKV_AMSF import FormerNet
# from lib.PVT_RWKV_WMSA import FormerNet
# from lib.PVT_AMSF_WMSA import FormerNet
# from lib.PVT_RWKV_AMSF_WMSA import FormerNet
# from lib.PVT_RFB_AMSF_WMSA import FormerNet
# from lib.PVT_RWKV_AGG_WMSA import FormerNet
# from lib.Res2Net_PRMWNet import FormerNet
# from lib.Vit_PRMWNet import FormerNet
# from lib.PVT_other_AMSF_WMSA import FormerNet
# from lib.PVT_RWKV_AMSF_RA import FormerNet
# from lib.Swin_PRMWNet import FormerNet
# from lib.Swinv2_PRMWNet import FormerNet
# from lib.mit_PRMWNet import FormerNet
# from lib.ResNet_PRMWNet import FormerNet
# from lib.PVT_RWKV import FormerNet
# from lib.PVT_AMSF import FormerNet
# from lib.PVT_WMSA import FormerNet
# from lib.PVT import FormerNet
# from lib.U_Net import NestedUNet
# from lib.Unet_v2 import UNetV2
# from lib.WBANet.networks import WBANetout
# from lib.CASCADE.networks import PVT_CASCADE
# from lib.G_CASCADE.networks import PVT_GCASCADE
# from lib.EMCAD.networks import EMCADNet
# from lib.CaraNet import caranet
# from lib.PraNet_Res2Net import PraNet
# from lib.MambaVision_FormerNet import FormerNet
# from lib.PVT_SSM_AMSF_WMSA import FormerNet
# from lib.PVT_Attention_AMSF_WMSA import FormerNet
# from lib.PVT_linearAttention_AMSF_WMSA import FormerNet
from lib.PVT_RWKV_AMSF_WMSA import FormerNet

from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
# from thop import profile, clever_format
# import segmentation_models_pytorch as smp  # pip install segmentation-models-pytorch
from timm.optim.adan import Adan


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    IOU = 0.0
    smooth = 1

    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res0, res1, res2, res3 = model(image)
        # eval Dice
        res = F.upsample(res0 + res1 + res2 + res3, size=gt.shape, mode='bilinear', align_corners=False)
        # res = F.interpolate(res + res1 + res2 + res3, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res > 0.5).astype(np.float32)

        input = res
        target = np.array(gt)
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)

        # dice
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

        # iou
        iou = (intersection.sum() + smooth) / (input.sum() + target.sum() - intersection.sum() + smooth)
        iou = '{:.4f}'.format(iou)
        iou = float(iou)
        IOU = IOU + iou

    return DSC / num1, IOU / num1


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    total_loss = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            # ---- loss function ----

            loss5 = structure_loss(lateral_map_5, gts)
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)

            loss = loss2 + loss3 + loss4 + loss5

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
                total_loss.update(loss.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[Loss:{:.4f}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(epoch, opt.epoch, i, total_step,
                         total_loss.show(),
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))

    # save model
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'Network-%d.pth' % epoch)
        print('[Saving model:]', save_path + 'Network-%d.pth' % epoch)
    # torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')

    # choose the best model

    global dict_plot_dice, dict_plot_iou

    test1path = './dataset_polyp_pvt/TestDataset/'
    datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    if (epoch + 1) % 1 == 0:

        for dataset in datasets:
            dataset_dice, dataset_iou = test(model, test1path, dataset)
            logging.info('epoch: {}, dataset: {}, dice: {}, iou: {}'.format(epoch, dataset, dataset_dice, dataset_iou))
            print(dataset, ': dice: ', dataset_dice, '   iou: ', dataset_iou)
            dict_plot_dice[dataset].append(dataset_dice)
            dict_plot_iou[dataset].append(dataset_iou)

        dice_test, iou_test = test(model, test_path, 'test')
        print('test: dice: ', dice_test, '    iou: ', iou_test)
        dict_plot_dice['test'].append(dice_test)
        dict_plot_iou['test'].append(iou_test)

        if dice_test > best:
            best = dice_test
            torch.save(model.state_dict(), save_path + 'Network-best.pth')
            torch.save(model.state_dict(), save_path + 'Network-%d-best.pth' % epoch)
            print('##################### best', best, ' ##########################')
            logging.info(
                '############(*^w^*)############# epoch{}:, best:{}'.format(epoch, best))

    return (total_loss.show(),
            loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show())


if __name__ == '__main__':
    dict_plot_dice = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
                      'test': []}
    dict_plot_iou = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
                     'test': []}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################

    model_name = 'PRMWNet-2026-5-25'

    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=70, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str, default='./image_style_transfer_argumentation2',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./dataset_polyp_pvt/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./results/' + model_name + '/')

    opt = parser.parse_args()

    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.basicConfig(filename='./results/' + model_name + '/' + model_name + '_train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = FormerNet()
    model_msg = f"Model Structure:\n{model}"
    logging.info(model_msg)
    print(model_msg)
    model.cuda()


    best = 0

    params_ = model.parameters()

    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params_, opt.lr, weight_decay=1e-4)
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_, opt.lr, weight_decay=1e-4)
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params_, opt.lr, weight_decay=1e-4, momentum=0.9)
    if opt.optimizer == 'Adan':
        optimizer = Adan(params_, opt.lr, weight_decay=1e-4)

    print(optimizer)
    logging.info(optimizer)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    csv_file = os.path.join(opt.train_save, 'loss.csv')
    os.makedirs(opt.train_save, exist_ok=True)

    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss',
                             'lateral2', 'lateral3', 'lateral4', 'lateral5',
                             'dice_CVC-300', 'dice_CVC-ClinicDB', 'dice_Kvasir',
                             'dice_CVC-ColonDB', 'dice_ETIS-LaribPolypDB', 'dice_test',
                             'iou_CVC-300', 'iou_CVC-ClinicDB', 'iou_Kvasir',
                             'iou_CVC-ColonDB', 'iou_ETIS-LaribPolypDB', 'iou_test'
                             ])

    print("##########(*^ω^*)##########", "Start Training", "##########(*^ω^*)##########")

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)

        (loss, epoch_loss_l2, epoch_loss_l3, epoch_loss_l4, epoch_loss_l5) = train(train_loader, model, optimizer,
                                                                                   epoch, opt.test_path)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                format(loss.item(), ".4f"),
                format(epoch_loss_l2.item(), ".4f"),
                format(epoch_loss_l3.item(), ".4f"),
                format(epoch_loss_l4.item(), ".4f"),
                format(epoch_loss_l5.item(), ".4f"),
                format(dict_plot_dice['CVC-300'][-1], ".4f"),
                format(dict_plot_dice['CVC-ClinicDB'][-1], ".4f"),
                format(dict_plot_dice['Kvasir'][-1], ".4f"),
                format(dict_plot_dice['CVC-ColonDB'][-1], ".4f"),
                format(dict_plot_dice['ETIS-LaribPolypDB'][-1], ".4f"),
                format(dict_plot_dice['test'][-1], ".4f"),
                format(dict_plot_iou['CVC-300'][-1], ".4f"),
                format(dict_plot_iou['CVC-ClinicDB'][-1], ".4f"),
                format(dict_plot_iou['Kvasir'][-1], ".4f"),
                format(dict_plot_iou['CVC-ColonDB'][-1], ".4f"),
                format(dict_plot_iou['ETIS-LaribPolypDB'][-1], ".4f"),
                format(dict_plot_iou['test'][-1], ".4f")
            ])

    print("##########(*^ω^*)##########", "Finish Training", "##########(*^ω^*)##########")



