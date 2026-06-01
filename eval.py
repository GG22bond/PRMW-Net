import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
from skimage.metrics import structural_similarity as ssim
from lib.PVT_RWKV_AMSF_WMSA import FormerNet

from utils.dataloader import test_dataset
from thop import profile, clever_format


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='snapshots/Comparative_experiment/PRMWNet/Network-best.pth')


def mae_score(pred, gt):
    pred, gt = pred.astype(np.float32), gt.astype(np.float32)
    return np.mean(np.abs(pred - gt))


def s_alpha(pred, gt, alpha=0.5):
    pred, gt = pred.astype(np.float32), gt.astype(np.float32)
    x_mean = np.mean(pred)
    y_mean = np.mean(gt)
    So = (2 * x_mean * y_mean) / (x_mean**2 + y_mean**2 + 1e-8)
    Sr = ssim(pred, gt, data_range=1.0)
    return alpha * So + (1 - alpha) * Sr


def e_phi_max(pred, gt):
    pred, gt = pred.astype(np.float32), gt.astype(np.float32)
    return np.mean(1 - np.abs(pred - gt))


def f_beta_weighted(pred, gt, beta2=0.3):
    pred, gt = pred.astype(np.float32), gt.astype(np.float32)
    TP = np.sum(pred * gt)
    FP = np.sum(pred * (1 - gt))
    FN = np.sum((1 - pred) * gt)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    return (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)

model = FormerNet()

# Dataset names
for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']:
    data_path = './dataset_polyp_pvt/TestDataset/{}/'.format(_data_name)

    opt = parser.parse_args()

    # Initialize the model

    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    dice_scores = []
    iou_scores = []
    f_beta_scores = []
    s_alpha_scores = []
    e_phi_scores = []
    mae_scores = []
    # hd95_scores = []

    total_infer_time = 0.0
    total_frames = 0

    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            torch.cuda.synchronize()
            start = time.time()

            res4, res3, res2, res1 = model(image)
            res = res4 + res3 + res2 + res1

            torch.cuda.synchronize()
            end = time.time()

            total_infer_time += (end - start)
            total_frames += 1

            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()

            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # Binarize
            res_bin = (res > 0.5).astype(np.uint8)  # Thresholding at 0.5

            # res_bin = res

            # Dice
            intersection = np.sum(res_bin * gt)
            dice = (2.0 * intersection) / (np.sum(res_bin) + np.sum(gt))
            dice_scores.append(dice)

            # IoU
            union = np.sum(res_bin) + np.sum(gt) - intersection
            iou = intersection / float(union)
            iou_scores.append(iou)

            # F_beta^w
            f_beta_scores.append(f_beta_weighted(res_bin, gt))

            # S_alpha
            s_alpha_scores.append(s_alpha(res_bin, gt))

            # E_phi^max
            e_phi_scores.append(e_phi_max(res, gt))

            # MAE
            mae_scores.append(mae_score(res, gt))

            # # HD95
            # if np.sum(res_bin) > 0 and np.sum(gt) > 0:
            #     hd = hd95(res_bin, gt.astype(np.uint8))
            #     hd95_scores.append(hd)
            # else:
            #     hd95_scores.append(np.nan)


    # Calculate average Dice and IoU
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_fbeta = np.mean(f_beta_scores)
    avg_salpha = np.mean(s_alpha_scores)
    avg_ephi = np.mean(e_phi_scores)
    avg_mae = np.mean(mae_scores)
    # avg_hd95 = np.nanmean(hd95_scores)


    if total_infer_time > 0:
        fps = total_frames / total_infer_time
    else:
        fps = 0.0


    print("{:*^60}".format("Dataset:" + _data_name))
    print(f"Average Dice score: {avg_dice:.4f}")
    print(f"Average IoU score: {avg_iou:.4f}")
    print(f"Average F_beta^w: {avg_fbeta:.4f}")
    print(f"Average S_alpha: {avg_salpha:.4f}")
    print(f"Average E_phi^max: {avg_ephi:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"FPS: {fps:.2f}")
    # print(f"Average HD95: {avg_hd95:.4f}")


model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

dummy_input = torch.randn(1, 3, opt.testsize, opt.testsize).cuda()
flops, params = profile(model, inputs=(dummy_input,), verbose=False)
flops, params = clever_format([flops * 2, params], "%.3f")


print('Total FLOPS: %s' % (flops))
print('Total params: %s' % (params))


