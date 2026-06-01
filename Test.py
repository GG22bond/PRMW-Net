import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
# from lib.PVT_RWKV_AMSF_WMSA import FormerNet
# from lib.smp_unet import UNet, UnetPlusPlus
# from lib.PraNet_Res2Net import PraNet
from lib.PVT_RWKV_AMSF_WMSA import FormerNet
# from lib.SANet import Model
# from lib.PVT_polyp import PolypPVT
# from lib.MSRAformer import MSRAformer
# from lib.CFANet import CFANet
# from lib.CASCADE.networks import PVT_CASCADE
# from lib.G_CASCADE.networks import PVT_GCASCADE
# from lib.EMCAD.networks import EMCADNet
# from lib.CTNet.Mymodel13 import PolypModule
# from lib.BMANet import BMANet
# from lib.Unet_v2 import UNetV2
# from lib.Pranet_v2.pranet import PraNet_V2
# from lib.WBANet.networks import WBANetout
from utils.dataloader import test_dataset

import imageio


##  ETIS-LaribPolypDB：https://www.kaggle.com/datasets/nguyenvoquocduong/etis-laribpolypdb
##  CVC-ClinicDB：https://www.kaggle.com/datasets/balraj98/cvcclinicdb  https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?dl=0
##  CVC-300: https://www.kaggle.com/datasets/nourabentaher/cvc-300


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='snapshots/Comparative_experiment/PRMWNet/Network-best.pth')

# for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
for _data_name in ['ETIS-LaribPolypDB']:
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './inference/PraNet/{}/'.format(_data_name)
    opt = parser.parse_args()

    # MODEL
    # model = FormerNet()
    model = FormerNet()
    # model = PolypModule()

    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5, res4, res3, res2, = model(image)

        # res3, res2 = model(image)
        # res = res3
        res = res5 + res4 + res3 + res2

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        # res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)

        # misc.save(save_path+name, res)
        imageio.imwrite(save_path + name, res)



