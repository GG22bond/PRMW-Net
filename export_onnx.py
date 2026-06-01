"""
https://github.com/Ascend/ModelZoo-PyTorch/tree/6a2804a358a5b18e3dac1ab902f41f88e240b00f/ACL_PyTorch/contrib/cv/segmentation/PraNet
"""

import torch
import torch.onnx
import sys
# sys.path.append('./PraNet')
from collections import OrderedDict
from lib.PVT_RWKV_AMSF_WMSA import FormerNet
# from lib.smp_unet import UNet, UnetPlusPlus
# from lib.SANet import Model
# from lib.PraNet_Res2Net import PraNet

def proc_node_module(checkpoint, attr_name):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[attr_name].items():
        if k[0:7] == "module.":
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict

def convert(pth_file_path, onnx_file_path):
    model = FormerNet()
    # model = PraNet()
    pretrained_dict = torch.load(pth_file_path, map_location="cpu")
    model.load_state_dict({k.replace('module.',''):v for k, v in pretrained_dict.items()}, strict=False)
    if "fc.weight" in pretrained_dict:
        pretrained_dict.pop('fc.weight')
        pretrained_dict.pop('fc.bias')
    model.load_state_dict(pretrained_dict)
    model.eval()

    input_names = ["actual_input_1"]

    # output_names = ["output1"]
    # dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}

    output_names = ["output2", "output3", "output4", "output5"]
    dynamic_axes={"actual_input_1": {0: "-1"},
                  "output2": {0: "-1"},
                  "output3": {0: "-1"},
                  "output4": {0: "-1"},
                  "output5": {0: "-1"},
                  }


    dummy_input = torch.randn(1, 3, 352, 352)
    torch.onnx.export(model, dummy_input, onnx_file_path,
                      input_names=input_names, dynamic_axes=dynamic_axes, output_names=output_names,
                      opset_version=11)

if __name__ == "__main__":
    pth_path = sys.argv[1]
    onnx_path = sys.argv[2]
    convert(pth_path, onnx_path)


# python export_onnx.py weight/Pranet/PraNet-19.pth weight/Pranet/PraNet-19.onnx
# python export_onnx.py snapshots/Ablation_experiment/PVT_RWKV_AMSF_WMSA-2025-9-04/Network-best.pth snapshots/Ablation_experiment/PVT_RWKV_AMSF_WMSA-2025-9-04/my_model.onnx


