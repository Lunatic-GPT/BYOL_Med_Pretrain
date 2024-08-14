import torch.onnx
import sys
import torchvision


# Function to Convert to ONNX
def Convert_ONNX(model, input_size, tar_device_id, tar_onnx_path):
    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(input_size).to(torch.device(tar_device_id))

    # Export the model
    torch.onnx.export(model.to(torch.device(tar_device_id)),  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      tar_onnx_path,  # where to save the model
                      input_names=['input'],  # the model's input names
                      output_names=['output'])  # the model's output names
    #  export_params=True,  # store the trained parameter weights inside the model file
    #  opset_version=10,    # the ONNX version to export the model to
    #  do_constant_folding=True,  # whether to execute constant folding for optimization
    #  input_names = ['modelInput'],  # the model's input names
    #  output_names = ['modelOutput'],  # the model's output names
    #  dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
    #                         'modelOutput' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')


from collections import OrderedDict

import torch
from torch import nn


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


# [('feat1', torch.Size([1, 64, 56, 56])), ('feat2', torch.Size([1, 256, 14, 14]))]


from torchvision.models.feature_extraction import create_feature_extractor
def byol_trans():
    model_path_base = "weight/window160_sz144/models_457.pth"
    out_path_base = "byol_alldata_1channel_win160_sz144_epoch457.pth"
    device = "cpu"
    # Load PyTorch Model
    model_param = torch.load(model_path_base, map_location=device)
    model_param.eval()

    resnet = model_param.net
    model = create_feature_extractor(resnet, {'avgpool': 'feat1'})

    tar_device_id = 0
    model.to(device)
    input_size = [1, 1, 144, 144]
    # Conversion to ONNX
    Convert_ONNX(model, input_size, tar_device_id, out_path_base)


if __name__ == "__main__":
    byol_trans()
