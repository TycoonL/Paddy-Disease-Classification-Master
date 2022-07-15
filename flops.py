from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch

def analysis_floaps(model,device):
    # # 分析FLOPs
    tensor = torch.rand(1, 3, 224, 224).to(device)
    flops = FlopCountAnalysis(model, tensor)
    print("FLOPs: ", flops.total())
    # #分析parameters
    print(parameter_count_table(model))