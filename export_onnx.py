import os
import torch
from torch.utils.data import DataLoader
from models.mymodel import Mymodel
from dataset import DerainDataset

def main():
        device = torch.device("cpu")
        model = Mymodel().to(device).eval()
        root_dir = "./data/Rain200L"
        dataset = DerainDataset(root_dir,is_train=False)
        rainy, _, _ = next(iter(DataLoader(dataset,batch_size=1)))
        rainy = rainy.to(device)

        checkpoint_path = "savemodel_onnx"
        mymodel_onnx_path = os.path.join(checkpoint_path,"mymodel.onnx")
        os.makedirs(checkpoint_path,exist_ok=True)

        n = sum(p.numel() for p in model.parameters())
        print(f"模型参数量大小为{n / 1e6 :.3f}M")
        torch.onnx.export(model,rainy,mymodel_onnx_path,export_params=True,dynamo=False,
                         opset_version=17,input_names=["rainy"],output_names=["derain"],
                         dynamic_axes={
                            "rainy":{0:"batch",2:"height",3:"width"},
                            "derain":{0:"batch",2:"height",3:"width"}
                         },
                         )
        print(f"已把mymodel导出至{mymodel_onnx_path},文件大小为{os.path.getsize(mymodel_onnx_path) / (1024 ** 2):.2f} MB")

if __name__ == "__main__":
        main()                         
