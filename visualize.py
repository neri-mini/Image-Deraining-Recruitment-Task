import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.models import vgg19, VGG19_Weights
from dataset import DerainDataset
from torch.utils.data import DataLoader

def main():
    dataset = DerainDataset(root_dir="./data/Rain200L",is_train=False)
    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
    x_raw,_,_ = next(iter(DataLoader(dataset,batch_size=1)))
    example_list = [4,9,18]
    example_dict = {"raw" : x_raw}
    x = x_raw
    for i,layer in enumerate(vgg):      
        x = layer(x)
        if i in example_list:
            example_dict[i] = x.detach()

    os.makedirs("./compare_image",exist_ok=True)
    save_image(example_dict["raw"], "./coampare_image/raw_image.png")
    print("已保存原图")
    for i in example_list:
        tensor = example_dict[i]
        tensor = tensor.squeeze(0)
        tensor = tensor.unsqueeze(1)
        save_image(tensor,f"./coampare_image/{i}_layer_image.png", nrow=8, normalize=True)
        print(f"保存第{i}层处理之后的图像")
if __name__ == "__main__":
        main()                   




