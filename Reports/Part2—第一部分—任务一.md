# Part2——第一部分——任务一

## 在 get_args 函数中加入 **“derainnet”** 和 **“mymodel”** 两个 choice

参数注册choice列表增加**“derainnet”** 和 **“mymodel”** 两个 choice：

```python
parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "unet","derainnet","mymodel"], help="Model to use")
```

##  DerainNet 和 Mymodel 两个模型文件

在导入区先导入模型类（用baseline和unet先套个皮）：

```python
from models.DerainNet import DerainNet
from models.Mymodel import Mymodel
```

注意到：

出题人说**本部分中，我们规定：损失函数只能使用 L2 Loss**

但是后面可能要用到L1

所以main.py里面加上一个新的命令行参数

并加上逻辑判断

```python
parser.add_argument("--loss", type=str, default="L1", choices=["L1", "L2"], help="L1 loss or L2 loss")

if args.loss == "L1":
        pixel_loss_fn = nn.L1Loss()
    elif args.loss == "L2":
        pixel_loss_fn = nn.MSELoss()
```

## 判断条件补充
```python
if args.model == "baseline":
        model = BaselineNet().to(device)
    elif args.model == "unet":
        model = UNet().to(device)
    elif args.model == "derainnet":
        model = DerainNet().to(device)
    elif args.model == "mymodel":
        model = Mymodel().to(device)
```
