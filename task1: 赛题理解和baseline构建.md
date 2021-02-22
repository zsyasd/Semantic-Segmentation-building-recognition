task1仅仅是作为一个简单介绍。关于赛题阅读和评价方法。同时也大概跑通了一个baseline来看看结果。
## 1 赛题理解
- 赛题名称：零基础入门语义分割-地表建筑物识别
- 赛题目标：通过本次赛题可以引导大家熟练掌握语义分割任务的定义，具体的解题流程和相应的模型，并掌握语义分割任务的发展。
- 赛题任务：赛题以计算机视觉为背景，要求选手使用给定的航拍图像训练模型并完成地表建筑物识别任务。

### 1.1 学习目标
- 理解赛题背景和赛题数据
- 完成赛题报名和数据下载，理解赛题的解题思路

### 1.2 赛题数据

遥感技术已成为获取地表覆盖信息最为行之有效的手段，遥感技术已经成功应用于地表覆盖检测、植被面积检测和建筑物检测任务。本赛题使用航拍数据，需要参赛选手完成地表建筑物识别，将地表航拍图像素划分为有建筑物和无建筑物两类。

如下图，左边为原始航拍图，右边为对应的建筑物标注。

![](https://github.com/datawhalechina/team-learning-cv/blob/master/AerialImageSegmentation/img/data-example.png)

赛题数据来源（Inria Aerial Image Labeling），并进行拆分处理。数据集报名后可见并可下载。赛题数据为航拍图，需要参赛选手识别图片中的地表建筑具体像素位置。
### 1.3 数据标签

赛题为语义分割任务，因此具体的标签为图像像素类别。在赛题数据中像素属于2类（无建筑物和有建筑物），因此标签为有建筑物的像素。赛题原始图片为jpg格式，标签为RLE编码的字符串。

RLE全称（run-length encoding），翻译为游程编码或行程长度编码，对连续的黑、白像素数以不同的码字进行编码。RLE是一种简单的非破坏性资料压缩法，经常用在在语义分割比赛中对标签进行编码。

RLE与图片之间的转换如下：

```python
import numpy as np
import pandas as pd
import cv2

# 将图片编码为rle格式
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 将rle格式进行解码为图片
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')
```

### 1.4 评价指标

赛题使用Dice coefficient来衡量选手结果与真实标签的差异性，Dice coefficient可以按像素差异性来比较结果的差异性。Dice coefficient的具体计算方式如下：

![](https://user-images.githubusercontent.com/55370336/108602833-3cce8200-73df-11eb-855c-e6039fc2fe6e.png)

其中$X$是预测结果，$Y$为真实标签的结果。当$X$与$Y$完全相同时Dice coefficient为1，排行榜使用所有测试集图片的平均Dice coefficient来衡量，分数值越大越好。

### 1.5 读取数据

| FileName                | Size     |                                                         含义 |
| :---------------------- | :------- | -----------------------------------------------------------: |
| test_a.zip              | 314.49MB | 测试集A榜图片 |
| test_a_samplesubmit.csv | 46.39KB  | 测试集A榜提交样例 |
| train.zip               | 3.68GB   | 训练集图片 |
| train_mask.csv.zip      | 97.52MB  | 训练集图片标注 |

具体数据读取案例：

```
import pandas as pd
import cv2
train_mask = pd.read_csv('train_mask.csv', sep='\t', names=['name', 'mask'])

# 读取第一张图，并将对于的rle解码为mask矩阵
img = cv2.imread('train/'+ train_mask['name'].iloc[0])
mask = rle_decode(train_mask['mask'].iloc[0])

print(rle_encode(mask) == train_mask['mask'].iloc[0])
# 结果为True
```

### 1.6 解题思路

由于本次赛题是一个典型的语义分割任务，因此可以直接使用语义分割的模型来完成：
- 步骤1：使用FCN模型模型跑通具体模型训练过程，并对结果进行预测提交；
- 步骤2：在现有基础上加入数据扩增方法，并划分验证集以监督模型精度；
- 步骤3：使用更加强大模型结构（如Unet和PSPNet）或尺寸更大的输入完成训练；
- 步骤4：训练多个模型完成模型集成操作；

### 1.7 本章小结

本章主要对赛题背景和主要任务进行讲解，并多对赛题数据和标注读取方式进行介绍，最后列举了赛题解题思路。

### 1.8 课后作业

1. 理解RLE编码过程，并完成赛题数据读取并可视化；
2. 统计所有图片整图中没有任何建筑物像素占所有训练集图片的比例；
3. 统计所有图片中建筑物像素占所有像素的比例；
4. 统计所有图片中建筑物区域平均区域大小；

## 2 baseline代码分析

### 2.1 将图片编码为rle格式

```python
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F') # 1）按列将二维数组展平成一维
    pixels = np.concatenate([[0], pixels, [0]])# 2）两端补0，目的是为了比较0到1变化或者1到0变化的下标
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1 #下标+1
    runs[1::2] -= runs[::2] #变成下表+个数的RLE编码形式
    return ' '.join(str(x) for x in runs)
```

### 2.2 将rle格式进行解码为图片

```
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split() #去掉RLE编码的时候返回的每两个数字的空格间隔
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # s[0:][::2表示从1开始的下标，[1:][::2]表示个数
    
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')
```

### 2.3 定义数据集
```python
class TianChiDataset(D.Dataset):
    def __init__(self, paths, rles, transform, test_mode=False):
        self.paths = paths
        self.rles = rles
        self.transform = transform
        self.test_mode = test_mode
        
        self.len = len(paths)
        self.as_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])
        
    # get data operation
    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        if not self.test_mode:
            mask = rle_decode(self.rles[index])
            augments = self.transform(image=img, mask=mask)
            return self.as_tensor(augments['image']), augments['mask'][None]
        else:
            return self.as_tensor(img), ''        
    
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
```
没特别多好讲的，利用opencv作了数据的预处理（也就是一些剪裁反转啥的随便爱咋弄咋弄，自由度很高的玩意儿）

baseline到这一步就检验了一下转换的一一对应性能，代码如下
<img width="730" alt="image" src="https://user-images.githubusercontent.com/55370336/108603231-cb440300-73e1-11eb-8e50-e7100d637166.png">
代码基准上面有，所以就放我运行的截图，可以看到是true的。~~这不是废话吗~~

```python
dataset = TianChiDataset(
    train_mask['name'].values,
    train_mask['mask'].fillna('').values,
    trfm, False
)
```
这也是很常规的操作，实例化数据集，注意`fillna('')`起到补全缺失值为`''`的作用

### 2.4 模型定义
```python
def get_model():
    model = torchvision.models.segmentation.fcn_resnet101(True)
    
#     pth = torch.load("../input/pretrain-coco-weights-pytorch/fcn_resnet50_coco-1167a1af.pth")
#     for key in ["aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias", "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.1.num_batches_tracked", "aux_classifier.4.weight", "aux_classifier.4.bias"]:
#         del pth[key]
    
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    return model

@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        output = model(image)['out']
        loss = loss_fn(output, target)
        losses.append(loss.item())
        
    return np.array(losses).mean()
```
注意我这里吧网络换成了fcn_resnet101，原本是r50，同时我的batch size保持了原本的32。这里需要提醒没有服务器的小伙伴们切勿模仿，我用的是实验室给的v100.经实测32batch size的情况下r50占用显存24g，r101直接去到31g了。同时r101跑一个epoch所用的时间大概是5分钟左右。

另外r101实际上有点太深了，如果没有进行足够的数据扩充。我跑了大概10轮，看loss就能看到大概从第7轮开始就陷入过拟合状态了。

<img width="721" alt="image" src="https://user-images.githubusercontent.com/55370336/108603528-65f11180-73e3-11eb-8aec-8def8acd4bf2.png">


### 2.5 优化器定义
```python
model = get_model()
model.to(DEVICE);

optimizer = torch.optim.AdamW(model.parameters(),
                  lr=1e-4, weight_decay=1e-3)

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2,-1)):

        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims
    
    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc
    
bce_fn = nn.BCEWithLogitsLoss()
dice_fn = SoftDiceLoss()

def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return 0.8*bce+ 0.2*dice
```

### 2.6 开始训练
```
header = r'''
        Train | Valid
Epoch |  Loss |  Loss | Time, m
'''
#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.3f}'*2 + '\u2502{:6.2f}'
print(header)

EPOCHES = 10
best_loss = 10
for epoch in range(1, EPOCHES+1):
    losses = []
    start_time = time.time()
    model.train()
    for image, target in tqdm_notebook(loader):
        
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        optimizer.zero_grad()
        output = model(image)['out']
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # print(loss.item())
        
    vloss = validation(model, vloader, loss_fn)
    print(raw_line.format(epoch, np.array(losses).mean(), vloss,
                              (time.time()-start_time)/60**1))
    losses = []
    
    if vloss < best_loss:
        best_loss = vloss
        torch.save(model.state_dict(), 'model_best.pth')
```

### 2.7 对模型进行预测
```python3
trfm = T.Compose([
    T.ToPILImage(),
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.625, 0.448, 0.688],
                [0.131, 0.177, 0.101]),
])

subm = []

model.load_state_dict(torch.load("./model_best.pth"))
model.eval()
```
```
test_mask = pd.read_csv('test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
test_mask['name'] = test_mask['name'].apply(lambda x: 'test_a/' + x)

for idx, name in enumerate(tqdm_notebook(test_mask['name'].iloc[:])):
    image = cv2.imread(name)
    image = trfm(image)
    with torch.no_grad():
        image = image.to(DEVICE)[None]
        score = model(image)['out'][0][0]
        score_sigmoid = score.sigmoid().cpu().numpy()
        score_sigmoid = (score_sigmoid > 0.5).astype(np.uint8)
        score_sigmoid = cv2.resize(score_sigmoid, (512, 512))

        
        # break
    subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])
```
### 2.8 可视化预测结果
```
plt.figure(figsize=(16,8))
plt.subplot(121)
plt.imshow(rle_decode(subm[1].fillna('').iloc[0]), cmap='gray')
plt.subplot(122)
plt.imshow(cv2.imread('test_a/' + subm[0].iloc[0]));
```
查看第一张图的预测结果，当然也可以改成第二张第三张这些来看。

<img width="940" alt="image" src="https://user-images.githubusercontent.com/55370336/108603617-f3346600-73e3-11eb-9cc2-6a8de4fb8f4f.png">

<img width="940" alt="image" src="https://user-images.githubusercontent.com/55370336/108603685-5de5a180-73e4-11eb-99fa-88d8d55384bf.png">
<img width="933" alt="image" src="https://user-images.githubusercontent.com/55370336/108603699-705fdb00-73e4-11eb-9675-4acd22bce2be.png">
<img width="932" alt="image" src="https://user-images.githubusercontent.com/55370336/108603707-7c4b9d00-73e4-11eb-8ab4-b303e4c40ab3.png">
分别是第一张 第二张 第三张 和第25张的预测。

最终提交结果是0.7706

<img width="350" alt="image" src="https://user-images.githubusercontent.com/55370336/108603762-c5035600-73e4-11eb-971b-22c03edf1769.png">

<img width="683" alt="image" src="https://user-images.githubusercontent.com/55370336/108603769-d2b8db80-73e4-11eb-9d92-a9a668087e21.png">

毕竟只是瞎几把挑的。后续估计还能有进步。

## 3 课后作业

### 统计所有图片整图中没有任何建筑物像素占所有训练集图片的比例
<img width="616" alt="image" src="https://user-images.githubusercontent.com/55370336/108603873-57a3f500-73e5-11eb-9a7d-da134998d256.png">

### 统计所有图片中建筑物像素占所有像素的比例
```
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


train_mask = pd.read_csv("train_mask.csv",sep="\t",names=["name","mask"])
train_mask["mask"]=train_mask["mask"].fillna("")
l = len(train_mask)

ratio_ls = []
for i in tqdm(range(l)):
    if train_mask["mask"].iloc[i]!="":
        ls = list(map(int,train_mask["mask"].iloc[i].split(" ")))
        number = sum(ls[1::2])
        pic_path = ""+"train/"+train_mask["name"].iloc[i]
        img = np.array(Image.open(pic_path))
        ratio = number/(img.shape[0]*img.shape[1])
    else:
        ratio = 0

    ratio_ls.append(ratio)
pd.Series(ratio_ls).to_csv("ratio_ls")
```
### 统计所有图片中建筑物区域平均区域大小
```python
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

train_mask = pd.read_csv("数据集/train_mask.csv",sep="\t",names=["name","mask"])
train_mask["mask"]=train_mask["mask"].fillna("")
l = len(train_mask)

sum_ls = []
for i in tqdm(range(l)):
    if train_mask["mask"].iloc[i]!="":
        ls = list(map(int,train_mask["mask"].iloc[i].split(" ")))
        number = sum(ls[1::2])
        # pic_path = "数据集/"+"train/"+train_mask["name"].iloc[i]
        # img = np.array(Image.open(pic_path))
        # ratio = number/(img.shape[0]*img.shape[1])
    else:
        number = 0
    sum_ls.append(number)
pd.Series(sum_ls).to_csv("point_sum_ls")
```
