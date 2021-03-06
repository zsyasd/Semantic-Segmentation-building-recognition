## task2 数据扩充

前面对baseline中可以看到，当尝试使用fcn_resnet101来拟合模型之后似乎出现了过拟合现象，另外train loss也并没有降下来太多。那么解决为了解决过拟合同时降低loss，我们可以使用数据扩充的
方法来增加我们的数据量

`吴恩达：「对于计算机视觉来说，几乎我们遇到的所有问题都与训练集不足有关」` 

~~（原话应该有点出入，但是大体意思是这个。我看他网课的时候对这句话特别有印象hhhh）~~

```数据扩增是一种有效的正则化方法，可以防止模型过拟合，在深度学习模型的训练过程中应用广泛。数据扩增的目的是增加数据集中样本的数据量，同时也可以有效增加样本的语义空间。

需注意：

1. 不同的数据，拥有不同的数据扩增方法；

2. 数据扩增方法需要考虑合理性，不要随意使用；

3. 数据扩增方法需要与具体任何相结合，同时要考虑到标签的变化；

对于图像分类，数据扩增方法可以分为两类：

1. 标签不变的数据扩增方法：数据变换之后图像类别不变；
2. 标签变化的数据扩增方法：数据变换之后图像类别变化；

而对于语义分割而言，常规的数据扩增方法都会改变图像的标签。如水平翻转、垂直翻转、旋转90%、旋转和随机裁剪，这些常见的数据扩增方法都会改变图像的标签，即会导致地标建筑物的像素发生改变。
```

![](https://github.com/datawhalechina/team-learning-cv/blob/master/AerialImageSegmentation/img/albu-example.jpeg)

那么对于cv的常用的数据扩充方法，还得是我 ~~或人~~ 不是，是OpenCV和albumentations。

### 2.1 OpenCV

OpenCV可以很方便的完成数据读取、图像变化、边缘检测和模式识别等任务。为了加深各位对数据可做的影响，这里介绍OpenCV完成数据扩增的操作。

```python
# 首先读取原始图片
img = cv2.imread(train_mask['name'].iloc[0])
mask = rle_decode(train_mask['mask'].iloc[0])

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.imshow(mask)
```

![](https://user-images.githubusercontent.com/55370336/108654523-d2633200-7503-11eb-9142-d4f930148e02.png)

```python
# 水平翻转
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(cv2.flip(img, 0))

plt.subplot(1, 2, 2)
plt.imshow(cv2.flip(mask, 0))
```
![image](https://user-images.githubusercontent.com/55370336/108654876-89f84400-7504-11eb-9420-5b85d1c2ad06.png)


```python
# 垂直翻转
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(cv2.flip(img, 0))

plt.subplot(1, 2, 2)
plt.imshow(cv2.flip(mask, 0))
```

![](https://user-images.githubusercontent.com/55370336/108654547-dee78a80-7503-11eb-9d1d-ec07d8d55c0b.png)

```python
# 负水平翻转
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(cv2.flip(img, -1))

plt.subplot(1, 2, 2)
plt.imshow(cv2.flip(mask, -1))
```
![image](https://user-images.githubusercontent.com/55370336/108733188-39ff9880-7569-11eb-985a-24c1ebcb73de.png)

```python
# 随机裁剪
x, y = np.random.randint(0, 256), np.random.randint(0, 256)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(img[x:x+256, y:y+256])

plt.subplot(1, 2, 2)
plt.imshow(mask[x:x+256, y:y+256])
```
![image](https://user-images.githubusercontent.com/55370336/108655646-bf049680-7504-11eb-9e73-c5ca76c874fb.png)
![image](https://user-images.githubusercontent.com/55370336/108655823-c9269500-7504-11eb-9577-7e5a9022613a.png)

随机剪裁每次提交都不一样的哦

### 2.2 albumentations数据扩增

albumentations是基于OpenCV的快速训练数据增强库，拥有非常简单且强大的可以用于多种任务（分割、检测）的接口，易于定制且添加其他框架非常方便。

albumentations也是计算机视觉数据竞赛中最常用的库：

- GitHub： [https://github.com/albumentations-team/albumentations](https://link.zhihu.com/?target=https%3A//github.com/albumentations-team/albumentations)
- 示例：[https://github.com/albumentations-team/albumentations_examples](https://link.zhihu.com/?target=https%3A//github.com/albumentations-team/albumentations_examples)

与OpenCV相比albumentations具有以下优点：

- albumentations支持的操作更多，使用更加方便；
- albumentations可以与深度学习框架（Keras或Pytorch）配合使用；
- albumentations支持各种任务（图像分流）的数据扩增操作

albumentations它可以对数据集进行逐像素的转换，如模糊、下采样、高斯造点、高斯模糊、动态模糊、RGB转换、随机雾化等；也可以进行空间转换（同时也会对目标进行转换），如裁剪、翻转、随机裁剪等。

```python
import albumentations as A

# 水平翻转
augments = A.HorizontalFlip(p=1)(image=img, mask=mask)
img_aug, mask_aug = augments['image'], augments['mask']

# 随机裁剪
augments = A.RandomCrop(p=1, height=256, width=256)(image=img, mask=mask)
img_aug, mask_aug = augments['image'], augments['mask']

# 旋转
augments = A.ShiftScaleRotate(p=1)(image=img, mask=mask)
img_aug, mask_aug = augments['image'], augments['mask']
```

albumentations还可以组合多个数据扩增操作得到更加复杂的数据扩增操作：

```python
trfm = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
])

augments = trfm(image=img, mask=mask)
img_aug, mask_aug = augments['image'], augments['mask']
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(augments['image'])

plt.subplot(1, 2, 2)
plt.imshow(augments['mask'])aug
```
![image](https://user-images.githubusercontent.com/55370336/108733387-6f0beb00-7569-11eb-8bfe-c37a02057f2e.png)

这里我是查看了第5张图片第情况，注意如果你是想查看第二张图片情况的话会报错。查看训练集数据会发现

```python
train_mask
```
![image](https://user-images.githubusercontent.com/55370336/108683261-592c0500-752c-11eb-9d0c-e955c68c3a83.png)

说明如果当该图中没有建筑物的话，是没办法查看的。

### 2.3 Pytorch数据读取

- Dataset：数据集，对数据进行读取并进行数据扩增；
- DataLoder：数据读取器，对Dataset进行封装并进行批量读取；

定义Dataset：

```python
import torch.utils.data as D
class TianChiDataset(D.Dataset):
    def __init__(self, paths, rles, transform):
        self.paths = paths
        self.rles = rles
        self.transform = transform
        self.len = len(paths)

    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        mask = rle_decode(self.rles[index])
        augments = self.transform(image=img, mask=mask)
        return self.as_tensor(augments['image']), augments['mask'][None]
   
    def __len__(self):
        return self.len
```

实例化Dataset：

```python
trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
])

dataset = TianChiDataset(
    train_mask['name'].values,
    train_mask['mask'].fillna('').values,
    trfm
)
```

实例化DataLoder，批大小为10：

```python
loader = D.DataLoader(
    dataset, batch_size=10, shuffle=True, num_workers=0)
```

### 2.4 课后作业
```python
# 使用OpenCV完成图像加噪数据扩增
def addGaussianNoise(image,percetage): 
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1]) 
    for i in range(G_NoiseNum): 
        temp_x = np.random.randint(0,h) 
        temp_y = np.random.randint(0,w) 
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0] 
    return G_Noiseimg

Gimg = addGaussianNoise(img, 0.5)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(Gimg)

plt.subplot(1, 2, 2)
plt.imshow(mask)
```

![image](https://user-images.githubusercontent.com/55370336/108714831-061a7800-7555-11eb-8787-99ea377f3347.png)

当然你也可以把precetage挑大一点，上面这张图是0.5的，下面放张1的
或者你也可以自己挑一下方差和期望

![image](https://user-images.githubusercontent.com/55370336/108715341-b5efe580-7555-11eb-9b26-3b24c978212e.png)

```python
# 使用OpenCV完成图像旋转数据扩增
def rotate(image, angle=15, scale=0.9):
    # 角度可以自己调
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image
    
roimg = rotate(img)
romask = rotate(mask)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(roimg)

plt.subplot(1, 2, 2)
plt.imshow(romask)
```

![image](https://user-images.githubusercontent.com/55370336/108715731-344c8780-7556-11eb-9178-79bc4d93c18b.png)

#### albumentations做组合变换

变换不仅可以单独使用，还可以将这些组合起来，这就需要用到 `Compose` 类，该类继承自 `BaseCompose`。`Compose` 类含有以下参数：

* `transforms`：转换类的数组，`list`类型
* `bbox_params`：用于 `bounding boxes` 转换的参数，`BboxPoarams` 类型
* `keypoint_params`：用于 `keypoints` 转换的参数， `KeypointParams` 类型
* `additional_targets`：`key`新`target` 名字，`value` 为旧 `target` 名字的 `dict`，如 `{'image2': 'image'}`，`dict` 类型
* `p`：使用这些变换的概率，默认值为 1.0

```python
# 使用albumentations其他的的操作完成扩增操作；
image3 = Compose([
        # 对比度受限直方图均衡
            #（Contrast Limited Adaptive Histogram Equalization）
        CLAHE(),
        # 随机旋转 90°
        RandomRotate90(),
        # 转置
        Transpose(),
        # 随机仿射变换
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        # 模糊
        Blur(blur_limit=3),
        # 光学畸变
        OpticalDistortion(),
        # 网格畸变
        GridDistortion(),
        # 随机改变图片的 HUE、饱和度和值
        HueSaturationValue()
    ], p=1.0)

augments = image3(image=img, mask=mask)
img_aug, mask_aug = augments['image'], augments['mask']

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(augments['image'])

plt.subplot(1, 2, 2)
plt.imshow(augments['mask'])
```

多次运行结果会不一样
![image](https://user-images.githubusercontent.com/55370336/108734871-dbd3b500-756a-11eb-819c-31f4e6bfe4da.png)
![image](https://user-images.githubusercontent.com/55370336/108734904-e68e4a00-756a-11eb-914e-881bd618ccec.png)

#### 组合与随机选择（Compose & OneOf）

使用上面的组合方式，执行的过程中会把每个在 transforms 中的转换都执行一遍，但有时候可能我们执行某一组类似操作中的一个，那么这时候就可以配合 OneOf 类来实现此功能。OneOf 类含有以下是参数：

transforms：转换类的列表
p：使转换方法的概率，默认值为 0.5

```python
image4 = Compose([
        RandomRotate90(),
        # 翻转
        Flip(),
        Transpose(),
        OneOf([
            # 高斯噪点
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            # 模糊相关操作
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            # 畸变相关操作
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            # 锐化、浮雕等操作
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),            
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=1.0)

augments = image4(image=img, mask=mask)
img_aug, mask_aug = augments['image'], augments['mask']

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(augments['image'])

plt.subplot(1, 2, 2)
plt.imshow(augments['mask'])
```

![image](https://user-images.githubusercontent.com/55370336/108735701-d2971800-756b-11eb-850f-0d6af213ffef.png)
