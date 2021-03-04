## 5: 模型评估和训练

一个成熟合格的深度学习训练流程至少具备以下功能：
- 在训练集上进行训练，并在验证集上进行验证；
- 模型可以保存最优的权重，并读取权重；

实际上 [segmentation_models.pytorch](https://smp.readthedocs.io/en/latest/models.html#id9) 这个库已经给了我们绝大部分我们需要用到的模型，很多时候我们只需要在里面选择常用的一些网络模型
进行训练就好。目前我用的效果最好的不是paper最新的的DeepLabv3Plus，而是网络比较简单的PSPNet。所以这个具体区别还是得自己尝试。但是总体来说做完数据扩充之后选择相对更大更复杂一点的网络总体性能要好
不少。

### 5.1 构造验证集

在机器学习模型（特别是深度学习模型）的训练过程中，模型是非常容易过拟合的。深度学习模型在不断的训练过程中训练误差会逐渐降低，但测试误差的走势则不一定。

在模型的训练过程中，模型只能利用训练数据来进行训练，模型并不能接触到测试集上的样本。因此模型如果将训练集学的过好，模型就会记住训练样本的细节，导致模型在测试集的泛化效果较差，
这种现象称为过拟合（Overfitting）。与过拟合相对应的是欠拟合（Underfitting），即模型在训练集上的拟合效果较差。

![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/loss.png)

如图所示：随着模型复杂度和模型训练轮数的增加，CNN模型在训练集上的误差会降低，但在测试集上的误差会逐渐降低，然后逐渐升高，而我们为了追求的是模型在测试集上的精度越高越好。

导致模型过拟合的情况有很多种原因，其中最为常见的情况是模型复杂度（Model Complexity ）太高，导致模型学习到了训练数据的方方面面，学习到了一些细枝末节的规律。

解决上述问题最好的解决方法：构建一个与测试集尽可能分布一致的样本集（可称为验证集），在训练过程中不断验证模型在验证集上的精度，并以此控制模型的训练。

当然，不是所有的在测试集上的泛化效果较差的网络模型都发生了过拟合，而是有可能本身模型的复杂度就很低，导致在训练集上都没能很好的拟合数据，这个时候我们就需要增加模型的复杂度。

![](https://nvsyashwanth.github.io/machinelearningmaster/assets/images/bias_variance.jpg)

训练集误差和交叉验证集误差近似时：high bias/underfitting 交叉验证集误差远大于训练集误差时：high varience/overfitting。

![](http://www.ai-start.com/ml2014/images/25597f0f88208a7e74a3ca028e971852.png)

回到本题中，由于比赛给出训练集有30000张，同时考虑到计算机视觉的特殊性而采取的数据增强方法，以及大家本地的硬件限制~~（基本上p100或者1080以下的卡都没啥机会过拟合吧···你跑到过拟合得训练到天荒地老了已经）~~我们在这题中其实过拟合的机会不大。在我尝试过的这几种模型中，encoder采用ResNet101和ResNet152训练30~50轮（分辨率：512，batch：30以上）的效果实际上是比较理想的。

在给定赛题后，赛题方会给定训练集和测试集两部分数据。参赛者需要在训练集上面构建模型，并在测试集上面验证模型的泛化能力。因此参赛者可以通过提交模型对测试集的预测结果，来验证自己模型的泛化能力。同时参赛方也会限制一些提交的次数限制，以此避免参赛选手“刷分”。

在一般情况下，参赛选手也可以自己在本地划分出一个验证集出来，进行本地验证。训练集、验证集和测试集分别有不同的作用：
- #### 训练集（Train Set）：模型用于训练和调整模型参数；
- #### 验证集（Validation Set）：用来验证模型精度和调整模型超参数；
- #### 测试集（Test Set）：验证模型的泛化能力。

因为训练集和验证集是分开的，所以模型在验证集上面的精度在一定程度上可以反映模型的泛化能力。在划分验证集的时候，需要注意验证集的分布应该与测试集尽量保持一致，不然模型在验证集上的精度就失去了指导意义。 

既然验证集这么重要，那么如何划分本地验证集呢。在一些比赛中，赛题方会给定验证集；如果赛题方没有给定验证集，那么参赛选手就需要从训练集中拆分一部分得到验证集。验证集的划分有如下几种方式：

![IMG](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/%E9%AA%8C%E8%AF%81%E9%9B%86%E6%9E%84%E9%80%A0.png)   

#### 留出法（Hold-Out）
直接将训练集划分成两部分，新的训练集和验证集。这种划分方式的优点是最为直接简单；缺点是只得到了一份验证集，有可能导致模型在验证集上过拟合。留出法应用场景是数据量比较大的情况。 

#### 交叉验证法（Cross Validation，CV）
将训练集划分成K份，将其中的K-1份作为训练集，剩余的1份作为验证集，循环K训练。这种划分方式是所有的训练集都是验证集，最终模型验证精度是K份平均得到。这种方式的优点是验证集精度比较可靠，训练K次可以得到K个有多样性差异的模型；CV验证的缺点是需要训练K次，不适合数据量很大的情况。 

~~还有对自己代码能力比较自信的情况下，我同k-flod十次有9次报错~~

#### 自助采样法（BootStrap）
通过有放回的采样方式得到新的训练集和验证集，每次的训练集和验证集都是有区别的。这种划分方式一般适用于数据量较小的情况。

~~稍微有点中学数学基础就能很轻易地推导出来会有1/e的数据集是不会被采样到的~~，实际上这种方法也适合在只有一部分数据的情况下划分training set，validation set和test set。

在本次赛题中数据量相对较大，因此我们可以采用最简单的留出法。大概保留2500~3000张图片作为vali_set就好。   

当然这些划分方法是从数据划分方式的角度来讲的，在现有的数据比赛中一般采用的划分方法是留出法和交叉验证法。如果数据量比较大，留出法还是比较合适的。当然任何的验证集的划分得到的验证集都是要保证训练集-验证集-测试集的分布是一致的，所以如果不管划分何种的划分方式都是需要注意的。

这里的分布一般指的是与标签相关的统计分布，比如在分类任务中“分布”指的是标签的类别分布，训练集-验证集-测试集的类别分布情况应该大体一致；如果标签是带有时序信息，则验证集和测试集的时间间隔应该保持一致。

### 5.2 模型训练与验证
在本节我们目标使用Pytorch来完成CNN的训练和验证过程，CNN网络结构与之前的章节中保持一致。我们需要完成的逻辑结构如下：   
- 构造训练集和验证集；
- 每轮进行训练和验证，并根据最优验证集精度保存模型。 

```python
train_loader = torch.utils.data.DataLoader(
  train_dataset,
  batch_size=32, #自己看着自己显卡办
  shuffle=True, 
  num_workers=10, 
)

val_loader = torch.utils.data.DataLoader(
  val_dataset,
  batch_size=32, #疯狂凡尔赛我的v100和四张3090
  shuffle=False, 
  num_workers=10, 
)

model = Model1()
criterion = nn.CrossEntropyLoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), 0.001)
best_loss = 0.106 #还是自己看着办，当然也可以每一轮都验证然后保留
for epoch in range(20):
print('Epoch: ', epoch)

train(train_loader, model, criterion, optimizer, epoch)
val_loss = validate(val_loader, model, criterion)

# 记录下验证集精度
if val_loss < best_loss: #这个可以写进for循环里面，每一次训练都查看是否需要保留
    best_loss = val_loss #其实我觉得每次都查看会好一点，你可以随时停掉训练
    torch.save(model.state_dict(), './model.pt')
```

其中每个Epoch的训练代码如下： 
```python
def train(train_loader, model, criterion, optimizer, epoch):
    # 切换模型为训练模式
    model.train()

    for i, (input, target) in enumerate(train_loader):
      # 正向传播
      # 计算损失
      # 反向传播
      pass
```

其中每个Epoch的验证代码如下：
```python 
def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # 正向传播
            # 计算损失
            pass
```

### 5.3 模型保存与加载
在Pytorch中模型的保存和加载非常简单，比较常见的做法是保存和加载模型参数： 
``` torch.save(model_object.state_dict(), 'model.pt') ``` 

```model.load_state_dict(torch.load(' model.pt')) ``` 

### 5.4 模型调参流程 
深度学习原理少但实践性非常强，基本上很多的模型的验证只能通过训练来完成。同时深度学习有众多的网络结构和超参数，因此需要反复尝试。训练深度学习模型需要GPU的硬件支持，也需要较多的训练时间，如何有效的训练深度学习模型逐渐成为了一门学问。

深度学习有众多的训练技巧，比较推荐的阅读链接有：   
- http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
- http://karpathy.github.io/2019/04/25/recipe/

本节挑选了常见的一些技巧来讲解，并针对本次赛题进行具体分析。与传统的机器学习模型不同，深度学习模型的精度与模型的复杂度、数据量、正则化、数据扩增等因素直接相关。所以当深度学习模型处于不同的阶段（欠拟合、过拟合和完美拟合）的情况下，大家可以知道可以什么角度来继续优化模型。

在参加本次比赛的过程中，建议大家以如下逻辑完成：  

- 初步构建简单的CNN模型，不用特别复杂，跑通训练、验证和预测的流程；
- 简单CNN模型的损失会比较大，尝试增加模型复杂度，并观察验证集精度； 
- 在增加模型复杂度的同时增加数据扩增方法，直至验证集精度不变。

![IMG](https://github.com/datawhalechina/team-learning-cv/blob/master/AerialImageSegmentation/img/%E8%B0%83%E5%8F%82%E6%B5%81%E7%A8%8B.png)

