模型集成是数据竞赛中很常用的一种方法。通常用于最后用来冲排名。实际上深度神经网络尤其是CNN对模型进行拟合可以用盲人摸象这个故事来很好的形容。CNN的一个很大的特性在于，它可以通过某个局部特征来推出总体
结果，由低层次的特征经过组合，组成高层次的特征，并且得到不同特征之间的空间相关性。

![](https://pic4.zhimg.com/80/v2-8555de443211e31f6e3967fe0fab83b3_720w.png)

* CNN抓住此共性的手段主要有四个：局部连接／权值共享／池化操作／多层次结构。

* 局部连接使网络可以提取数据的局部特征；权值共享大大降低了网络的训练难度，一个Filter只提取一个特征，在整个图片（或者语音／文本） 中进行卷积；池化操作与多层次结构一起，实现了数据的降维，将低层次的
局部特征组合成为较高层次的特征，从而对整个图片进行表示。

![](https://pic4.zhimg.com/80/v2-27961b1ce1d39d970fae7e40fd99edf3_720w.png)

那么既然是这种特性，同时还有梯度下降的一些基本原理，就必然会导致神经网络在对训练集特征进行学习的时候不可避免的陷入某种局部最优解。神经网络不可能可以按我们所想的那样学习到所有我们训练数据的所有特点，
那么在这种情况下我们的神经网络就像是摸象的那个瞎子。它只能根据自己摸到的一部分来学习大象的特征，因此在这个时候，多模型集成就可以类比于好多个瞎子同时摸象，然后根据他们各自获得的特征组合得到一个更加
具有鲁棒性的结果。

## 模型集成

### 6.1 集成学习方法

在机器学习中的集成学习可以在一定程度上提高预测精度，常见的集成学习方法有Stacking、Bagging和Boosting，同时这些集成学习方法与具体验证集划分联系紧密。

由于深度学习模型一般需要较长的训练周期，如果硬件设备不允许建议选取留出法，如果需要追求精度可以使用交叉验证的方法。

下面假设构建了10折交叉验证，训练得到10个语义分割模型。
![IMG](img/交叉验证.png)

那么在10个CNN模型可以使用如下方式进行集成：

- 对预测的结果的概率值进行平均，然后解码为具体字符；
- 对预测的字符进行投票，得到最终字符；

### 6.2 深度学习中的集成学习

此外在深度学习中本身还有一些集成学习思路的做法，值得借鉴学习：          

#### 6.2.1 Dropout

Dropout可以作为训练深度神经网络的一种技巧。在每个训练批次中，通过随机让一部分的节点停止工作。同时在预测的过程中让所有的节点都其作用。
![IMG](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/Droopout.png)

Dropout经常出现在在先有的CNN网络中，可以有效的缓解模型过拟合的情况，也可以在预测时增加模型的精度。

#### 6.2.2 TTA
测试集数据扩增（Test Time Augmentation，简称TTA）也是常用的集成学习技巧，数据扩增不仅可以在训练时候用，而且可以同样在预测时候进行数据扩增，对同一个样本预测三次，然后对三次结果进行平均。

![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/tta.png)

```python
for idx, name in enumerate(tqdm_notebook(glob.glob('./test_mask/*.png')[:])):
    image = cv2.imread(name)
    image = trfm(image)
    with torch.no_grad():
        image = image.to(DEVICE)[None]
        score1 = model(image).cpu().numpy()
        
        score2 = model(torch.flip(image, [0, 3]))
        score2 = torch.flip(score2, [3, 0]).cpu().numpy()

        score3 = model(torch.flip(image, [0, 2]))
        score3 = torch.flip(score3, [2, 0]).cpu().numpy()
        
        score = (score1 + score2 + score3) / 3.0
        score_sigmoid = score[0].argmax(0) + 1
```

#### 6.2.3 Snapshot

本章的开头已经提到，假设我们训练了10个CNN则可以将多个模型的预测结果进行平均。但是加入只训练了一个CNN模型，如何做模型集成呢?

在论文Snapshot Ensembles中，作者提出使用cyclical learning rate进行训练模型，并保存精度比较好的一些checkopint，最后将多个checkpoint进行模型集成。
![IMG](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/Snapshot.png)
          
由于在cyclical learning rate中学习率的变化有周期性变大和减少的行为，因此CNN模型很有可能在跳出局部最优进入另一个局部最优。在Snapshot论文中作者通过使用表明，此种方法可以在一定程度上提高模型精度，但需要更长的训练时间。
![IMG]https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/%E5%AF%B9%E6%AF%94.png)    

### 6.3 小结

- 集成学习只能在一定程度上提高精度，并需要耗费较大的训练时间，因此建议先使用提高单个模型的精度，再考虑集成学习过程；
- 具体的集成学习方法需要与验证集划分方法结合，Dropout和TTA在所有场景有可以起作用。
- 最后如果成绩还不错的话，可以尝试在测试集上训练来获得一个更好的结果。
