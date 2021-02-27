## 4 评价函数与损失函数

### 4.1 TP TN FP FN

在讲解语义分割中常用的评价函数和损失函数之前，先补充一**TP(真正例 true positive) TN(真反例 true negative) FP(假正例 false positive) FN(假反例 false negative)**的知识。在分类问题中，我们经常看到上述的表述方式，以二分类为例，我们可以将所有的样本预测结果分成TP、TN、 FP、FN四类，并且每一类含有的样本数量之和为总样本数量，即TP+FP+FN+TN=总样本数量。其混淆矩阵如下：

![image-20210115164322758](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/Task4%EF%BC%9A%E8%AF%84%E4%BB%B7%E5%87%BD%E6%95%B0%E4%B8%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0_image/%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png)

上述的概念都是通过以预测结果的视角定义的，可以依据下面方式理解：

- 预测结果中的正例 → 在实际中是正例 → 的所有样本被称为真正例（TP）<预测正确>

- 预测结果中的正例 → 在实际中是反例 → 的所有样本被称为假正例（FP）<预测错误>

- 预测结果中的反例 → 在实际中是正例 → 的所有样本被称为假反例（FN）<预测错误>

- 预测结果中的反例 → 在实际中是反例 → 的所有样本被称为真反例（TN）<预测正确>

这里就不得不提及精确率（precision）和召回率（recall）：

![image](https://user-images.githubusercontent.com/55370336/109387026-b0b9de80-7939-11eb-8134-c2609767d2c0.png)

Precision代表了预测的正例中真正的正例所占比例；Recall代表了真正的正例中被正确预测出来的比例。

转移到语义分割任务中来，我们可以将语义分割看作是对每一个图像像素的的分类问题。根据混淆矩阵中的定义，我们亦可以将特定像素所属的集合或区域划分成TP、TN、 FP、FN四类。

![](https://github.com/datawhalechina/team-learning-cv/raw/master/AerialImageSegmentation/img/Task4%EF%BC%9A%E8%AF%84%E4%BB%B7%E5%87%BD%E6%95%B0%E4%B8%8E%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0_image/%E5%88%86%E5%89%B2%E5%AF%B9%E7%85%A7.png)

以上面的图片为例，图中左子图中的人物区域（黄色像素集合）是我们**真实标注的前景信息（target）**，其他区域（紫色像素集合）为背景信息。当经过预测之后，我们会得到的一张预测结果，图中右子图中的黄色像素为**预测的前景（prediction）**，紫色像素为预测的背景区域。此时，我们便能够将预测结果分成4个部分：

- 预测结果中的黄色无线区域 → 真实的前景 → 的所有像素集合被称为真正例（TP）<预测正确>

- 预测结果中的蓝色斜线区域 → 真实的背景 → 的所有像素集合被称为假正例（FP）<预测错误>

- 预测结果中的红色斜线区域 → 真实的前景 → 的所有像素集合被称为假反例（FN）<预测错误>

- 预测结果中的白色斜线区域 → 真实的背景 → 的所有像素集合被称为真反例（TN）<预测正确>

### 4.2 Dice评价指标

**Dice系数**

Dice系数（Dice coefficient）是常见的评价分割效果的方法之一，同样也可以改写成损失函数用来度量prediction和target之间的距离。Dice系数定义如下：

![image](https://user-images.githubusercontent.com/55370336/109387059-f4ace380-7939-11eb-8bd4-a27f4078bf2c.png)

式中：T表示真实前景（target），$P$表示预测前景（prediction）。Dice系数取值范围为`[0,1]`，其中值为1时代表预测与真实完全一致。仔细观察，Dice系数与分类评价指标中的F1 score很相似：

![image](https://user-images.githubusercontent.com/55370336/109387096-3473cb00-793a-11eb-8298-9aef3fdfad71.png)

所以，Dice系数不仅在直观上体现了target与prediction的相似程度，同时其本质上还隐含了精确率和召回率两个重要指标。

计算Dice时，将T ∩ P近似为prediction与target对应元素相乘再相加的结果。|T| 和|P|的计算直接进行简单的元素求和（也有一些做法是取平方求和），如下示例：

![image](https://user-images.githubusercontent.com/55370336/109387215-0e025f80-793b-11eb-857c-5e0ec8bb2485.png)

**Dice Loss**

Dice Loss是在[V-net](https://arxiv.org/abs/1606.04797)模型中被提出应用的，是通过Dice系数转变而来，其实为了能够实现最小化的损失函数，以方便模型训练，以1 - Dice的形式作为损失函数：

![image](https://user-images.githubusercontent.com/55370336/109387261-4f930a80-793b-11eb-9cee-0d11dac54d9f.png)

在一些场合还可以添加上**Laplace smoothing**减少过拟合：

![image](https://user-images.githubusercontent.com/55370336/109387264-5a4d9f80-793b-11eb-9820-2bb8a0096fd3.png)

**代码实现**

```python
import numpy as np

def dice(output, target):
    '''计算Dice系数'''
    smooth = 1e-6 # 避免0为除数
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

# 生成随机两个矩阵测试
target = np.random.randint(0, 2, (3, 3))
output = np.random.randint(0, 2, (3, 3))

d = dice(output, target)
# ----------------------------
target = array([[1, 0, 0],
       			[0, 1, 1],
			    [0, 0, 1]])
output = array([[1, 0, 1],
       			[0, 1, 0],
       			[0, 0, 0]])
d = 0.5714286326530524
```


