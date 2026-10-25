day 1 10.24

初步尝试使用GitHub

transformer part

有关深度学习和神经网络的基本运作方式进行了初步了解，3B1B深度学习part1，以图像识别（28*28pixels 0-9数字图像识别）举例

卷积神经网络，上一层所有神经元的激活值会对下一层神经元的激活值产生影响，权重 偏置值 矩阵形式进行计算 sigmoid映射用于压缩加权和至0-1的范围内

<img width="2348" height="1320" alt="dd2e1ea2ba2f3967f5616c92b87882d6" src="https://github.com/user-attachments/assets/0c86b3c3-d31c-4b68-891f-2a0ac800042c" />
(有关具体计算公式）

<img width="2348" height="1320" alt="661092bb4999adaf67f532c6e630d16a" src="https://github.com/user-attachments/assets/fef2494c-d761-47ad-86cd-44fd98b149a5" />
（采用RELU）

但后面又不采用sigmoid 转用RELU（Rectified Linear unit)  原因？？
有关sigmoid与RELU差异
<img width="2459" height="598" alt="1a6767a4704eb6296c89173aff800ed3" src="https://github.com/user-attachments/assets/15d2a4d0-409f-4549-b956-84fa781a00e4" />



day 2 10.25

(3B1B 深度学习2)
关键：虽然对于神经网络已知该如何计算输出的灰度值，但对于层级之间的权重值和偏置的设置方法尚未提及

初始的权重值和偏置几乎是完全随机的，通过大量训练数据的输入(例如 给予大量手写数字样本以及对应输出值)来对权重值和偏置进行调试和优化
因此需要一个评价的函数(cost function/loss function 代价函数)，to tell the computer/network how bad it is见下图:

<img width="1280" height="800" alt="屏幕截图 2026-10-25 105800" src="https://github.com/user-attachments/assets/b34ee284-b695-4b7f-b661-8a6bc51576dc" />
<img width="638" height="380" alt="屏幕截图 2026-10-25 111740" src="https://github.com/user-attachments/assets/cbd29723-0764-4c98-877d-932f6ec879ff" />


通过输出的这个数值的平均值 （Empirical Risk平均代价） 来表示和评估当前这个神经网络的权重和偏置值的优秀程度（empirical cost越接近0则越优秀，越大则越糟糕）

评估结束后需告诉电脑应如何改变
目的：通过调整权重值和偏置值 能够取到代价函数的最小值；
取最小值的方法：函数输入空间极大（包括所有的权重值和偏置值）多元函数采用梯度运算，梯度的反方向（负梯度）就是下降速度最快的方向————梯度下降法
<img width="635" height="370" alt="image" src="https://github.com/user-attachments/assets/8c7700fa-9e34-4179-9401-850e02e40375" />

“让网络学习，实质上就是通过找到特定的权重偏置，从而让代价函数的值最小” 为实现此效果 因此采用连续的activation而非生物上的二元激活状态

<img width="631" height="387" alt="image" src="https://github.com/user-attachments/assets/3bdd7e95-30a6-4dfa-a6dc-4e5a3b779461" />

但目前这个“图像识别神经网络”的实际功能其实更倾向于“分类”而非真正的“识别”（换句话说，虽然网络可以很好的识别数字，但不能自己去写一个数字）

<img width="632" height="338" alt="image" src="https://github.com/user-attachments/assets/956c90ab-0de7-47bd-ae55-b983ce5434bd" />

“网络倾向收敛到的各个局部最小值，其实都差不多一样大”



（3B1B深度学习3 反向传播算法【BP】）
作用：用于求上面所提及的那个“能使代价函数下降最快”的 具有13000+个变量的梯度向量

另一套思路：梯度向量的每一个值 除了显示我们应该如何对权重值和偏置值应该如何进行调整之外，同时还彰显了代价函数对于每个参数有多敏感（由梯度向量中每一项的绝对值大小就可以彰显）！
<img width="629" height="364" alt="image" src="https://github.com/user-attachments/assets/9c0881aa-08ef-47b3-ab55-b9c9e5a9ec53" />

<img width="632" height="358" alt="image" src="https://github.com/user-attachments/assets/e74a4fc3-ed87-4904-8e7f-80db7e437293" />
<img width="637" height="365" alt="image" src="https://github.com/user-attachments/assets/2d12e5c1-2814-44a9-9161-428897c7e7b9" />

意即，在梯度下降时，不仅看每个参数应如何变化；还看改动哪个参数的性价比最高
<img width="635" height="351" alt="image" src="https://github.com/user-attachments/assets/54d6ac04-0691-49ef-819c-6de7cbf7d70f" />

即改权重————看激活值；改激活值————看权重

<img width="635" height="363" alt="image" src="https://github.com/user-attachments/assets/1e0d480d-482c-4e08-93ec-31e1b57574fe" />

从期待产生的改变（本层激活值）倒推上一层的激活值、权重和偏置该如何变  以此类推，不断重复循环，即反向传播

<img width="635" height="348" alt="image" src="https://github.com/user-attachments/assets/d9bb6767-5972-4c10-a96a-c33d724567a2" />
<img width="635" height="364" alt="image" src="https://github.com/user-attachments/assets/59ee2294-d286-4c0a-8473-78eb2794301a" />

<img width="633" height="351" alt="image" src="https://github.com/user-attachments/assets/59453b6a-14d7-4988-a577-930aa26d1213" />

summary：反向传播算法 算的是单个训练样本想怎样修改权重和偏置，包括上升还是下降以及对应比例。从而实现代价最小 但如果要对所有的训练样本都采用此方法再取平均值则成本过高
         从而出现了【随机梯度下降】（总样本分成若干小块，在每个小块上算梯度？）收敛到代价函数的一个局部最小值上
         
<img width="629" height="353" alt="image" src="https://github.com/user-attachments/assets/9fa5f83f-f49a-4e69-b8f0-843aeaa1d7d3" />

“所有包括神经网络在内的机器学习，要让他们工作就需要大量的样本范例。所以机器学习领域的人，最熟悉的一个难关，莫过于获取标记好的训练数据”

