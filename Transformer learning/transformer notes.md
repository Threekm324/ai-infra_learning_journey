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



day 3 10.26
(3B1B深度学习4 有关反向传播算法的微积分解释)

下图举例为仅包含一个训练样本时 代价对wL的导数
<img width="631" height="356" alt="image" src="https://github.com/user-attachments/assets/49da0ba1-64f2-41ff-a042-6d9d964da609" />
链式法则

<img width="633" height="338" alt="image" src="https://github.com/user-attachments/assets/9ab309e4-4009-439d-91e9-14637679f6fd" />
含有多个样本时的情况————求平均值   跟总的代价函数是所有训练样本代价的总平均是一个道理

zL关于wL的偏导体现了：权重的改变量对最后一层的影响，由上一层神经元的激活值决定

<img width="626" height="338" alt="image" src="https://github.com/user-attachments/assets/4ae023d8-d8ad-4c15-88b5-df23e9a9b847" />
负梯度 代价函数涉及每一个权重和偏置

<img width="628" height="353" alt="image" src="https://github.com/user-attachments/assets/835d81c3-82b0-4c45-866d-9cd35d3604bf" />
代价函数对于上一层激活值的敏感度 虽然无法直接改变上一层的激活值，但由链式法则可以看出实际上是与权重直接相关

<img width="626" height="363" alt="image" src="https://github.com/user-attachments/assets/ef57d4e7-d7f8-43c2-93a4-5e6937d84333" />
单层多神经元时 几乎和单神经元时没区别。最主要的区别在于总代价函数对上一层的激活值求偏导，此时上一层的a激活值可能会通过j种路径对下一层产生影响（设下一层有j个神经元），因此要将影响相加。

<img width="632" height="368" alt="image" src="https://github.com/user-attachments/assets/b3467e19-950a-4269-a208-3d00d7f28b0e" />


day 4 10.28
(3B1B 深度学习系列5 what is GPT? 直观理解transformer)

GPT :Genertive Pre-trained Transformer 注意此处的generative指的是生成文本
<img width="642" height="377" alt="image" src="https://github.com/user-attachments/assets/f10ada2c-9f70-4086-b556-bc0e4cfd33c0" />
许多模型都是基于transformer所构建的 (voice-to-text / text-to-voice / text-to-image...)

构建LLM的transformer的功能：本质仍为概率推断。给一个初始片段，transformer基于上下文对下一个可能的词进行推理，然后加上这个词至文本末尾获得新的文本，再重复推理预测过程。    即  “重复预测与抽样”
<img width="631" height="362" alt="image" src="https://github.com/user-attachments/assets/31f2d56b-f3ac-4ab7-8874-160cc37f857a" />

--transformer中的数据流动方式（当LLM生成特定单词时背后的行为）
1. 拆分输入内容成若干token （可为词、词组、小块图像、音频片段等）。token——一个向量，编码片段含义。（相当于根据含义的不同给token分配空间坐标，含义越接近的坐标越近。）

2. “注意力模块”（attention block）
   向量被传到注意力模块后使得彼此之间能够相互“交流”，并且更新含义向量（实现基于上下文），避免孤立翻译token导致语句不通
注意力模块的作用：
<img width="637" height="371" alt="image" src="https://github.com/user-attachments/assets/61a3afba-9af0-40b2-9851-046c5e859bca" />
<img width="638" height="348" alt="image" src="https://github.com/user-attachments/assets/c1dcb00c-34d8-41a9-9413-ac3c0eaf97c3" />

3.Multilayer Perception（多层感知器）/Feed-Forward layer（前馈层）
    向量间不再交流，而是“并行经历同一处理”

<img width="639" height="368" alt="image" src="https://github.com/user-attachments/assets/6007f97b-4fdc-4a71-aca7-f641531e7140" />
机器学习 “不要试图在代码中 明确定义如何实现一个任务”（ai发展初期的行为）
而应该去构建一个“具有可调参数的灵活架构”，根据大量的训练样本以进行调试和模拟，不断提高模型能力。

但问题是：并不是一味地直接扩大模型参数量就行了，可能会出现过拟合或者完全训练不出来的情况
规避方法？反向传播算法

反向传播方法对模型结构有要求：
<img width="633" height="380" alt="image" src="https://github.com/user-attachments/assets/4267ef4b-7a2d-43cc-94a9-706ea592b358" />

数组的维数没有限制

<img width="629" height="367" alt="image" src="https://github.com/user-attachments/assets/693ac017-4ec5-4140-a256-0d96def4ab47" />
<img width="635" height="362" alt="image" src="https://github.com/user-attachments/assets/e0353207-8b99-48c7-8501-4a56e57b6e66" />

注意权重矩阵中的每一个权重值都是“可调的”，作为变换工具

<img width="635" height="374" alt="image" src="https://github.com/user-attachments/assets/eb01831c-7be3-44b3-acf1-28033e274090" />

权重———“大脑”，决定模型的行为模式；
待处理数据————作为一次次的输入数据
<img width="635" height="366" alt="image" src="https://github.com/user-attachments/assets/3da80a9c-0f6e-4217-bab3-99314612b3f9" />



————Embedding matrix（基于预设词汇库）————
模型中的第一组权重
将输入文本分成若干token，再转为向量时所需用到的矩阵————每个单词对应一组列向量

Q？嵌入矩阵中是一个token对应一个列向量还是就是一个词对应一个列向量，token的包含范围会随着训练样本的不断增加而改变吗？比如说原本1个token对应一个单词，后来1个token对应一个词组（缓存命中？合并同类项，减少重复建设花费？）

<img width="635" height="380" alt="image" src="https://github.com/user-attachments/assets/fd9b44ea-8458-4784-bd92-06cd6f263a75" />
<img width="632" height="358" alt="image" src="https://github.com/user-attachments/assets/53b0257e-820c-4bd6-9d6b-9a5a627f820f" />

嵌入的向量在空间中的方向 往往具有某种语义意义

<img width="638" height="374" alt="image" src="https://github.com/user-attachments/assets/d0adf613-c251-40c6-a6eb-f3738e180561" />

<img width="641" height="365" alt="image" src="https://github.com/user-attachments/assets/f2c01c0c-f947-447a-8115-8e68eac972e6" />

多维空间中点的相对距离 实际就为各token语义的近似程度

<img width="629" height="356" alt="image" src="https://github.com/user-attachments/assets/5163bf2b-f781-4c1c-9dc8-0cf3faf5d225" />
<img width="634" height="362" alt="image" src="https://github.com/user-attachments/assets/94c59757-cc67-4484-b4ad-eff560ba987a" />

“相对应的位置中容易发现相对应的语义”  eg.此处德国与意大利相对应 希特勒与墨索里尼相对应，两个向量差几乎相等

————两向量的点积 可以被看作是衡量二者对齐程度的一种方法
（因为向量的方向具有语义意义，所以粗略的讲可以根据点积的正负以及绝对值大小来判断语义近似程度；eg.见下图

<img width="634" height="360" alt="image" src="https://github.com/user-attachments/assets/a182c93a-425a-4cb3-a49c-f8a60c2944c7" />

**对transformer的意义
嵌入空间的向量，不仅代表了token本身的语义，还具有空间位置信息
更重要的是这些向量能“结合上下文语境”


