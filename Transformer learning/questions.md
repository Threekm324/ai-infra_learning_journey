(1)sigmoid & RULE间的优劣比较

(2)有关“基于transformer进行的音频转文本究竟是如何实现的”的疑问
主要疑问点：假如基于卷积神经网络的“分类”功能以实现“语音转文字”，则类比在28*28的像素网格图上根据不同像素块的灰度值推断0到9之间的数字案例；常用的波形图纵轴显示的为声音的强度，则对于相同波形进行分类可能会出现多种歧义（只要声音大小的变化在同一时间段内相同即可，无法对不同的音和字进行区分）

<img width="586" height="640" alt="image" src="https://github.com/user-attachments/assets/86ef6788-4886-45df-bcc5-9bb40a412b25" />

https://www.doubao.com/thread/xGqfQ7cGlOSsKbEiT


(4)有关transformer实现词语预测，其最后一步流程的问题
主要疑问点：为什么在预测时只采用上下文中的最后一个向量？而不像之前神经网络那样每个样本点都要输入，用来进行反向传播，优化权重矩阵？
解答：
<img width="743" height="591" alt="image" src="https://github.com/user-attachments/assets/aacfac07-d3e2-4dd4-93af-055f162d3a7e" />
<img width="758" height="551" alt="image" src="https://github.com/user-attachments/assets/e93ae216-5b34-44ef-9d3e-861fcecbde9f" />
<img width="737" height="307" alt="image" src="https://github.com/user-attachments/assets/77479c0c-7602-4627-a672-0ee438ba678a" />
<img width="750" height="383" alt="image" src="https://github.com/user-attachments/assets/02f03642-d2d5-4b8b-be60-779d9847ae22" />


总结：

--首先明确此处的目的为“预测下一个词”而非“进一步优化这个transformer的功能”

上下文矩阵————本质为原本的token-词汇含义对应矩阵，经过注意力模块后矩阵中的每个向量都会跟前面“上文”的向量产生联系，含义被丰富/细化、在多维空间中点的位置被拖拽，上下文矩阵中的向量具有严格的位置先后关系，后面的向量由于经过注意力模块可以认为已经包含前面向量的信息；同时一个输入对应一个输出，因此只取用上下文矩阵中的最后一个向量。
