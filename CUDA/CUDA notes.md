11.13\
（1）what is CUDA & function；\
CUDA（Compute Unified Device Architecture）是 NVIDIA 推出的并行计算平台和编程模型\
让开发者能利用 GPU（显卡）进行通用计算，而不只用于图形渲染。\
<img width="741" height="327" alt="image" src="https://github.com/user-attachments/assets/43c9aeba-9b59-4716-8075-c1adfdeddf26" />\
<img width="737" height="553" alt="image" src="https://github.com/user-attachments/assets/6b4b4bb6-8ee2-4462-9eac-07dfdef747cc" />\
关键作用：能够通过gpu实现大量并行数据的计算，大幅提高算力，从而满足深度神经网络的计算需求。\


CPU与GPU的设计目标差异：\
CPU-多功能性；GPU-并行处理与极速运行（gpu的核通常远远多于CPU）\
<img width="1280" height="782" alt="image" src="https://github.com/user-attachments/assets/b987342b-1b78-4f8f-beb4-c20967ee00ab" />

CUDA工作原理:\
1.编写一个名为CUDA的内核函数，其在GPU上运行；\
2.将一些数据从主内存复制到GPU内存；\
3.CPU发出指令指示GPU执行CUDA这一内核函数；\
？？4.内核函数被成千上万个线程同时执行，线程被分组（block），代码在若干个“thread block”中运行，每个block中包含有许多线程，所有的block又组成一个多维网络；\
5.最后，GPU的最终结果会被复制回主内存；\
<img width="496" height="274" alt="image" src="https://github.com/user-attachments/assets/c9b18ab6-9ca2-4454-8085-3cc1838004fe" />
<img width="733" height="428" alt="image" src="https://github.com/user-attachments/assets/3b1a3365-dd77-4c8a-b8ac-15ce142f52ed" />
？所谓多维网络是：\
形象比喻：一个巨型小区住着 10 万人，光靠门牌号肯定乱套。所以小区这样编址： 你的地址 = 12 栋（楼栋号） + 305 室（门牌号）\
    班级（block）太多，也得给班级编号。网格就是“班级的座位表”。\
<img width="715" height="317" alt="image" src="https://github.com/user-attachments/assets/dcd561d1-1fe8-4171-b68c-f263fd26ce60" />\
完整地址 = blockIdx（楼栋号） + threadIdx（门牌号）\
           └── 网格负责的 ──┘   └── 块负责的 ──┘\
\
<img width="627" height="351" alt="image" src="https://github.com/user-attachments/assets/7849816b-48bb-420e-87b8-a06a21919ff0" />\
CUDA实际代码通常用C++编写
