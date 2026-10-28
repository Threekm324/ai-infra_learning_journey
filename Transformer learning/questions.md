(1)sigmoid & RULE间的优劣比较

(2)有关“基于transformer进行的音频转文本究竟是如何实现的”的疑问
主要疑问点：假如基于卷积神经网络的“分类”功能以实现“语音转文字”，则类比在28*28的像素网格图上根据不同像素块的灰度值推断0到9之间的数字案例；常用的波形图纵轴显示的为声音的强度，则对于相同波形进行分类可能会出现多种歧义（只要声音大小的变化在同一时间段内相同即可，无法对不同的音和字进行区分）

<img width="586" height="640" alt="image" src="https://github.com/user-attachments/assets/86ef6788-4886-45df-bcc5-9bb40a412b25" />

https://www.doubao.com/thread/xGqfQ7cGlOSsKbEiT
