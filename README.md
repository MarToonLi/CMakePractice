# Opencv的100道题目
## 1-10题的题目：
### 1-20需要引入头文件：A_1_10.h
- [通道交换](/A1.cpp "通道交换")

- [灰度化](/A2.cpp "灰度化")
- [二值化](/A3.cpp "二值化")
- [大津二值化算法](/A4.cpp "大津二值化算法")
- [RGB->HSV色域变换](/A5.cpp "RGB->HSV色域变换")
- [减色处理](/A6.cpp "减色处理")
- [平均池化](/A7.cpp "平均池化")
- [最大池化](/A8.cpp "最大池化")
- [高斯滤波](/A9.cpp "高斯滤波")
- [中值滤波](/A10.cpp "中值滤波")

## 1-10题重要算法解析：
### 一维高斯分布
<img src="/images/一维高斯分布.png" alt="GitHub set up" style="zoom:50%;" />

### 二维高斯分布
<img src="/images/二维高斯分布.png" alt="GitHub set up" style="zoom:50%;" />

### 大津二值化算法--最大类内方差
<img src="/images/最大类内方差1.png" alt="GitHub set up" style="zoom: 67%;" />
<img src="/images/最大类内方差2.png" alt="GitHub set up" style="zoom:67%;" />

> 大津二值化算法的要点：
>
> 1. 大津二值化算法的算法假设：图像灰度分布呈现双峰性，同时类间方差与图像分割的质量具有较大相关性；
>    1. 大津算法假设，如果能够找到一个阈值 t，使得分割后的两个类别（C0 和 C1）的类间方差  \sigma_B^2(t) *σ**B*2(*t*) 达到最大值，那么这个阈值 t 将能够最有效地将图像分成两部分，使得前景和背景之间的差异最大化；
>    2. 存在两个明显的峰值，一个代表前景（目标），另一个代表背景；
> 2. 在图像灰度分布呈现双峰时，为什么类间方差最大能够很好地分割图像呢？
>    1. 如果图像灰度呈现双峰，那么前景和背景的平均灰度值之间的差异较为明显，而从数学上，方差表达的是数据集中每个样本点与其均值之间的差异程度；而大津二值化算法中如果A类像素的平均灰度值与图像整体平均灰度值的差异程度，和 B类像素的平均灰度值与图像整体平均灰度值的差异程度的加和，最大，就此时的分割值能带来最好的分割效果。
> 3. ![image-20240803220436871](README.assets/image-20240803220436871.png)



### HSV色调翻转

> **HSV色调翻转：**
>
> 1. HSV图像的色调翻转是指将色调通道H，进行反转；其取值为0~180；因此翻转：180-H也是可以的；
> 2. fmod函数用于计算两个浮点数的余数；fmod(imgHSV.at<Vec3f>(y, x)[0] + 180, 360);
>
> **RGB图像转换为HSV图像：**
>
> 1. 该转换过程遵循固定的数据公式和计算流程：RGB通道归一化、先计算亮度V，再计算饱和度S，然后计算色调H；
>
> **HSV图像转换为RGB图像：**
>
> 1. 根据饱和度和亮度计算颜色的强度C、根据色调H将色彩图360度分成六个部分，每一部分对应一个RGB颜色的分量变化、根据色调分区，计算RGB分量的值；





### **减色处理**

> 1. 设定一个减色因子；原始数据遵循一个公式变换成新数据：原始像素 / n * n + n / 2; 





### 平均池化

> 1. 平均池化是将一张图像平均切分为N份网格；
> 1. 池化范围即卷积核尺寸，步长默认是与池化范围保持一致；但是也可以不一致；
> 1. 事实上，在很多经典的模型中，池化层的步长和池化范围是不一致的，这样一是为了降低空间分辨率，二是获取到更大空间区域的特征信息；



### 高斯滤波

> 1. 卷积操作，在工程实现上，卷积核是通过一行向量存在的，而卷积操作，相同读取顺序（先行后列还是先列后行）的卷积核和原始图像的卷积区域，进行乘积和相加；






## 11-10题的题目：
- [均值滤波器](/A11.cpp "均值滤波器")

> 1. 区别于池化层，它的步长是1；
> 2. 先定位到某个像素，根据卷积核的size计算对应区域的像素与卷积核的加权和。

- [Motion Filter](/A12.cpp "Motion Filter")

> 1. 与均值滤波器不同的是，后者的卷积核设计不同；
> 2. 卷积操作最需要注意的是边界的控制；
> 3. 卷积核所有权重的加和是需要等于1的；

- [MAX-MIN滤波器](/A13.cpp "MAX-MIN滤波器")

> 1. 虽然不是一个滤波核，但是**更为本质的是，卷积核代表了一种当前区域像素的计算规则**。

- [差分滤波器](/A14.cpp "差分滤波器")

> 1. 卷积核，在当前的工程环境下，似乎二维数组也是很合理的；
> 2. 差分滤波器，用于**检测覆盖区域内像素值的变化程度**；
> 3. 以水平差分滤波器为例，在当前工程实现中-1，1，0，这种搭配，配合uchar的数据范围（0~255），只能展现低像素向高像素的变化；而高像素向低像素的变化会呈现0；

- [Sobel滤波器](/A15.cpp "Sobel滤波器")

>1. 水平卷积核：不仅仅是2向-2的变化；而且还有
>
>2. ❓ 2与-2的卷积核 与 -2与2的卷积核的效果差别如何？—— 效果上还真有差别，但是具体的差别不太懂(从图像上看是黑白的差别)
>
>   <img src="README.assets/image-20240805004433394.png" alt="image-20240805004433394" style="zoom:50%;" /><img src="README.assets/image-20240805004421493.png" alt="image-20240805004421493" style="zoom:50%;" /> 
>
>   

- [Prewitt滤波器](/A16.cpp "Prewitt滤波器")

> 1. 两个卷积核
>
>    ```
>    	double kPrewittHorizontal[kSize][kSize] = { {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1} };
>    	double kPrewittVertical[kSize][kSize] = { {-1, -1, -1}, {0, 0, 0}, {1, 1, 1} };
>    ```
>
> 2. 目前来看，综合soble和当前核，一般左边或者上边的权重都是负值；

- [Laplacian滤波器](/A17.cpp "Laplacian滤波器")

> 1. double kLaplacian[kSize][kSize] = { {0, 1, 0}, {1, -4, 1}, {0, 1, 0} }; // 特殊的设计；

- [Emboss滤波器](/A18.cpp "Emboss滤波器")

> 1. double kEmboss[kSize][kSize] = { {-2, -1, 0}, {-1, 1, 1}, {0, 1, 2} };
> 2. 模仿光照方向，呈现立体感和浮雕的感觉；

- [LoG滤波器](/A19.cpp "LoG滤波器")

> 1. LoG滤波器是一种结合了高斯平滑和拉普拉斯操作的滤波器，用于检测图像中的边缘和斑点（blob）。

- [直方图(C++)](/A20.cpp "直方图")
- [直方图(python)](/A20.py "直方图")

### 11-20需要引入头文件：A_11_20.h
11-20题全是关于滤波器的题目

### 什么是边缘？
边缘一般是指图像中某一局部强度剧烈变化的区域。强度变化一般有2中情况，阶跃效应和屋顶效应。而边缘检测的任务就是找到具有阶跃变化或者屋顶变化的像素点的集合。
边缘检测基本原理：
- 对于阶跃效应：一阶微分的峰值为边缘点，二阶微分的零点为边缘点。
具体来说，从阶跃效应的那张图可以看到边缘处的斜率（一阶导）最大，所以一阶微分的峰值是边缘点，而斜率是先增大后减小的，即边缘点的二阶导为0处。
#### 阶跃效应
![GitHub set up](/images/阶跃效应.png)

- 对于屋顶效应：一阶微分的零点为边缘点，二阶微分的峰值为边缘点。
#### 屋顶效应
![GitHub set up](/images/屋顶效应.png)


## 11-20题重要算法解析：
### 均值滤波器
![GitHub set up](/images/均值滤波卷积核.png)
### 运动滤波器
![GitHub set up](/images/运动滤波器.png)
### MAX-MIN滤波器(效果图)
![GitHub set up](/images/max-min.png)
### 差分滤波器
![GitHub set up](/images/垂直滤波器.png)

![GitHub set up](/images/水平滤波器.png)

### Sobel滤波器
![GitHub set up](/images/Sobel卷积核.png)
### Prewitt滤波器
![GitHub set up](/images/prewitt算子.png)
### Laplacian滤波器
![GitHub set up](/images/4领域拉普拉斯算子.png)
### Emboss滤波器
![GitHub set up](/images/Emboss算子.png)
### LoG滤波器
![GitHub set up](/images/高斯拉普拉斯算子.png)
### 直方图
<img src="/images/直方图.png" alt="GitHub set up" style="zoom:50%;" />

## 21-30题的题目：
### 11-20需要引入头文件：A_21_30.h
- [直方图归一化](/A21.cpp "直方图归一化")

> 1. 当前的工程实现中，归一化是在调整图像像素的取值范围，从非0非255的区间调整到0~255的区间。最为关键的操作是：imgOut.at<Vec3b>(y, x)[c] = (uchar)((newEnd-newStart)/(end-start)*(val-start)+newStart);

- [方图操作](/A22.cpp "方图操作")

> 1. 具体的操作目的不清楚。应该是希望调整图像的正态分布；

- [直方图均衡化](/A23.cpp "直方图均衡化")

> 1. double hist[255];` 这里声明了一个长度为255的数组，但实际上应该是256，因为灰度级范围是从0到255，共256个可能的值。

- [伽玛校正](/A24.cpp "伽玛校正")

> 1. 图像的像素矫正，是通过统计图像整体或者局部的信息，来调整每个位置的像素值。
> 2. 伽马矫正的时候，是基于0~1区间的像素值进行的；

- [最邻近插值](/A25.cpp "最邻近插值")

> 1. 工程实现中，输出图像每个位置的像素，也都是与输入图像每个位置的像素是一一对应的

- [双线性插值](/A26.cpp "双线性插值")
- [双三次插值](/A27.cpp "双三次插值")
- [仿射变换-平行移动](/A28.cpp "仿射变换-平行移动")
- [仿射变换-放大缩小](/A29.cpp "仿射变换-放大缩小")
- [仿射变换-旋转](/A30.cpp "仿射变换-旋转")

## 21-30题重要算法解析：
### 均值方差均方差
![GitHub set up](/images/均值方差标准差.jpg)
### 伽马矫正
![GitHub set up](/images/伽马矫正.png)
### 最邻近插值
![GitHub set up](/images/最邻近插值.png)
### 双线性插值
![GitHub set up](/images/双线性插值1.png)
![GitHub set up](/images/双线性插值2.png)
![GitHub set up](/images/双线性插值4.png)
![GitHub set up](/images/双线性插值3.png)
### BiCubic函数
![GitHub set up](/images/BiCubic函数.png)
### 双三次插值
![GitHub set up](/images/双三次插值2.png)
![GitHub set up](/images/双三次插值1.png)
![GitHub set up](/images/双三次插值5.png)
### 齐次坐标系到笛卡尔坐标系转换
![GitHub set up](/images/齐次到笛卡尔坐标.png)
### 放射变换
![GitHub set up](/images/仿射变换1.png)
![GitHub set up](/images/仿射变换2.png)
![GitHub set up](/images/仿射变换3.png)
#### 平移量为0的仿射变换
![GitHub set up](/images/平移量为0的仿射变换.png)  
![GitHub set up](/images/仿射变换4.png)  
![GitHub set up](/images/放射变换5.png)
## 31-40题的题目：
### 31-40需要引入头文件：A_31_40.h
- [仿射变换（Afine Transformations）——倾斜](/A31.cpp "仿射变换（Afine Transformations）——倾斜")
- [傅立叶变换](/A32.cpp "傅立叶变换")
- [傅立叶变换——低通滤波](/A33.cpp "傅立叶变换——低通滤波")
- [傅立叶变换——高通滤波](/A34.cpp "傅立叶变换——高通滤波")
- [傅立叶变换——带通滤波](/A35.cpp "傅立叶变换——带通滤波")
- [霍夫变换：直线检测-霍夫逆变换](/A36.cpp "霍夫变换：直线检测-霍夫逆变换")
- [离散余弦变换](/A37.cpp "离散余弦变换")
- [离散余弦变换+量化](/A38.cpp "离散余弦变换+量化")
- [YCbCr 色彩空间](/A39.cpp "YCbCr 色彩空间")
- [YCbCr+离散余弦变换+量化](/A40.cpp "YCbCr+离散余弦变换+量化")

## 31-40题重要算法解析：
### 傅里叶级数
![GitHub set up](/images/傅里叶级数1.png)
![GitHub set up](/images/傅里叶级数2.png)
### 欧拉公式
![GitHub set up](/images/欧拉公式.png)
### 傅里叶变换
![GitHub set up](/images/傅里叶变换2.png)
![GitHub set up](/images/傅里叶变换1.png)
### 二维离散余弦变换
![GitHub set up](/images/二维离散余弦变换1.png)
### 均方误差
![GitHub set up](/images/均方误差.png)
### 均方误差
![GitHub set up](/images/均方误差.png)
### YCbCr色彩空间
![GitHub set up](/images/YCbCr_1.png)
![GitHub set up](/images/YCbCr_2.png)
![GitHub set up](/images/YCbCr_3.png)

## 41-50题的题目：
### 41-50需要引入头文件：A_41_50.h
- [Canny边缘检测：边缘强度](/A41.cpp "Canny边缘检测：边缘强度")
- [Canny边缘检测：边缘细化](/A42.cpp "Canny边缘检测：边缘细化")
- [Canny 边缘检测：滞后阈值](/A43.cpp "Canny 边缘检测：滞后阈值")
- [霍夫变换：直线检测](/A44.cpp "霍夫变换：直线检测")
- [霍夫变换：直线检测-NMS](/A45.cpp "霍夫变换：直线检测-NMS")
- [霍夫变换：直线检测-霍夫逆变换](/A46.cpp "霍夫变换：直线检测-霍夫逆变换")
- [膨胀（Dilate）](/A47.cpp "膨胀（Dilate）")
- [腐蚀（Erode）](/A48.cpp "腐蚀（Erode）")
- [开运算（Opening Operation）](/A49.cpp "开运算（Opening Operation）")
- [闭运算（Closing Operation）](/A50.cpp "闭运算（Closing Operation）")
## 41-50题重要算法解析：
### Canny边缘检测-高斯核
![GitHub set up](/images/canny1.png)
### Canny边缘检测-Sobel核（垂直和水平）
![GitHub set up](/images/canny2.png)
### Canny边缘检测-求梯度
![GitHub set up](/images/canny3.png)
### 霍夫变换
#### 笛卡尔坐标系下直线
![GitHub set up](/images/霍夫变换1.png)  
#### 霍夫变换下坐标方程
![GitHub set up](/images/霍夫变换2.png)  
#### 笛卡尔坐标系->霍夫坐标系（直线）
![GitHub set up](/images/霍夫变换3.png)  
#### 笛卡尔坐标系->霍夫坐标系（点）
![GitHub set up](/images/霍夫变换4.png)  
#### 笛卡尔坐标系->霍夫坐标系（两点）
![GitHub set up](/images/霍夫变换5.png)  
#### 笛卡尔坐标系->霍夫坐标系（三点）
![GitHub set up](/images/霍夫变换6.png)  
#### 笛卡尔坐标系->霍夫坐标系（多点）
![GitHub set up](/images/霍夫变换7.png)  
#### 通过霍夫坐标系下找最好共线点
![GitHub set up](/images/霍夫变换8.png)  
#### 极坐标系
![GitHub set up](/images/霍夫变换9.png)  
#### 极坐标系->霍夫坐标系
![GitHub set up](/images/霍夫变换10.png)  
#### 伪代码
![GitHub set up](/images/霍夫变换11.png)  

## 51-60题的题目：
### 51-60题需要引入头文件：A_51_60.h
- [形态学梯度（Morphology Gradient）](/A51.cpp "形态学梯度（Morphology Gradient）")
- [顶帽（Top Hat）](/A52.cpp "顶帽（Top Hat）")
- [黑帽（Black Hat）](/A53.cpp "黑帽（Black Hat）")
- [误差平方和](/A54.cpp "误差平方和")
- [绝对值差和](/A55.cpp "绝对值差和")
- [归一化交叉相关](/A56.cpp "归一化交叉相关")
- [零均值归一化交叉相关](/A57.cpp "零均值归一化交叉相关")
- [4邻域连通域标记](/A58.cpp "4邻域连通域标记")
- [8邻域连通域标记](/A59.cpp "8邻域连通域标记")
- [透明混合（Alpha Blending）(py)](/A60.py "透明混合（Alpha Blending）")
## 51-60题重要算法解析：
### 形态学梯度
![GitHub set up](/images/形态学梯度.png)
### 模式匹配-误差平方和(SSD)
![GitHub set up](/images/误差平方和(SSD).png)

### 模式匹配-绝对值差和
![GitHub set up](/images/绝对值差和.png)
### 模式匹配-归一化交叉相关
![GitHub set up](/images/归一化交叉相关.png)
### 模式匹配-零均归一化
![GitHub set up](/images/零均归一化.png)
### 模式匹配-ncc zncc
![GitHub set up](/images/ncc_zncc.png)

## 61-70题的题目：
### 61-70需要引入头文件：A_61_70.h
- [4连接数(c++)](/A61.cpp "4连接数")
- [4连接数(python)](/A61.py "4连接数")
- [8连接数(c++)](/A62.cpp "8连接数")
- [8连接数(python)](/A62.py "8连接数")
- [细化处理(c++)](/A63.cpp "细化处理")
- [细化处理(python)](/A63.py "细化处理")
- [Hilditch 细化算法(c++)](/A64.cpp "Hilditch 细化算法")
- [Hilditch 细化算法(python)](/A64.py "Hilditch 细化算法")
- [Zhang-Suen细化算法(c++)](/A65.cpp "Zhang-Suen细化算法")
- [Zhang-Suen细化算法(python)](/A65.py "Zhang-Suen细化算法")
- [HOG-梯度幅值・梯度方向(python)](/A66.py "HOG-梯度幅值・梯度方向")
- [HOG-梯度直方图(python)](/A67.py "HOG-梯度直方图")
- [HOG-直方图归一化(python)](/A68.py "HOG-直方图归一化")
- [HOG-可视化特征量(python)](/A69.py "HOG-可视化特征量")
- [色彩追踪（Color Tracking）(python)](/A70.py "色彩追踪（Color Tracking）")
- [色彩追踪（Color Tracking）(c++)](/A70.cpp "色彩追踪（Color Tracking）")
## 61-70题重要算法解析：
### 4连通
![GitHub set up](/images/4连通.png) 
### 8连通
![GitHub set up](/images/8连通.png)  
### HOG
![GitHub set up](/images/HOG1.png)  
![GitHub set up](/images/图像梯度1.png)  
![GitHub set up](/images/图像梯度2.png)  
![GitHub set up](/images/9bin.png)  
![GitHub set up](/images/block.png)  

## 71-80题的题目：
### 71-80题需要引入头文件：A_71_80.h
- [掩膜（Masking）](/A71.cpp "掩膜（Masking）")
- [掩膜（色彩追踪（Color Tracking）+形态学处理）](/A72.cpp "掩膜（色彩追踪（Color Tracking）+形态学处理）")
- [缩小和放大](/A73.cpp "缩小和放大")
- [使用差分金字塔提取高频成分](/A74.cpp "使用差分金字塔提取高频成分")
- [高斯金字塔（Gaussian Pyramid）](/A75.cpp "高斯金字塔（Gaussian Pyramid）")
- [显著图（Saliency Map）](/A76.cpp "显著图（Saliency Map）")
- [Gabor 滤波器（Gabor Filter）](/A77.cpp "Gabor 滤波器（Gabor Filter）")
- [旋转Gabor滤波器](/A78.cpp "旋转Gabor滤波器")
- [使用Gabor滤波器进行边缘检测](/A79.cpp "使用Gabor滤波器进行边缘检测")
- [使用Gabor滤波器进行特征提取](/A80.py "使用Gabor滤波器进行特征提取")  
## 71-80题重要算法解析：
#### 三角函数+高斯分布
![GitHub set up](/images/三角函数_高斯分布.png)   
### gabor滤波器
![GitHub set up](/images/gabor滤波器.png)  
### gabor实数
![GitHub set up](/images/gabor实数.png)  
### gabor虚数
![GitHub set up](/images/gabor虚数.png)  
### gabor复数
![GitHub set up](/images/gabor复数.png)  
![GitHub set up](/images/gabor1.png)  
![GitHub set up](/images/gabor3.png)  

## 81-90题的题目：
- [Hessian角点检测](/A81.cpp "Hessian角点检测")
- [Harris角点检测第一步：Sobel + Gausian）](/A82.cpp "Harris角点检测第一步：Sobel + Gausian")
- [Harris角点检测第二步：角点检测](/A83.cpp "Harris角点检测第二步：角点检测")
- [简单图像识别第一步：减色化+柱状图^3](/A84.cpp "简单图像识别第一步：减色化+柱状图^3")
- [简单图像识别第二步：判别类别](/A85.cpp "简单图像识别第二步：判别类别")
- [简单图像识别第三步：评估](/A86.cpp "简单图像识别第三步：评估")
- [简单图像识别第四步：k-NN](/A87.cpp "简单图像识别第四步：k-NN")
- [k-平均聚类算法 第一步：生成质心](/A88.cpp "k-平均聚类算法第一步：生成质心")
- [k-平均聚类算法 第二步：聚类](/A89.cpp "k-平均聚类算法第二步：聚类")
- [k-平均聚类算法 第三步：调整初期类别](/A90.py "k-平均聚类算法 第三步：调整初期类别")  
## 81-90题重要算法解析：
#### Hessian矩阵
![GitHub set up](/images/Hessian矩阵.png)   
### 高斯滤波+Hessian矩阵
![GitHub set up](/images/高斯滤波_Hessian矩阵.png)  
### 图像一阶导数
![GitHub set up](/images/图像一阶导数.png)  
### 图像二阶导数
![GitHub set up](/images/图像二阶导数.png)  
### result
![GitHub set up](/images/结果.png)  
### 距离函数
![GitHub set up](/images/距离函数.png)  
### KNN
![GitHub set up](/images/KNN.png)  
### kmeans_dis
![GitHub set up](/images/kmeans_dis.png) 
### 质心
![GitHub set up](/images/质心.png) 

## 91-100题的题目：
- [平均聚类算法-按颜色距离分类](/A91.cpp "平均聚类算法-按颜色距离分类")
- [平均聚类算法-减色处理](/A92.cpp "平均聚类算法-减色处理n")
- [计算IoU](/A93.cpp "计算IoU")

- [随机裁剪（Random Cropping）](/A94.cpp "随机裁剪（Random Cropping）")

- [深度学习（Deep Learning）](/A95.cpp "深度学习（Deep Learning）")

- [神经网络（Neural Network）第二步——训练](/A96.cpp "神经网络（Neural Network）第二步——训练")

- [简单物体检测第一步----滑动窗口（Sliding Window）+HOG](/A97.cpp "简单物体检测第一步----滑动窗口（Sliding Window）+HOG")

- [简单物体检测第二步——滑动窗口（Sliding Window）+ NN](/A98.cpp "简单物体检测第二步——滑动窗口（Sliding Window）+ NN")

- [简单物体检测第三步——非极大值抑制（Non-Maximum Suppression）](/A99.cpp "简单物体检测第三步——非极大值抑制（Non-Maximum Suppression）")

- [简单物体检测第四步——评估（Evaluation）：Precision、Recall、F-Score、mAP](/A100.py "简单物体检测第四步——评估（Evaluation）：Precision、Recall、F-Score、mAP")  

## 91-100题重要算法解析：
#### IOU
![GitHub set up](/images/IOU1.png)  
![GitHub set up](/images/IOU2.png)  
#### NMS：非极大值抑制
![GitHub set up](/images/NMS.png) 
