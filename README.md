# yolo3
   YOLO3 base on tensorflow
## yolo3处理流程
1. 使用**K-means**聚类方法求出预测出9个预测边框的宽Pw, 高Ph。
2. 使用 **Drrknet-53** 网络，求出基础特征矩阵。Darknet-53 在每层卷积网络后面都会batch-normalizatin处理。
   能够加快模型收敛，并提供一定的正则化。
   <div>
      <img src="https://github.com/ch135/yolo3/blob/master/formule/network3.png"/>
   </div>
3. 将基础特征矩阵输入到 feature pyramiad networks(FPN) 中，FPN输出一个数组，数组中包含三个不同尺度的
   结果。其中，最后一个包含编码边界框，目标和预测类信息。
4. 根据特征数组的数据，计算求出预测边框（公式如下）和目标类型和概率。训练时，参数使用均值方差
   （sum of squared error loss）求导更新。对于训练出来每个物体的9个预测边框，当目标概率大于阙值且为当
   前最大时，保留边框；当大于阙值但不为最大时，忽视边框；当边框没有目标物体时，惩罚objectness。最后保留
   一个边框。
   <div>
      <img src="https://github.com/ch135/yolo3/blob/master/formule/border.png"/>
   </div>
5. 计算目标类型和概率。训练时，目标类型概率使用逻辑回归（Sigmoid functiion）进行计算，参数使用交叉熵误差
   （binary cross-entropy loss）进行更新。
6. 计算mAP，衡量算法质量。

总结：
  yolo3 计算速度快，但IOU不高，提高阙值时，性能减低。

注意：
  AP：训练边框与实际边框的IOU比值。<br/>
  mAP: 每次循环训练中，不同类别的AP均值。<br/>
  AP50: 阙值为0.5的mAP。
   
## yolo2处理流程
   当前 objects detection 问题
   - 能检测到物体少
   - 用于object detection 的数据集比较少
   - 对比于其他算法，如Fast R-CNN, YOLO存在物体定位出错，低召回率的问题<br/>
   基于以上问题，YOLO2 提出了一系列改进方法，使网络**Better, Faster, Stronger**
### Better
   - Btach Normalization<br/>
      Batch Normalization 能加快网络的训练速度，同时提供正则化，避免过拟合。作者在每个conv层都加上了了BN层，同时去掉了原来模型中的drop out部分，这带来了2%的性能提升。
   
   - Hight Resolution<br/>
      在**检测模型**训练完成后，作者利用**ImageNet**数据集，首次将448*448分辨率的子块输入到网络进行**分类**训练，训练次数为10 epoch。这让网络更好地处理高分辨率图像。
      
   - Convolutional With Anchor Boxes<br/>
      YOLO2 移除了 YOLO1的全连接层，使用 anchor boxes 来预测目标边框。同时，作者移除了一个池化层，提高特征矩阵的分辨率。
      
      在检测模型中，将网络输入从448x448转化成 416x416，这能够得到一个 center cell(大目标物体更能占据特征矩阵中间位置)
      
      
      
## yolo1处理流程
      yolo1 将目标检测问题设计为回归问题。能更好的区分目标和普通物体，但对于边框定位会产生更大误差。
     
### 1. 网格划分
   yolo1将图片切分成古固定大小的 SxS小格。在特征提取后，会产生 B个预测边框，每个边框会产生对应的5个参数，分别是 x, y, w, h, confidence（置信      度），x, y代表下相对于网格单元边界框的中心坐标，h, w代表相对与整张图片的高和宽，confidence用于判断边框中是否有目标物体，同时判断物体是目标物体    的概率，公式如下图。
   <div>
      <img src="https://github.com/ch135/yolo3/blob/master/formule/formule2_1.png"/>
   </div>
   为了提高网络更好得区分目标物体，网络对目标框和非目标框分别添加了参数（惩罚机制），发别为 **Ycoord=5**和 **Ynoobj=0.5**。
   为更好选中小目标，网络对小于某个阙值的边框进行平方计算。
   
   同时，grid cell会产生一个数量为C的类别概率，用于在非极大值抑制时，判断是否保留边框的条件，公式如下图。
   <div>
      <img src="https://github.com/ch135/yolo3/blob/master/formule/formule2_2.png"/>
   </div>
   
   
   所以， 网络最终输出张量大小为 S*S*(B*5+C)。
 
### 2. 网络设计
   网络结合 **GoogLeNet**网络的思想，有24个卷积层，加上2个全连接层(FCL)，卷积核使用 1*1和 3*3的大小，网络结构如下图所示。
   <div>
      <img src="https://github.com/ch135/yolo3/blob/master/formule/network2.png"/>
   </div>
   其中，为了提高检测的性能，yolo1添加了4个卷积层进行预训练，在网络后面加上2个FCL。
   
   另外， 网络中交替使用 1*1和 3*3卷积核是为了减少特征矩阵空间维度

### 3. 模型训练
   网络的运行流程如下
   1）**Darknet-20**: 提取图片的特征矩阵；
   2）Detection: 通过两个FCL进行目标检测。
   3）Class probabilities and Bounding box coordinates：回归预测目标概率并选中目标
   
   网络最后一层使用了线性激活函数，其他层使用了LReLU(0.1)函数；最后一层使用求和平方误差(sum-square error)，其他层使用的误差函数如下图(MSE)：
   <div>
      <img src="https://github.com/ch135/yolo3/blob/master/formule/loss2.png"/>
   </div>
   其中，1i 代表是否有物体在小格子中；1ij 代表 i格子的第 j个边框是新的预测边框
   
  - 在训练过程中，学习效率随epoch变化而变化：<br/>
     - epoch=75: 10^-2 to 10^-3
     - epoch=105:10-3 to 10^-4
     - epoch=150: 10^-4
  
  - 为避免过拟合，用了 dropout(0.5)
   
   
   
   
   

