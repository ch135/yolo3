# yolo3
YOLO3 base on tensorflow
## yolo3处理流程
1. 使用K-means聚类方法求出预测出9个预测边框的宽Pw, 高Ph。
2. 使用 Drrknet-53 网络，求出基础特征矩阵。Darknet-53 在每层卷积网络后面都会batch-normalizatin处理。
   能够加快模型收敛，并提供一定的正则化。
3. 将基础特征矩阵输入到 feature pyramiad networks(FPN) 中，FPN输出一个数组，数组中包含三个不同尺度的
   结果。其中，最后一个包含编码边界框，目标和预测类信息。
4. 根据特征数组的数据，计算求出预测边框（公式如下）和目标类型和概率。训练时，参数使用均值方差
   （sum of squared error loss）求导更新。对于训练出来每个物体的9个预测边框，当目标概率大于阙值且为当
   前最大时，保留边框；当大于阙值但不为最大时，忽视边框；当边框没有目标物体时，惩罚objectness。最后保留
   一个边框。
5. 计算目标类型和概率。训练时，目标类型概率使用逻辑回归（Sigmoid functiion）进行计算，参数使用交叉熵误差
   （binary cross-entropy loss）进行更新。
6. 计算mAP，衡量算法质量。

总结：
yolo3 计算速度快，但IOU不高，提高阙值时，性能减低。

注意：
   AP：训练边框与实际边框的IOU值
   mAP: 每次循环训练中，不同类别的AP均值
   AP50: 阙值为0.5的mAP
   
## yolo2处理流程

## yolo1处理流程
