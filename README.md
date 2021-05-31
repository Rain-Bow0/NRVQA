# NRVQA

> 本科毕业设计-用户生成内容视频客观质量评价方法研究
> 这是2021年我在fzu本科毕设的代码文件，是由https://github.com/lidq92/VSFA 方法进行改进得到

网络结构如下图，改进部分为CNN提取，使用了ResNet50和DenseNet101进行，改进神经网络降维部分，并使用LSTM来进行提取时序关系


![未命名文件 (3)](https://user-images.githubusercontent.com/36041684/120127571-493e5400-c1f2-11eb-9962-3d57c0eecd90.jpg)

代码部分分为两部

- 第一部分是CNNfeatures，作用是对数据集视频进行特征提取并保存，方便后面的回归

- 第二部分是regression，对提取的特征，分为训练集、验证集、测试集，进行训练然后在验证集上提取最好的效果，并在测试集上进行测试

```
python CNNfeatures.py
python regression.py
python test_demo.py
```
