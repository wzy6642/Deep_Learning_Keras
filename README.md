# Deep_Learning_Keras
### 第一个案例：二分类问题（基于Pima Indians数据集）
`Pima Indians`数据集为糖尿病患者医疗记录数据，是一个二分类问题。本代码采用80%数据训练，20%数据测试的方法。若数据不做归一化处理，最终模型的分类精度为 79.17%；而数据进行归一化以后，最终模型的分类精度为**81.38%**。

其中还包括一份10折交叉验证的代码，最终的运行结果为**76.69% (+/- 2.95%)**。

sklearn结尾的代码为用sklearn包的`KerasClassifier`进行多分类，通过10折交叉验证，得到最终的精度为**0.7681989076737664**

GridSearch结尾的代码为用sklearn包的`GridSearchCV`搜索超参，得到最终的结果为**Best: 0.781250 using {'batch_size': 10, 'epochs': 150, 'init': 'normal', 'optimizer': 'rmsprop'}**
### 第二个案例：多分类问题（基于Iris数据集）
`Iris`数据集为鸢尾花数据集，是一个拥有4个特征的3分类问题，数据集共有150个样本，最终的精度为**Accuracy: 83.33% (0.30)**
### 第三个案例：回归问题（基于Boston House Price数据集）
`Boston House Price`数据集为1978年波士顿房价的统计数据，共计14个特征，506个样本。最终的精度为**MSE: 12.36 (4.37)。__注意：cross_val_score函数当loss函数为mean_squared_error、mae等时，其得分为负数(cross_val_score里用的指标是负均方误差)，所以最终的MSE指标要在交叉验证的结果上取相反数！__**
### 第四个案例：二分类问题（基于Banking Marking数据集）
本案例中使用`replace`对数据集中的英文进行了编码处理，这是这份代码的一个亮点！对数据作`StandardScaler()`处理，使用`GridSearchCV`搜索超参。最终结果为：**Accuracy: 88.92% (0.01)  Best: 0.886308 using {'units_list': [16]}**
### 第五个案例：神经网络模型的保存与加载（基于Iris数据集）
使用`model.to_json()`保存网络结构，使用`model.save_weights()`保存权重。使用`model_from_json(model_json)`加载已经保存好的模型。另一份代码采用YAML格式对模型进行保存与加载。**通过加载模型的方式建立新的模型后，必须先编译模型，后对新的数据集进行预测。**最终结果为**acc: 99.33%**
### 第六个案例：模型的增量更新（基于Iris数据集）
先用构建好的模型对部分数据进行训练，并将网络的结构以及权重进行保存。然后将保存的网络加载并对剩余的样本进行`增量训练`。相比于全量更新，这样可以大大缩短训练时间。最终结果为：**Base acc: 98.33%\Increment acc: 96.67%**
### 第七个案例：保存检查点（基于Iris数据集）
本代码将val上的accuracy提升时候的模型的权重进行保存(不覆盖前一次的保存结果)。用到了Keras的`ModelCheckpoint`。最后一次的保存结果为：**Epoch 00177: val_acc improved from 0.80000 to 0.83333, saving model to ../model/weights-improvement-177-0.83.h5**
### 第八个案例：保存最好的模型并计算该模型的得分（基于Iris数据集）
本代码中将val上accuracy表现最好的模型的权重进行保存，并将该权重导入用于计算此时的all_accuracy。最终得分为：**acc: 96.00%**
### 第九个案例：模型的accuracy/loss可视化（基于Iris数据集）
通过调用fit返回的`history`绘制网络在train/val上accuracy/loss曲线，以观察模型训练是否收敛。
### 第十个案例：在网络中使用Dropout（基于Iris数据集）
通过在网络中添加`Dropout`层，随机使一部分神经元不参与训练。本代码中首先对输入层添加Dropout层，然后对隐层以及输出层添加Dropout层，经过10折交叉验证，最终的结果分别为：**Accuracy: 74.00% (0.28)/Accuracy: 65.33% (0.29)**
### 第十一个案例：利用学习率衰减找到最优结果（基于Iris数据集）
本代码块包含两种学习率衰减模式，一种为线性衰减，一种为指数衰减。**线性衰减的最终的结果为：loss: 0.4630 - acc: 0.6533/指数衰减的最终结果为loss: 0.3380 - acc: 0.9000**
### 第十二个案例：对手写体识别进行多分类（基于mnist数据集）
`mnist`数据集拥有60000个样本，每张图片均为28x28。在本案例中首先采用传统的多层感知器构建手写体识别的模型，其原理是把每一张图片看成一个向量，其label为图片代表的数字，通过构造神经网络学习feature与label之间的映射关系。其精度为`MLP: 98.09%`;使用卷积神经网络：输入层->卷积层->池化层->Dropout层->Flatten层->全连接层->输出层。最终的分类精度为：**CNN_Small: 99.07%。**

<div align=center><img width="400" height="600" src="https://github.com/wzy6642/Deep_Learning_Keras/blob/master/CNN_mnist/code/Flatten.png" alt="CNN结构图"/></div>

### 第十三个案例：图像增强（基于mnist数据集）
本代码中利用`ImageDataGenerator`对图像进行特征标准化、ZCA白化、旋转、移动、反转、透视操作，并且介绍了文件路径的创建(try:/except:)以及图像的自动保存。
