# Deep_Learning_Keras
### 第一个案例：二分类问题（基于Pima Indians数据集）
`Pima Indians`数据集为糖尿病患者医疗记录数据，是一个二分类问题。本代码采用80%数据训练，20%数据测试的方法。若数据不做归一化处理，最终模型的分类精度为 79.17%；而数据进行归一化以后，最终模型的分类精度为**81.38%**。

其中还包括一份10折交叉验证的代码，最终的运行结果为**76.69% (+/- 2.95%)**。

sklearn结尾的代码为用sklearn包的`KerasClassifier`进行多分类，通过10折交叉验证，得到最终的精度为**0.7681989076737664**

GridSearch结尾的代码为用sklearn包的`GridSearchCV`搜索超参，得到最终的结果为**Best: 0.781250 using {'batch_size': 10, 'epochs': 150, 'init': 'normal', 'optimizer': 'rmsprop'}**
### 第二个案例：多分类问题（基于Iris数据集）
`Iris`数据集为鸢尾花数据集，是一个拥有4个特征的3分类问题，数据集共有150个样本，最终的精度为**Accuracy: 83.33% (0.30)**
### 第三个案例：回归问题（基于Boston House Price数据集）
`Boston House Price`数据集为1978年波士顿房价的统计数据，共计14个特征，506个样本。最终的精度为**MSE: 12.36 (4.37)**。__注意：cross_val_score函数当loss函数为mean_squared_error时，其得分为负数，所以最终的MSE指标要在交叉验证的结果上取负数！__
