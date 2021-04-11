# COVID19_prediction
    这是一个基于COVID19公开数据集的时间序列模型，对它进行构建并训练来预测未来全球每日康复人数。运行环境是windows系统，基于Python3下的Tensorflow框架进行实现。
    它的数据集由 https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset 的公开数据集运算转化而来，变为了存有日期和全球每日康复人数这两列数据的recovered.csv文件。而data_load.py用于读取该文件数据，并将数据划分为训练集和测试集两部分。
    该工程的主文件是train_rnn.py，需要运行它来构建并训练测试这个模型。在主函数处可对模型的序列大小、学习率、迭代次数等超参数进行设置，然后对类Predictor进行实例化。训练时直接使用了已有的BasicLSTM网络，损失函数选择了交叉熵函数，优化器选择了AdamOptimizer。每迭代500次输出一次训练集误差和测试集误差，最终完成训练后保存当前网络参数模型并返回此时的模型性能指标。为了进行交叉验证，通过split来改变训练数据和测试数据的取值范围，一共进行了五次独立的模型训练并分别输出它们的MSE来比较性能。最后还对模型的结果进行了可视化，将训练结果、预测结果、实际结果在一张图上进行绘制连接，保存为prediction.png。

