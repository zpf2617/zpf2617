# zpf2617
神经网络测试，neuralnetwork

jupyter book 中文乱码，直接上传文本试试

第一部分 NeuralNetwork_train.py
#《Py神经网络编程》塔里克·拉希德，人民邮电出版社，出版时间2018-04，第2章第4节编程实例
# zpf2617，测试环境anaconda base Python 3.9.13
#2022.10.26测试，测试原代码地址：https://github.com/zpf2617
#感谢这本书让我这个0基础的人从头手撸了一个神经网络！感谢作者！感谢微信读书让我免费读了这本书！新疆库尔勒疫情期间无聊的生活里的一点小乐趣，希望自己能扛过去，希望家人健康平安，媳妇早早回家
#MNIST数据集的下载地址：https://pjreddie.com/projects/mnist-in-csv/,原书源代码：https://github.com/makeyourownneuralnetwork
##增加世代，重复训练，可以减小训练样本数量，提高计算准确率，比如循环训练训练样本5遍，训练太多遍会过度拟合，只对特定样本敏感epochs=5
#所有可以修改和应该修改的变量：所有路径（权重保存和读取最好默认）、170行epochs训练循环次数、151行hidden_nodes隐藏层节点数量、
#、157行learning_rate学习率、252行label标签（0~9)，MNIST.csv自行下载，第一列是标签，2~785是样本数据
#新手0基础学习，有问题多多指教！

第二部分 NeuralNetwork_build_img.py
#把MNIST数据csv格式绘制成图片，方便目测检查结果

第三部分 NeuralNetwork_databuild.py
#生成自己的MNIST验证样本数据
#1首先准备图片，在photoshop里面新建保存28*28像素的JPG白底黑字jPG格式，surface pen手写数字，可以用动作录制更快，放在一个文件夹里批量读取，注：文件名要有图片内容的标签，标记标签
#2运行程序生成csv格式的测试（训练）样本数据
#3生成myMNIST.csv后要把第一列索引从文件名改成目标值（0-9）标签
