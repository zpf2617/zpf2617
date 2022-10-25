#把MNIST数据csv格式绘制成图片，方便目测检查结果
import numpy
import matplotlib.pyplot    #把csv格式的MNIST数据显示出来
# import maplotlib.pyplot
#装饰器，只用matplotlib.pyplot 绘图，装饰器后面不能加注释
%matplotlib inline  

#一次性读取文件
data_file = open("C:/mnist_test_100.csv",'r')

# data_file = open("C:/myMNIST.csv",'r')    #自建的MNIST样本
data_list = data_file.readlines()
data_file.close()
#数据集第n条数据（从1开始）第n行

#调整这个行数来显示
n=4

all_values=data_list[n-1].split(",")
image_array=numpy.asfarray(all_values[1:]).reshape(28,28)
matplotlib.pyplot.imshow(image_array,cmap="Greys",interpolation='None')

print("样本模板：",all_values[0])