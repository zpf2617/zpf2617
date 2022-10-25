#生成自己的MNIST验证样本数据
#1首先准备图片，在photoshop里面新建保存28*28像素的JPG白底黑字jPG格式，surface pen手写数字，可以用动作录制更快，放在一个文件夹里批量读取，注：文件名要有图片内容的标签，标记标签
#2运行程序生成csv格式的测试（训练）样本数据
#3生成myMNIST.csv后要把第一列索引从文件名改成目标值（0-9）标签

import os
import pandas as pd
import matplotlib.image as mp
%matplotlib inline

file_list=[]        #文件名列表
# df=pd.DataFrame(index=range(30),columns=range(784))
df=pd.DataFrame(columns=range(784)) #新建一个数据784个字段，第一列是索引，后面把索引改成文件名，最后再根据文件名把索引手动改成样本目标值（0~9）
file_path = "C:\\Users\\zero\\.ipython\\myMNIST\\"  #把28*28像素的白底黑字jpg图片放到这个文件夹下批量读取，文件名要有数据目标标签内容，方便标记标签
folders = os.listdir(file_path)
for file in folders:
        #判断文件后缀名是否为txt
        if(file.split('.')[-1]=='jpg'):     #用"."号把完整的文件名分割成  文件名    和(.)   后缀（jpg)
            # 打印所有txt文件名
            # print(file)
            file_list.append(file.split('.')[0])
            full_path=file_path+file
            img_array = mp.imread(full_path)  #图像读取为数组
            img_array=img_array.max(axis=2)   #图像灰度化，最大值法，
            # img_data=img_data.mean(axis=2)   #图像灰度化，平均值值法
            img_data=255-img_array.reshape(784)     #颜色值取反，图像和MNIST数据黑白相反，28*28转化为784，二维转化为一维数组
            # img_list=img_data.tolist()
            df.loc[len(df)]=img_data    #添加一行数据
            pass

# print(file_list)
# print(len(file_list))
df.index=file_list  #更新索引
# print(len(folders))
# print(df)

#生成csv格式测试样本
df.to_csv("myMNIST.csv",header=0)
#生成之后需要打开csv格式把第一列索引（文件名）手动改成0~9的数字标签
