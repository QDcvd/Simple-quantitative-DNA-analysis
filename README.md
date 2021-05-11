# Simple-quantitative-DNA-analysis
用于对单个DNA条带进行简单的DNA定量分析脚本

# 使用方法
1. 准备含有DNA条带的图片(png, jpg)，示例：

![origin](https://user-images.githubusercontent.com/54057111/117839592-f38b2180-b2ad-11eb-8432-642617e9e7f7.jpg)


2. 创建一个txt文件，写入ug梯度，示例：

![0b45ec4919b04e1fd6ce22417fc1954](https://user-images.githubusercontent.com/54057111/117829222-2250ca00-b2a5-11eb-828b-7405933d1280.png)

3. 在命令地址行输入命令：
语法：python rgbcolorContoursInside2.0.py [样品数目] [DNA条带图片] [含有梯度txt文件]

示例：python rgbcolorContoursInside2.0.py 5 con1.txt origin.jpg

4. 对图片进行裁剪，左键拖动框框至合适位置，关闭窗口。

![0531408f1517aaf39f00221aa6fa745](https://user-images.githubusercontent.com/54057111/117839764-1b7a8500-b2ae-11eb-9984-d0034aa9f481.png)



# 输出
类别：
含有识别出条带轮廓的csv坐标文件

算出坐标内的RGB数值的csv文件

识别出轮廓效果的jpg文件

![Contours2](https://user-images.githubusercontent.com/54057111/117839926-41a02500-b2ae-11eb-9cad-c4076892e30b.jpg)
![Contours3](https://user-images.githubusercontent.com/54057111/117839930-42d15200-b2ae-11eb-9329-198f2b0eb9bd.jpg)
![Contours4](https://user-images.githubusercontent.com/54057111/117839932-42d15200-b2ae-11eb-91d3-d17283feb89a.jpg)
![Contours0](https://user-images.githubusercontent.com/54057111/117839935-4369e880-b2ae-11eb-92ca-618c49ee9823.jpg)
![Contours1](https://user-images.githubusercontent.com/54057111/117839937-44027f00-b2ae-11eb-93e6-5da32e67e38e.jpg)
![Contours-1](https://user-images.githubusercontent.com/54057111/117839940-44027f00-b2ae-11eb-98d6-c0836f4f809f.jpg)

经过图像增强（图像增强只为获取条带轮廓坐标点）的单个条带的png文件

![blurred](https://user-images.githubusercontent.com/54057111/117840036-5e3c5d00-b2ae-11eb-958c-fd37acd5b40f.png)
![imgcrop2](https://user-images.githubusercontent.com/54057111/117840005-57154f00-b2ae-11eb-975b-e8a6694e6d79.png)
![imgcrop3](https://user-images.githubusercontent.com/54057111/117840007-57ade580-b2ae-11eb-89e5-f53d495ede3d.png)
![imgcrop4](https://user-images.githubusercontent.com/54057111/117840010-58467c00-b2ae-11eb-9d2e-3a548414545c.png)
![imgcrop0](https://user-images.githubusercontent.com/54057111/117840012-58df1280-b2ae-11eb-8c07-eb5001c59307.png)
![imgcrop1](https://user-images.githubusercontent.com/54057111/117840014-58df1280-b2ae-11eb-9e39-9a64a487b03d.png)

计算出条带亮度的ex1data3.txt文件

计算出条带亮度并画图的jpg文件(含折线图以及聚类算法分类出的点状DNA坐标图)

![全局平均](https://user-images.githubusercontent.com/54057111/117840054-63011100-b2ae-11eb-949d-6e09870f7e07.jpg)


