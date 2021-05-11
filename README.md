# Simple-quantitative-DNA-analysis
用于对单个DNA条带进行简单的DNA定量分析脚本

# 使用方法
1. 准备含有DNA条带的图片(png, jpg)，示例：

![c](https://user-images.githubusercontent.com/54057111/117829266-2d0b5f00-b2a5-11eb-941d-988b8b623810.png)

2. 创建一个txt文件，写入ug梯度，示例：

![0b45ec4919b04e1fd6ce22417fc1954](https://user-images.githubusercontent.com/54057111/117829222-2250ca00-b2a5-11eb-828b-7405933d1280.png)

3. 在命令地址行输入命令：
语法：python rgbcolorContoursInside1.0.py [样品数目] [DNA条带图片] [含有梯度txt文件]

示例：python rgbcolorContoursInside1.0.py 5 c.png con1.txt

# 输出
类别：
含有识别出条带轮廓的csv坐标文件
算出坐标内的RGB数值的csv文件
识别出轮廓效果的jpg文件

![Contours2](https://user-images.githubusercontent.com/54057111/117830809-917aee00-b2a6-11eb-9657-a73246db2623.jpg)
![Contours3](https://user-images.githubusercontent.com/54057111/117830813-92138480-b2a6-11eb-906e-845242ee04ce.jpg)
![Contours4](https://user-images.githubusercontent.com/54057111/117830815-92ac1b00-b2a6-11eb-8d4e-96aa81fa6878.jpg)
![Contours0](https://user-images.githubusercontent.com/54057111/117830817-9344b180-b2a6-11eb-8db9-09ea848e3f00.jpg)
![Contours1](https://user-images.githubusercontent.com/54057111/117830820-9344b180-b2a6-11eb-8fb0-0e5072976929.jpg)
![Contours-1](https://user-images.githubusercontent.com/54057111/117830822-93dd4800-b2a6-11eb-81b0-a607e3bff462.jpg)

经过图像增强的单个条带的png文件

![blurred](https://user-images.githubusercontent.com/54057111/117830920-abb4cc00-b2a6-11eb-8a38-ffb34ddaf9f6.png)
![imgcrop3](https://user-images.githubusercontent.com/54057111/117830881-a192cd80-b2a6-11eb-8136-99494a35d6c4.png)
![imgcrop4](https://user-images.githubusercontent.com/54057111/117830885-a22b6400-b2a6-11eb-8d27-9527b8ff90bd.png)
![imgcrop0](https://user-images.githubusercontent.com/54057111/117830886-a22b6400-b2a6-11eb-92a3-ace1e524e30f.png)
![imgcrop1](https://user-images.githubusercontent.com/54057111/117830888-a2c3fa80-b2a6-11eb-8737-0d54060649a0.png)
![imgcrop2](https://user-images.githubusercontent.com/54057111/117830891-a35c9100-b2a6-11eb-9236-4d6b5cdaf52c.png)

计算出条带亮度的ex1data3.txt文件

计算出条带亮度并画图的jpg文件(含折线图以及聚类算法分类出的点状DNA坐标图)

![全局平均](https://user-images.githubusercontent.com/54057111/117830744-82943b80-b2a6-11eb-9d8a-b3c4996e622d.jpg)

