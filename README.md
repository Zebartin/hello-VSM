# hello-VSM
海量数据处理课程作业：计算文本两两之间的相似度

# 计算公式
采用TF-IDF模型计算各个文档中各个词的权重，将文档向量化后计算余弦相似度。

![](https://latex.codecogs.com/gif.latex?\text{similarity}\left(\vec{d_1},\vec{d_2}\right)=\frac{\vec{d_1}\cdot\vec{d_2}}{\left|\vec{d_1}\right|\left|\vec{d_1}\right|})

其中：

![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%5Cvec%7Bd_i%7D%26%3D%5Cleft%28d_%7Bi%2C1%7D%2Cd_%7Bi%2C2%7D%2C%5Cdots%2Cd_%7Bi%2CM%7D%5Cright%29%5C%5C%20d_%7Bi%2Cj%7D%26%3D%5Cfrac%7B%5Ctext%7Btf%7D_%7Bi%2Cj%7D%7D%7B%5Cmax_%7Bk%5Cin%20i%7D%7B%5Ctext%7Btf%7D_%7Bi%2Ck%7D%7D%7D%5Ccdot%5Ctext%7Bidf%7D_j%20%5Cend%7Baligned%7D)

tf指词语j在文档i中的词频，并进行了归一化。词语w的idf则如下定义：

![](https://latex.codecogs.com/gif.latex?\text{idf}_w=\log{\frac{N}{n_w}})

其中N为文档总数，n_w为包含有该词语的文档数量。
