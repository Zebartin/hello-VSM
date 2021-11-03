# hello-VSM
海量数据处理课程作业：计算文本两两之间的相似度

# 计算公式
采用TF-IDF模型计算各个文档中各个词的权重，将文档向量化后计算余弦相似度。

\[\text{similarity}\left(\vec{d_1},\vec{d_2}\right)=\frac{\vec{d_1}\cdot\vec{d_2}}{\left|\vec{d_1}\right|\left|\vec{d_1}\right|}\]

其中：
$$\begin{aligned}
\vec{d_i}&=\left(d_{i,1},d_{i,2},\dots,d_{i,M}\right)\\
d_{i,j}&=\left[a+(1-a)\frac{\text{tf}_{i,j}}{\max_{k\in i}{\text{tf}_{i,k}}}\right]\cdot\text{idf}_j
\end{aligned}$$

$\text{tf}_{i,j}$指词语$j$在文档$i$中的词频，并采用了[maximum tf normalization](https://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html)进行归一化，其中$a$取$0.4$。

词语$w$的$\text{idf}_w$则如下定义：

\[\text{idf}_w=\log{\frac{N}{n_w}}\]

其中$N$为文档总数，$n_w$为包含有该词语的文档数量。

- Pivoted Length Normalization VSM [Singhal et al 96]

  \[f(q,d)=\sum_{w\in q\cap d}{c(w,q)\frac{\ln{\left[1+\ln{\left[1+c(w,d)\right]}\right]}}{1-b+b\frac{|d|}{\text{avdl}}}}\ln{\frac{M+1}{\text{df}(w)}}\]

- BM25/Okapi [Robertson & Walker 94]

  \[f(q,d)=\sum_{w\in q\cap d}{c(w,q)\frac{(k+1)c(w,d)}{c(w,d)+k\left(1-b+b\frac{|d|}{\text{avdl}}\right)}}\ln{\left(\frac{M-\text{df}(w)+0.5}{\text{df}(w)+0.5}+1\right)}\]
