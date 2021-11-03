# hello-VSM
## 计算公式
采用TF-IDF模型计算各个文档中各个词的权重，将文档向量化后计算余弦相似度。

![](https://latex.codecogs.com/gif.latex?\text{similarity}\left(\vec{d_1},\vec{d_2}\right)=\frac{\vec{d_1}\cdot\vec{d_2}}{\left|\vec{d_1}\right|\left|\vec{d_1}\right|})

其中：

![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%5Cvec%7Bd_i%7D%26%3D%5Cleft%28d_%7Bi%2C1%7D%2Cd_%7Bi%2C2%7D%2C%5Cdots%2Cd_%7Bi%2CM%7D%5Cright%29%5C%5C%20d_%7Bi%2Cj%7D%26%3D%5Cfrac%7B%5Ctext%7Btf%7D_%7Bi%2Cj%7D%7D%7B%5Cmax_%7Bk%5Cin%20i%7D%7B%5Ctext%7Btf%7D_%7Bi%2Ck%7D%7D%7D%5Ccdot%5Ctext%7Bidf%7D_j%20%5Cend%7Baligned%7D)

tf指词语j在文档i中的词频，并进行了归一化。词语w的idf则如下定义：

![](https://latex.codecogs.com/gif.latex?\text{idf}_w=\log{\frac{N}{n_w}})

其中N为文档总数，n_w为包含有该词语的文档数量。

## 代码说明

### 读取文件

逐行读取文件，取每行开头的信息作为文档ID，记录每个文档的信息，以及所有词语的倒排表。

这里参照了[百度停用词表](https://github.com/goto456/stopwords/blob/master/baidu_stopwords.txt)，选择一些停用的词性如下：

```python
stop_type = {
    '/u',   # 助词，得、地、的
    '/w',   # 标点符号
    '/k',   # 们
    '/p',   # 连词，因为、根据
    '/f',   # 方位、时间，以前、里面、中间
    '/r',   # 代词，他们、大家、本报
    '/c',   # 介词，虽然、要不是、尽管
    '/y'    # 语气词，了、啊、吗
}
```

读取文件的代码如下，此处省略了一部分，具体见源代码文件。

```python
def read_file(file_name='199801_clear.txt'):
    stop_type = { ... }
    doc_info = list()               # 文档信息
    inverted = defaultdict(dict)    # 倒排索引
    doc_counter = Counter()
    doc_id = ''
    doc_content = ''
    def update_doc():
        """录入当前计数器的结果"""
        pass
    def filtered_words(words):
        """去除不符合条件的词语"""
        pass
    def split_word(word):
        """分离词语和词性"""
        pass
    with open(file_name, 'r', encoding='gbk') as f:
        while True:
            # 读文件直到EOF
            l = f.readline()
            if not l:
                # 录入最后一篇document
                update_doc()
                break
            l = l.split('  ')
            if l[0] == '\n':
                continue
            cur_doc_id = l[0][6:15]
            # 当前document已结束
            if doc_id != cur_doc_id:
                update_doc()
                doc_id = cur_doc_id
            doc_counter.update(filtered_words(l[1:]))
            doc_content += ''.join([split_word(w) for w in l[1:]])
    return doc_info, inverted
```

最后得到的`doc_info`是一个`list`，以`doc_info[0]`为例：

```json
{
  "id": "01-01-001",
  "max_tf": 24,
  "length": 626,
  "words": ["经验/n", "永远/d", "..."],
  "content": "迈向充满希望的新世纪——一九九八年新年讲话..."
} 
```

得到的`inverted`记录了词语在不同文档中的词频，形如：

```json
{
  "迈向/v": {
    0: 2,
    2: 1,
    36: 1,
    ...,
    2969: 1
  },
  "充满/v": {
    0: 3,
    1: 3,
    2: 2,
    ...,
    3125: 1
  },
  ...
}
```

这里的词语是将词本身和词性放在一起看待的，这出于两方面的考量：

- 一方面，不同词性的同一个词意思可能不同，比如“江”既可以指代人的姓（词性nr），也可以指代江水（词性n）；“高度”可以是名词，也可以是副词，比如“高度评价”

- 另一方面，虽然有一些不同词性的词语意思是很相近的，比如“讲话”的词性可以是n、v或vn，但筛选这些词语具有一定难度，故只好按不同词语计数

### 预计算

#### 计算tf-idf

预先计算各个词语在不同文档中的tf-idf，并保存在倒排表中。

```python
def cal_tf_idf(docs, words):
    """计算各个词语在不同文档中的tf-idf"""
    n = len(docs)
    ret = defaultdict(dict)
    for w in words:
        idf = math.log(n/len(words[w]))
        for doc_index in words[w]:
            ret[w][doc_index] = words[w][doc_index] / \
                docs[doc_index]['max_tf'] * idf
    return ret
```

#### 计算文档向量模长（范数）

通过上述计算好的tf-idf进一步计算文档向量的模长，保存在`docs`的新字段中，方便后续计算。

```python
def cal_doc_norm(docs, tf_idfs):
    """计算每个文档向量的模长"""
    for i in range(len(docs)):
        vec_square = [tf_idfs[w][i] * tf_idfs[w][i] for w in docs[i]['words']]
        docs[i]['norm'] = math.sqrt(sum(vec_square))
```

### 计算文本相似度

完成预计算后，即可开始计算文档两两之间的相似度。计算的时候只计算当前文档与后续文档的相似度，不重复计算。因此结果是一个主对角线为0的上三角矩阵，返回结果是一个`list`，从上到下从左到右地保存着这个三角矩阵，需要用特殊的方法来读取指定的相似度。

具体计算过程如下：

- 找到文档i和文档j共有的词语；
- 遍历这些词语，将它们在两个文档中的tf-idf值两两相乘并累加；
- 将结果除以两个文档的模的乘积。

```python
def cal_similarity(docs, tf_idfs):
    """计算两两之间的的相似度"""
    n = len(docs)
    ret = []
    for i in range(n):
        for j in range(i + 1, n):
            # 只考虑交集内的词语
            common_words = docs[i]['words'] & docs[j]['words']
            similarity = 0
            for w in common_words:
                similarity += tf_idfs[w][i] * tf_idfs[w][j]
            ret.append(similarity / (docs[i]['norm'] * docs[j]['norm']))
    return ret
```

预计算减少了大量重复计算，因此这一部分耗时不算太高，3000多的文档相似度计算耗时约40秒左右。

### 多进程计算相似度

可以考虑通过多进程进一步减少耗时。以下的计算过程与单进程版基本一致，只是使用了[multiprocessing.pool.Pool](https://docs.python.org/zh-cn/3/library/multiprocessing.html#multiprocessing.pool.Pool.starmap)的`starmap`方法来分发进程。

```python
def cal_similarity_mp(docs, tf_idfs, pnum=multiprocessing.cpu_count()):
    """计算两两之间的的相似度（多进程版）"""
    n = len(docs)
    proc = partial(_cal_similarity_proc, docs=docs, tf_idfs=tf_idfs)
    with multiprocessing.Pool(processes=pnum) as p:
        return p.starmap(proc, [(i, j) for i in range(n) for j in range(i + 1, n)])


def _cal_similarity_proc(i, j, docs, tf_idfs):
    # 只考虑交集内的词语
    common_words = docs[i]['words'] & docs[j]['words']
    ret = 0
    for w in common_words:
        ret += tf_idfs[w][i] * tf_idfs[w][j]
    return ret / (docs[i]['norm'] * docs[j]['norm'])
```

尝试了不同并行数，最后选取4为并行数。使用多进程优化后耗时在20秒左右，计算速度提升1倍左右。

## 结果验证

以上计算的结果需要特殊的方式才能正确读取，因此ipynb文件中定义了一些实用方法来读取文档的相似度。利用其中的方法可以查看相似度结果的效果如何。这里举几个比较有意思的例子：

首先是前国家主席江泽民同志的新年讲话（文档ID为01-01-001），其中谈到了香港回归、党的十五大、国民经济、中国外交、澳门回归、台湾问题、世界局势等方面。而与之最相似的文档是前国家总理李鹏同志在春节团拜会上的讲话（文档ID为28-01-002），也谈到了基本一样的话题。两者之间的相似度约为0.55，算是比较高的。

另一篇文档标题为《落实帮扶责任制实施开发式扶贫哈尔滨２７万人告别贫困》（文档ID为29-02-014），与之最相似的文档为《贫困农户自立工程初见成效，中国扶贫基金会呼吁全社会继续关注支持扶贫事业》（文档ID为29-02-005），两者都提及扶贫基金的话题，相似度为0.32。

也有一些文档与其他文档都不太相似，比如《可口饭菜送井下》（文档ID为21-12-011）讲述给矿工送饭盒的事情，其最相似的文档为《湖南组织特困矿工共同脱贫》（文档ID为25-02-015），相同点仅有“矿工”，相似度仅有0.08。

## 更多改进

以上仍然存在一些问题，但碍于种种原因无暇顾及：

- 可以考虑用[Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law)提取关键词后再计算相似度；
- 相似度结果的保存和读取没必要那么扭曲，有待改进
