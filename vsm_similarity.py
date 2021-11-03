import math
import multiprocessing
from collections import Counter, defaultdict
from functools import partial


def read_file(file_name='199801_clear.txt'):
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
    doc_info = list()               # 文档信息
    inverted = defaultdict(dict)    # 倒排索引
    doc_counter = Counter()
    doc_id = ''
    doc_content = ''

    def update_doc():
        """录入当前计数器的结果"""
        nonlocal doc_content
        doc_len = sum(doc_counter.values())
        if doc_len == 0:
            return
        index = len(doc_info)
        doc_info.append({
            'id': doc_id,
            'max_tf': doc_counter.most_common(1)[0][1],
            'length': doc_len,
            'words': set(doc_counter.keys()),
            'content': doc_content
        })
        for word in doc_counter:
            inverted[word][index] = doc_counter[word]
        doc_counter.clear()
        doc_content = ''

    def filtered_words(words):
        """去除不符合条件的词语"""
        ret = []
        for x in words:
            t = x.strip()
            if len(t) == 0 or t[-2:] in stop_type:
                continue
            ret.append(t)
        return ret

    def split_word(word):
        """分离词语和词性"""
        try:
            t = word.rindex('/')
        except ValueError:
            return word
        return word[:t]

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


def cal_doc_norm(docs, tf_idfs):
    """计算每个文档向量的模长"""
    for i in range(len(docs)):
        vec_square = [tf_idfs[w][i] * tf_idfs[w][i] for w in docs[i]['words']]
        docs[i]['norm'] = math.sqrt(sum(vec_square))


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


if __name__ == '__main__':
    docs, words = read_file()
    tf_idfs = cal_tf_idf(docs, words)
    cal_doc_norm(docs, tf_idfs)
    import time
    for i in range(multiprocessing.cpu_count()):
        t = time.time()
        cal_similarity_mp(docs, tf_idfs, pnum=i+1)
        print(f'并行数{i+1}：耗时{time.time()-t:.3f}s')
