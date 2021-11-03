from collections import defaultdict, Counter
import math


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

    def update_doc():
        """录入当前计数器的结果"""
        doc_len = sum(doc_counter.values())
        if doc_len == 0:
            return
        index = len(doc_info)
        doc_info.append({
            'id': doc_id,
            'max_tf': doc_counter.most_common(1)[0][1],
            'length': doc_len,
            'words': set(doc_counter.keys())
        })
        for word in doc_counter:
            inverted[word][index] = doc_counter[word]
        doc_counter.clear()

    def filtered_words(words):
        """去除不符合条件的词语"""
        ret = []
        for x in words:
            t = x.strip()
            if len(t) == 0 or t[-2:] in stop_type:
                continue
            ret.append(t)
        return ret

    with open(file_name, 'r', encoding='gbk') as f:
        while True:
            # 读文件直到EOF
            l = f.readline()
            if not l:
                break
            l = l.split('  ')
            # 当前document已结束
            if l[0] == '\n':
                update_doc()
                continue
            # 记录document id并更新词语计数器
            doc_id = l[0][6:15]
            doc_counter.update(filtered_words(l[1:]))
        # 录入最后一篇document
        update_doc()            

    return doc_info, inverted


def cal_tf_idf(docs, words):
    n = len(docs)
    ret = defaultdict(dict)
    for w in words:
        idf = math.log(n/len(words[w]))
        for i in words[w]:
            ret[w][i] = words[w][i] / docs[i]['max_tf'] * idf
    return ret


def cal_doc_norm(docs, tf_idfs):
    for i in range(len(docs)):
        vec_square = [tf_idfs[w][i] * tf_idfs[w][i] for w in docs[i]['words']]
        docs[i]['norm'] = math.sqrt(sum(vec_square))


def cal_similarity(docs, tf_idfs):
    n = len(docs)
    ret = []
    for i in range(n):
        for j in range(i, n):
            common_words = docs[i]['words'] & docs[j]['words']
            similarity = 0
            for w in common_words:
                similarity += tf_idfs[w][i] * tf_idfs[w][j]
            ret.append(similarity / (docs[i]['norm'] * docs[j]['norm']))
    return ret


def main():
    docs, words = read_file()
    tf_idfs = cal_tf_idf(docs, words)
    cal_doc_norm(docs, tf_idfs)
    res = cal_similarity(docs, tf_idfs)


if __name__ == '__main__':
    main()
