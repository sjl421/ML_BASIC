import numpy as np


def create_vocabulary_list(dataset):
    '''
    通过数据集创建对应的语料库
    :param dataset: [list]数据集
    :return : 语料库[list]
    '''
    vocab_list = set()
    for sample in dataset:
        vocab_list = vocab_list | set(sample)  # 求并集

    return list(vocab_list)


def words2vec_1(input_vector, vocab_list):
    '''
    将文档 inX 转换成词向量表示
    采用词集模型

    举个例子：
    vocab_list = ['a','b','c'], input_vector = ['b', 'a', 'b']，那么该输入向量的特征向量为[1,1,0]

    :param input_vector : [list]输入样本向量。
    :param vocab_list : [list]语料库。
    :return : input_vector 的词向量[list]
    '''
    output_vector = [0]*len(vocab_list)
    for word in set(input_vector):
        if word in vocab_list:
            output_vector[vocab_list.index(word)] = 1
        else:
            print('the word "{}" is not in my vocabulary!'.format(word))
    return output_vector


def words2vec_2(input_vector, vocab_list):
    '''
    将文档 inX 转换成词向量表示
    采用词袋模型

    举个例子：
    vocab_list = ['a','b','c'], input_vector = ['b', 'a', 'b']，那么该输入向量的特征向量为[1,2,0]

    :param input_vector : [list]输入样本向量。
    :param vocab_list : [list]语料库。
    :return : input_vector 的词向量[list]
    '''
    output_vector = [0]*len(vocab_list)
    for word in input_vector:
        if word in vocab_list:
            output_vector[vocab_list.index(word)] += 1
        else:
            print('the word "{}" is not in my vocabulary!'.format(word))
    return output_vector


def trainNB0(trainset):
    '''
    训练朴素贝叶斯分类器，对一文档/句子是否是侮辱文档/句子
    :param dataset : [np.array]训练集，包括了标签。维度(m,n+1)。m是样本数，n是语料库大小（因为是one-hot编码）
    :return : prob1_vector, prob0_vector, prob_abusive
    '''
    m = trainset.shape[0]
    n = trainset.shape[1] - 1
    prob_abusive = np.sum(trainset[:,-1]) / m  # 计算侮辱样本占总样本的比例, P(c_i = 1)
    prob1_num = np.zeros((1, n))  # 统计所有侮辱样本中语料库每个词出现的次数
    prob0_num = np.zeros((1, n))  # 统计所有非侮辱样本中语料库每个词出现的次数
    prob1_denom = 0.0  # 所有侮辱样本出现的单词总数
    prob0_denom = 0.0  # 所有非侮辱样本出现的单词总数

    for sample in trainset:  # 遍历每个样本
        if sample[-1] == 1:  # 第 i 个样本是侮辱样本
            prob1_num += sample[:-1]
            prob1_denom += np.sum(sample[:-1])
        else:
            prob0_num += sample[:-1]
            prob0_denom += np.sum(sample[:-1])

    prob1_vector = prob1_num / prob1_denom  # 代表每个单词在侮辱性样本出现的概率,可借此判断每个词的词性是否带有侮辱性.P(w_i | c_i = 1)
    prob0_vector = prob0_num / prob0_denom  # 代表每个单词在非侮辱性文档出现的概率,即 P(w_i | c_i = 0)
    return prob1_vector, prob0_vector, prob_abusive


def trainNB1(trainset):
    '''
    (改)训练朴素贝叶斯分类器，对一文档/句子是否是侮辱文档/句子

    trainNB0() 存在两个问题：
    1. P(w|1) = P(w_1|1)*..P(w_n|1) 如果其中有一个概率为0，就导致结果为0.
    2. 下溢出, 很多值很小的数相乘会越来越小,当小到一定程度时,python会四舍五入为0

    :param dataset : [np.array]训练集，包括了标签。维度(m,n+1)。m是样本数，n是语料库大小（因为是one-hot编码）
    :return : prob1_vector, prob0_vector, prob_abusive
    '''
    m = trainset.shape[0]
    n = trainset.shape[1] - 1
    prob_abusive = np.sum(trainset[:,-1]) / m  # 计算侮辱样本占总样本的比例, P(c_i = 1)
    prob1_num = np.ones((1, n))  # 统计所有侮辱样本中语料库每个词出现的次数 （修改处,解决问题1）
    prob0_num = np.ones((1, n))  # 统计所有非侮辱样本中语料库每个词出现的次数 （修改处,解决问题1）
    prob1_denom = 2.0  # 所有侮辱样本出现的单词总数 （修改处,解决问题1）
    prob0_denom = 2.0  # 所有非侮辱样本出现的单词总数 （修改处,解决问题1）

    for sample in trainset:  # 遍历每个样本
        if sample[-1] == 1:  # 第 i 个样本是侮辱样本
            prob1_num += sample[:-1]
            prob1_denom += np.sum(sample[:-1])
        else:
            prob0_num += sample[:-1]
            prob0_denom += np.sum(sample[:-1])

    prob1_vector = np.log(prob1_num / prob1_denom)  # 代表每个单词在侮辱性样本出现的概率,可借此判断每个词的词性是否带有侮辱性.P(w_i | c_i = 1) （修改处,解决问题2）
    prob0_vector = np.log(prob0_num / prob0_denom)  # 代表每个单词在非侮辱性文档出现的概率,即 P(w_i | c_i = 0) （修改处,解决问题2）
    return prob1_vector, prob0_vector, prob_abusive


def classify0(inX, prob0_vector, prob1_vector, prob_abusive):
    '''
    (仅适用于trainNB0得到的结果,因为p1,p0计算用的是np.sum)
    利用朴素贝叶斯训练得到的三个概率。计算 P(inX | c_i) 概率,并取最大值作为 inX 的类别
    :param inX : [np.array(1,n)] 测试样本的词向量
    :param prob1_vector : [np.array(1,n)] 代表每个单词在侮辱性样本出现的概率,可借此判断每个词的词性是否带有侮辱性.P(w_i | c_i = 1)
    :param prob0_vector : [np.array(1,n)] 代表每个单词在非侮辱性样本出现的概率,可借此判断每个词的词性是否带有非侮辱性.P(w_i | c_i = 0)
    :param prob_abusive : [float] 代表侮辱样本出现概率
    '''
    assert (prob1_vector.shape == prob0_vector.shape)
    assert (inX.shape == prob1_vector.shape)
    p1 = 1.0  # inX 类别为 1 的概率
    vector = inX * prob1_vector
    for i in range(vector.shape[1]):
        if inX[0,i] == 1:
            p1 *= vector[0,i]
    p1 *= prob_abusive
    p0 = 1.0  # inX 类别为 0 的概率
    vector = inX * prob0_vector
    for i in range(vector.shape[1]):
        if inX[0,i] == 1:
            p0 *= vector[0, i]
    p0 *= prob_abusive
    # print(p1,p0)
    predict_class = 1 if p1 > p0 else 0
    return predict_class


def classify1(inX, prob0_vector, prob1_vector, prob_abusive):
    '''
    (仅适用于trainNB1得到的结果,因为p1,p0计算用的是np.sum)
    利用朴素贝叶斯训练得到的三个概率。计算 P(inX | c_i) 概率,并取最大值作为 inX 的类别
    :param inX : 测试样本的词向量
    :param prob1_vector : 代表每个单词在侮辱性样本出现的概率,可借此判断每个词的词性是否带有侮辱性.P(w_i | c_i = 1)
    :param prob0_vector : 代表每个单词在非侮辱性样本出现的概率,可借此判断每个词的词性是否带有非侮辱性.P(w_i | c_i = 0)
    :param prob_abusive : 代表侮辱样本出现概率
    '''
    assert (prob1_vector.shape == prob0_vector.shape)
    assert (inX.shape == prob1_vector.shape)
    p1 = np.sum(inX * prob1_vector) + np.log(prob_abusive)  # inX 类别为 1 的概率。其中np.sum()是因为在trainNB1()取了对数
    p0 = np.sum(inX * prob0_vector) + np.log(1 - prob_abusive)  # inX 类别为 0 的概率
    # print(p1, p0)
    predict_class = 1 if p1 > p0 else 0
    return predict_class


def test():
    '''
    简单测试各函数是否正常
    :return:
    '''
    # 伪造训练集和测试集来测试函数
    dataset = ['my dog has flea problems help please', 'maybe not take him to dog park stupid not to',
               'my dalmation is so cute I love him', 'stop posting stupid worthless garbage',
               'mr licks ate my steak how to stop him', 'quit buying worthless dog food stupid']
    dataset = [sentence.split(' ') for sentence in dataset]
    labels = [0, 1, 0, 1, 0, 1]

    testset = ['love my dalmation', 'stupid garbage']
    testset = [s.split(' ') for s in testset]

    # 1. 创建语料库
    vocab_list = create_vocabulary_list(dataset)
    # print(len(vocab_list))

    # 2. 创建词向量表示,并构建训练集
    # for sentence in dataset:
    #     print('the sentence is {}'.format(sentence))
    #     # print('the word vector is {}'.format(words2vec_1(sentence, vocab_list)))
    #     print('the word vector is {}'.format(words2vec_2(sentence, vocab_list)))
    #     print('==============================')
    trainset = np.zeros((len(dataset), len(vocab_list)+1))  # 训练集，维度为(m,n+1)
    for i in range(len(dataset)):
        trainset[i,:-1] = words2vec_1(dataset[i], vocab_list)
        trainset[i, -1] = labels[i]

    # 3. 训练分类器
    p1_v_0, p0_v_0, pA_0 = trainNB0(trainset)
    p1_v_1, p0_v_1, pA_1 = trainNB1(trainset)

    print('语料库', vocab_list)
    print('修改前的：')
    print('发现文档是侮辱类的概率 ', pA_0)
    print('每个单词在侮辱性文档出现的概率', p1_v_0)
    print('每个单词在非侮辱性文档出现的概率', p0_v_0)
    print('修改后的：')
    print('发现文档是侮辱类的概率 ', pA_1)
    print('每个单词在侮辱性文档出现的概率', p1_v_1)
    print('每个单词在非侮辱性文档出现的概率', p0_v_1)

    # 4. 测试
    test_matrix = np.zeros((len(testset), len(vocab_list)))
    for i in range(test_matrix.shape[0]):  # 将测试集每个样本转换为词向量
        test_matrix[i, :] = words2vec_1(testset[i], vocab_list)
    for i in range(test_matrix.shape[0]):  # 循环每个样本
        print(testset[i], ' classified as ', classify0(test_matrix[i, :].reshape(p0_v_0.shape),
                                                       p0_v_0, p1_v_0, pA_0))
        print(testset[i], ' classified as ', classify1(test_matrix[i, :].reshape(p0_v_1.shape),
                                                       p0_v_1, p1_v_1, pA_1))

if __name__ == '__main__':
    test()