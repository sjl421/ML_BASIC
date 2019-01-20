import numpy as np


def stump_train(trainset, D, is_print=False):
    '''
    单层决策树的训练过程
    :param trainset: [np.array(m,n+1)] 训练集
    :param D: [np.array(m,1)] 样本权重向量
    :param is_print: [boolean] 是否输出中间过程
    :return: [dict] best_stump 最佳单层决策树, 字段有{dim:特征索引号; thresh_val:阈值; inequal:假设;}
             [float] min_error 最佳单层决策树的训练错误率
    '''
    m, n = trainset.shape
    n = n-1
    labels = trainset[:,-1].reshape((m,1))  # 真实标签
    step_num = 10  # 步长, 即待选阈值的个数
    assert(D.shape == (m,1))

    best_stump = {}  # 最佳单层决策树
    min_error = float('inf')  # 最佳单层决策树的错误率（也是最小的错误率）,初始化为正无穷
    best_predicts = None  # 最佳单层决策树的分类结果
    for i in range(n):  # 遍历每个特征
        feature_min = trainset[:,i].min()  # 第 i 个特征的最小值
        feature_max = trainset[:,i].max()  # 第 i 个特征的最大值
        step_size = (feature_max - feature_min) / float(step_num)  # 步距
        for j in range(-1, step_num+1):  # 遍历每个阈值
            thresh_val = feature_min + j * step_size  # 当前选择的阈值
            for inequal in ['lt', 'gt']:  # 遍历两种假设
                # 1. 预测分类结果
                cur_predicts = stump_classify(trainset, i, thresh_val, inequal)  # 构建单层决策树并预测分类结果

                # 2. 计算当前分类器的加权错误率
                error_array = np.ones(cur_predicts.shape)  # 错误向量, 分类错误的样本的标记为1
                error_array[cur_predicts == labels] = 0  # 分类正确的样本的标记置零
                # 因为 D 就是一个概率分布,所有不需要再除以 m
                weight_error = float(np.dot(D.T, error_array))  # 加权错误率 np.dot((1,m),(m,1))

                # 3. 如果加权错误率最小，则将当前的单层决策树设为最佳单层决策树
                if weight_error < min_error:
                    min_error = weight_error
                    best_stump['dim'] = i
                    best_stump['thresh_val'] = thresh_val
                    best_stump['inequal'] = inequal
                    best_predicts = cur_predicts.copy()

                if is_print:
                    print('split: 第 {} 个特征, 阈值 {}, 假设 {}, 加权错误率 {}'.format(i, thresh_val, inequal, weight_error))

    return best_stump, min_error, best_predicts


def stump_classify(trainset, dim, thresh_val, inequal):
    '''
    单层决策树的分类函数
    预测结果 +1 和 -1
    :param trainset: [np.array] 训练集
    :param dim: [int] 第 dim 个特征
    :param thresh_val: [float] 阈值
    :param inequal: [str] 假设
    :return: [np.array(m,1)] 单层决策树的分类函数
    '''
    cur_predicts = np.ones((trainset.shape[0],1))  # 当前分类器的分类结果
    if inequal == 'lt':  # 大于阈值为正样本
        cur_predicts[trainset[:, dim] <= thresh_val] = -1.0
    else:  # 大于阈值为负样本
        cur_predicts[trainset[:, dim] > thresh_val] = -1.0
    return cur_predicts


def adaboost_trainDS(trainset, iter_num=40, is_print=False):
    '''
    基于单层决策树的 AdaBoost 训练过程
    :param trainset: [np.array(m,n+1)] 训练集
    :param iter_num: [int] 迭代次数
    :param is_print: [boolean] 是否输出中间过程
    :return: [list] DS_array 单层决策树的数组,即多个弱分类器的组合，形成最终分类器
             [np.array(m,1)] 最终分类器的分类结果
    '''
    m,n = trainset.shape
    n = n-1
    labels = trainset[:,-1].reshape((m,1))

    # 1. 初始化
    D = np.ones(labels.shape) / m  # 初始化样本权重向量
    DS_array = []  # 单层决策树的数组
    agg_predicts = np.zeros(labels.shape)  # 最终分类器的分类结果

    # 2. 得到单层决策树的数组
    for i in range(iter_num):
        # 2.1 基于当前样本权重向量，找到最佳的单层决策树
        best_stump, min_error, best_predicts = stump_train(trainset, D)

        # 2.2 计算当前弱分类器的权重
        alpha = float(0.5 * np.log((1 - min_error) / max(min_error, 1e-16)))  # 分母的处理是为了分母过小,程序四舍五入为 0 造成零溢出


        # 2.3 将最佳单层决策树加入到单层决策树数组
        best_stump['alpha'] = alpha
        DS_array.append(best_stump)

        # 2.4 计算新的样本权重向量 D
        expon = np.multiply(-1*alpha, labels * best_predicts)  # 因为正确分类和错分的样本计算公式不同
        D = D * np.exp(expon)
        D = D / np.sum(D)

        # 2.5 计算由当前弱分类数组组成形成的最终分类器的分类结果
        agg_predicts += alpha * best_predicts  # 是个累加的过程

        # 2.6 更新最终分类器错误率
        error_arr = np.ones(labels.shape)
        error_arr[np.sign(agg_predicts) == labels] = 0
        error_rate = float(np.sum(error_arr)) / m

        if is_print:
            print('第 {} 次迭代'.format(i + 1))
            print('当前弱分类器的错误率 ：', min_error)
            print('最终分类器的错误率 ：', error_rate)
            print('=======================================')

        if error_rate == 0.0:
            break

    return DS_array, agg_predicts


def adaboost_classify(classifier_array, testset):
    '''
    用 AdaBoost 分类器进行分类
    :param classifier_array: [list] 弱分类器组
    :param testset: [np.array(m,n)] 测试集
    :return: [np.array(m,1)] 分类结果
    '''
    agg_labels = np.zeros((testset.shape[0],1))
    for c in classifier_array:
        predicts = stump_classify(testset, c['dim'], c['thresh_val'], c['inequal'])
        agg_labels += c['alpha'] * predicts
    return np.sign(agg_labels)


def load_file_data(filename):
    '''
    从文件中读取数据
    :param filename: [str] 完整的文件名
    :return: [np.array(m,n+1)] 数据集
    '''
    fr = open(filename)
    lines = fr.readlines()
    m = len(lines)
    n = len(lines[0].strip().split('\t')) - 1
    dataset = np.zeros((m,n+1))
    index = 0
    for l in lines:
        dataset[index,:] = l.strip().split('\t')
        index += 1
    return dataset


def test():
    '''
    各个函数的单元测试
    :return:
    '''
    dataset = np.array([
        [1., 2.1, 1.0],
        [2., 1.1, 1.0],
        [1.3, 1., -1.0],
        [1., 1., -1.0],
        [2., 1., 1.0]
    ])
    # print('最佳单层决策树：',stump_train(dataset, np.ones((dataset.shape[0],1))/dataset.shape[0], True))
    # DS_array, agg_predicts = adaboost_trainDS(dataset, 10, True)

    # 较大数据集上测试
    trainset = load_file_data('./dataset/horseColicTraining2.txt')
    DS_array,_ = adaboost_trainDS(trainset, 10, True)
    testset = load_file_data('./dataset/horseColicTest2.txt')
    labels = adaboost_classify(DS_array, testset)
    error_arr = np.ones(labels.shape)
    error_arr[labels == testset[:,-1].reshape(labels.shape)] = 0
    print('测试错误率：', np.sum(error_arr) / labels.shape[0])


if __name__ == '__main__':
    test()