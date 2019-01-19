import numpy as np


def stump_classify(testset, dim, thresh_val, inequal):
    '''
    单层决策树分类函数
    :param testset : [np.array(m,n)] 训练集
    :param dim : [int] 特征的索引号
    :param thresh_val : [float] 阈值
    :param inequal : [str] 假设
    :return : [np.array(m,1)] 分类结果
    '''
    m = testset.shape[0]
    predict_class = np.ones((m,1))
    if inequal == 'lt':  # 假设：大于阈值为正样本
        predict_class[testset[:,dim] <= thresh_val] = -1.0
    else:  # 假设：大于阈值为负样本
        predict_class[testset[:,dim] > thresh_val] = -1.0
    return predict_class


def build_stump(trainset, D, is_print=False):
    '''
    建立最佳单层决策树
    :param trainset : [np.array(m,n+1)]训练集
    :param D : [np.array(m,1)]样本权重矩阵
    :param is_print : [boolean]是否打印每次迭代结果
    :return : [dict]best_stump 最佳单层决策树, [float]min_error 最小分类器错误率, [np.array(m,1)]分类器的分类结果
    '''
    m, n = trainset.shape
    n = n - 1  # 因为trainset包含了标签
    labels = trainset[:, -1].reshape((m, 1))  # 标签向量
    assert (D.shape == (m, 1))
    min_error = float('inf')  # 最小错误率，并初始化为正无穷
    step_num = 10  # 步长,即分支数
    best_stump = {}  # 最佳单层决策树
    best_predict = np.zeros((m,1))  # 最佳单层决策树对训练集产生的分类结果
    for i in range(n):  # 遍历每个特征
        feature_min = trainset[:, i].min()  # 训练集中第 i 个特征的最小值
        feature_max = trainset[:, i].max()  # 训练集中第 i 个特征的最大值
        step_size = (feature_max - feature_min) / float(step_num)
        for j in range(-1, step_num + 1):
            thresh_val = feature_min + step_size * j  # 阈值
            for inequal in ['lt', 'gt']:  # 遍历不等号，lt 是 less than，gt 是 great than
                predict_labels = stump_classify(trainset, i, thresh_val, inequal)  # 单层决策树分类函数，与下面5行代码等价。函数定义在下面.
                error_array = np.ones(labels.shape)
                error_array[predict_labels == labels] = 0
                # weight_error = float(np.dot(D.T, error_array) / m)  # 加上样本权重后的错误率
                weight_error = np.dot(D.T, error_array)  # 加上样本权重后的错误率
                if is_print:
                    print('split: 第 {} 个特征, 阈值 {}, 假设 {}, 加权错误率 {}'.format(i, thresh_val, inequal, weight_error))
                if weight_error < min_error:
                    min_error = weight_error
                    best_stump['dim'] = i
                    best_stump['thresh_val'] = thresh_val
                    best_stump['inequal'] = inequal
                    best_predict = predict_labels.copy()

    return best_stump, min_error, best_predict.reshape((m, 1))


def adaBoost_trainDS(trainset, iter_num=40, is_print=False):
    '''
    AdaBoost 分类器训练，其中 DS 代表弱分类器为单层决策树
    :param trainset : [np.array]训练集,维度为(m,n+1)
    :param iter_num : [int]最大迭代次数
    :param is_print : [boolean] 是否打印中间结果
    :return :
    '''
    weak_classifiers = []  # 单层决策树数组
    m = trainset.shape[0]
    D = np.ones((m, 1)) / m # 初始化样本权重
    agg_predict = np.zeros(D.shape)  # 最终分类器的分类结果，等于每个弱分类器分类结果乘以其权重的总和。
    for i in range(iter_num):
        print('第 {} 次迭代'.format(i + 1))
        # 训练最佳单层决策树
        best_stump, min_error, best_predict = build_stump(trainset, D, False)

        # 更新分类器权重 alpha 并将最佳单层决策树加入到单层决策树数组
        alpha = float(0.5 * np.log((1.0 - min_error) / max(min_error, 1e-16)))  # 更新分类器权重，同时需要防止分母过小变成零然后发生零溢出
        best_stump['alpha'] = alpha
        weak_classifiers.append(best_stump)  # 将最佳单层决策树加入到单层决策树数组

        # 计算新的 D
        labels = trainset[:, -1].reshape(D.shape)  # 标签向量
        expon = np.multiply(-1 * alpha, labels * best_predict)  # 公式中的分子中 e 的指数
        D = D * np.exp(expon)  # 分子
        D = D / np.sum(D)

        # 求最终分类器的分类结果
        agg_predict += alpha * best_predict

        # 求最终分类器的错误率
        temp = np.ones(agg_predict.shape)
        temp[np.sign(agg_predict) == labels] = 0
        error_rate = float(np.sum(temp)) / m

        if is_print:
            print('当前训练集样本权重D : ', D.T)
            print('当前弱分类器的分类结果 ：', best_predict.T)
            print('当前弱分类器的错误率 : ', min_error)
            print('最终分类器的分类结果 : ', agg_predict.T)
        print('最终分类器的错误率 ： ', error_rate)
        print('=======================================')

        if error_rate == 0.0:
            break

    return weak_classifiers


def adaboost_classify(testset, classifiers, is_print=False):
    '''
    AdaBoost 分类函数
    :param testset : [np.array(m,n)] 测试集
    :param classifiers : [list] AdaBoost 分类器,即弱分类器数组
    :param is_print : [boolean] 是否打印中间结果
    :return : [np.array(m,1)] 分类结果
    '''
    m = testset.shape[0]
    predict_class = np.zeros((m,1))
    for c in classifiers:  # 遍历每个弱分类器
        temp_class = stump_classify(testset, c['dim'], c['thresh_val'], c['inequal'])  # 每个弱分类器的分类结果
        predict_class += c['alpha'] * temp_class  # 加权求和
        if is_print:
            print('当前 AdaBoost 分类中间结果 ：', predict_class.T)

    return np.sign(predict_class)

# print('[[5,5],[0,0]] 标签分别为 1，-1 \n AdaBoost 分类结果是', adaboost_classify(np.array([[5,5],[0,0]]), weak_classifiers, True).T)



def load_dataset(filename):
    '''
    加载数据
    :param filename : [str] 完整的文件名
    :return : [np.array(m,n+1)] 数据集
    '''
    fr = open(filename)
    n = len(fr.readline().split('\t')) - 1
    m = len(fr.readlines()) + 1  # 因为上一行代码读走了一行
    dataset = np.ones((m,n+1))
    index = 0
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip().split('\t')
        dataset[index,:] = np.array(line)
        # dataset[index, -1] = np.where(dataset[index, -1] == 0, -1., 1.)
        index += 1
    return dataset


def create_simple_dataset():
    '''
    创建简单的训练集
    :return : [np.array]训练集，维度是(m,n+1)
    '''
    dataset = np.array([
        [1., 2.1, 1.0],
        [2., 1.1, 1.0],
        [1.3, 1., -1.0],
        [1., 1., -1.0],
        [2., 1., 1.0]
    ])
    return dataset


def test():
    # weak_classifiers = adaBoost_trainDS(create_simple_dataset(), is_print=True)
    # print(weak_classifiers)
    # print('[[5,5],[0,0]] 标签分别为 1，-1 \n AdaBoost 分类结果是',
    #       adaboost_classify(np.array([[5, 5], [0, 0]]), weak_classifiers, True).T)

    trainset = load_dataset('./dataset/horseColicTraining2.txt')

    classifiers = adaBoost_trainDS(trainset=trainset, iter_num=50, is_print=False)
    # print(classifiers)
    testset = load_dataset('./dataset/horseColicTest2.txt')
    predic_class = adaboost_classify(testset=testset, classifiers=classifiers)
    error_num = np.ones(predic_class.shape)
    print(error_num[predic_class != testset[:, -1].reshape((testset.shape[0], 1))].sum() / testset.shape[0])

if __name__ == '__main__':
    test()