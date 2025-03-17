import numpy as np
import matplotlib.pyplot as plt


def naive_grad(x, mu):  # 其中x是从高斯分布中采样到的data_size个样本
    # the naive gradient to mu
    # \nabla_\mu \mathbb{E}_q[x^2] = \mathbb{E}_q[x^2(x-\mu)]
    return np.mean(x ** 2 * (x - mu))


def reparam_grad(eps, mu):
    #### You need to finish the reparameterization gradient to mu here ####
    return np.mean(2*(eps + mu))


def main():
    data_size_list = [10, 100, 500, 1000, 5000]
    sample_num = 100
    mu, sigma = 2.0, 1.0
    # variance of the gradient to mu
    var1 = np.zeros(len(data_size_list))
    var2 = np.zeros(len(data_size_list))
    for i, data_size in enumerate(data_size_list):
        estimation1 = np.zeros(sample_num)
        estimation2 = np.zeros(sample_num)
        for n in range(sample_num):  # 每次sample，从高斯分布N(mu, sigma)中采样data_size个样本
            # 1.naive method
            x = np.random.normal(mu, sigma, size=(data_size,))  # 这个data_size是指数据集的规模么？
            estimation1[n] = naive_grad(x, mu)
            # 2.reparameterization method
            eps = np.random.normal(0.0, 1.0, size=(data_size,))
            x = eps * sigma + mu
            estimation2[n] = reparam_grad(eps, mu)
        var1[i] = np.var(estimation1)
        var2[i] = np.var(estimation2)
    print('naive grad variance: {}'.format(var1))
    print('reparameterization grad variance: {}'.format(var2))
    # plot figure
    index = [_ for _ in range(len(data_size_list))]
    plt.plot(index, var1)
    plt.plot(index, var2)
    plt.xticks(index, data_size_list)
    plt.legend(['naive', 'reparam'])
    plt.savefig('reparam.png')
    plt.show()


if __name__ == "__main__":
    main()
