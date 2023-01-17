import numpy as np
import matplotlib.pyplot as plt


def naive_grad(x,mu):
    # (mean) [x^2 * (x-u)]
    return np.mean( x**2 * (x-mu) ) # 得分函数估计法求出来的，E关于mu的梯度

def reparam_grad(eps,mu):  # eps = x-u
    # 对重参数化后的式子进行关于mu求导  I_1 + I_2 = 2(eps+mu) + 0
    return np.mean( 2*(eps+mu) )



def main():
    data_size_list = [10,100,500,1000,5000]
    sample_num = 100
    mu,sigma = 2.0 , 1.0

    var1 = np.zeros(len(data_size_list))
    var2 = np.zeros(len(data_size_list))

    for i,data_size in enumerate(data_size_list):   # 10,100,500,1000,5000 五种数据规模，每种数据规模进行100次的采样试验
        estimation1 = np.zeros(sample_num)        # 记录100次试验中期望的分布情况
        estimation2 = np.zeros(sample_num)
        for n in range(sample_num):
            #1.原始梯度计算方法
            x = np.random.normal(2.0, 1.0, size=(data_size, )) #随机化一批数据
            estimation1[n] = naive_grad(x,mu)

            #2.重参数化的梯度计算方法
            eps = np.random.normal(0.0, 1.0, size=(data_size,)) # x = eps * sigma +mu
            estimation2[n] = reparam_grad(eps,mu)

        var1[i] = np.var(estimation1)   # 每完成100次的采样试验后，进行一次100个期望的方差求值。然后进行下一个数据规模的测试
        var2[i] = np.var(estimation2)



    #得出的结论，求期望时候的数据规模越大，多次重复试验后 多个期望的方差越小
    print("grad1 variance: {}".format(var1))
    print("grad2 variance: {}".format(var2))
    index = [_ for _ in range(len(data_size_list))]
    plt.plot(index,var1)
    plt.plot(index,var2)
    plt.xticks(index,data_size_list)
    plt.show()






if __name__ == "__main__":
    main()


