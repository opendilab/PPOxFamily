## RND主要创新点
1、采用随机生成的固定网络作为目标网络，另一网络不断最小化与其的误差，从而达到评估观察的新颖性</br>
2、介绍了一种灵活地结合内在和外在奖励的方法

## RND结果分析
1、little_RND_net和small_RND_net版本，一个little指标都接近于0，一个small指标很高，模型的目标就是最小化MSE，但是small版本的MSE很高，感觉这两个版本都没有训练成功。</br>
2、standard_RND_net这个版本是最好的版本。</br>
3、large_RND_net和very_large_RND_net版本，很明显，reward_min值开始提高，min值就是代表网络开始过拟合，开始往训练集过度靠拢，所以这两个版本出现了过拟合的情况。</br>
