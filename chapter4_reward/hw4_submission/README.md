# 作业4

## 4-1：奖励模型的训练实践

在代码执行完毕后，我们得到如下各规格的RND网络在test数据中产生的内在奖励：

`<img src="hw_4_1/RND_estimate_log.png">`

另外，打印 `train_data` 和 `test_data` 可以查看外在奖励的范围：

`<img src="hw_4_1/train_test_reward.png">`


可以看到，

`<ul>`

`<li>参与训练的外在奖励的范围是0~1，内在奖励的范围是0~0.01。</li>`

`<li>预测网络的训练有收敛和平稳两个阶段。</li>`

`</ul>`


具体来看每一个规格的RND网络（尝试给出解释）：

`<ul>`

`<li>little：[32, 16]，mean接近于0，说明网络能力太差，完全没法辨别状态，不管输入什么状态，输出的特征都差不多。</li>`

`<li>small：[64, 64]，min，max，mean三个指标都远高于其他网络。由于max和min存在较大的差值，也许最终结合Actor使用时不一定效果会差。</li>`

`<li>standard：[128, 64]，三个指标都表现良好，min接近于0，且max与mean有较大的差值。</li>`

`<li>large：[256, 256]，三个指标的曲线挨得很近，导致奖励缺少区分度。min值略高于standard网络，说明一些常见的状态被过度区分了，即出现了过拟合。</li>`

`<li>very large：[512, 512]，过拟合程度继续提高，min值略高于large网络。</li>`

`</ul>`


小结：无法观察单独的某一项指标来判断是否存在欠拟合/过拟合。一个好的RND网络应该满足：（1）min趋于0（2）max与mean有足够大的差值。


## 4-2：应用实践——minigrid迷宫

训练日志：

`<img src="hw_4_2/cmd_log.png">`

`<img src="hw_4_2/wandb_log.png">`
