# pytorch_forward_forward_zh
中文注释版 of forward-forward (FF) training algorithm - an alternative to back-propagation

原理可以参考中文翻译：

https://www.163.com/dy/article/HNLP6E5N0511831M.html

详细注释，请看 main.py 文件，本仓库fork于 https://github.com/mohammadpz/pytorch_forward_forward 感谢他们的付出，so cool ！！！

---

Below is my understanding of the FF algorithm presented at [Geoffrey Hinton's talk at NeurIPS 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf).\
The conventional backprop computes the gradients by successive applications of the chain rule, from the objective function to the parameters. FF, however, computes the gradients locally with a local objective function, so there is no need to backpropagate the errors.

![](./imgs/BP_vs_FF.png)

The local objective function is designed to push a layer's output to values larger than a threshold for positive samples and to values smaller than a threshold for negative samples.

A positive sample $s$ is a real datapoint with a large $P(s)$ under the training distribution.\
A negative sample $s'$ is a fake datapoint with a small $P(s')$ under the training distribution.

![](./imgs/layer.png)

Among the many ways of generating the positive/negative samples, for MNIST, we have:\
Positive sample $s = merge(x, y)$, the image and its label\
Negative sample $s' = merge(x, y_{random})$, the image and a random label

![](./imgs/pos_neg.png)

After training all the layers, to make a prediction for a test image $x$, we find the pair $s = (x, y)$ for all $0 \leq y < 10$ that maximizes the network's overall activation.

With this implementation, the training and test accuracy on MNIST are:
```python
> python main.py
train error: 0.06754004955291748
test error: 0.06840002536773682
```
