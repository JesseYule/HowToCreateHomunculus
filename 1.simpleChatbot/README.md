### version 1——基于seq2seq实现的非任务型对话系统



原始项目：[DeepQA](https://github.com/Conchylicultor/DeepQA)



主要基于seq2seq模型实现的非任务型对话系统，和原始项目相比减少了部分功能，只保留了模型核心模块，同时加上大量注释（主要针对模型，数据预处理没有怎么注释），方便理解模型如何处理数据实现对话功能，同时也确保了下载代码之后，只要安装了需要的库，就能直接训练直接出效果（数据都不需要下载），当然如果有什么问题也欢迎交流。



和原项目相比，主要删减了（主要删除了代码文件，没有对保留的文件代码做修改）：

* 只保留了cornell数据及预处理代码，其他数据都删除了
* 删除了docker相关代码文件
* 删除了部分对模型和训练影响不大的代码文件



本人的训练环境配置（可和原项目对比参考，比如原项目说用python3.5，我用3.7也没问题）：

* python 3.7
* tensorflow (使用NVIDIA T4训练)
* numpy
* CUDA 
* nltk (natural language toolkit for tokenized the sentences)
* tqdm (for the nice progression bars)



为了让nltk能顺利工作，需要下载相关数据：

```
python3 -m nltk.downloader punkt
```



如果想通过web interface进行交互，需要安装：

* django (1.10，和原项目相同)

* channels(1.1.8，原项目没指明版本，经过测试太新的版本不行，可以安装和我一样或者接近的版本)

* Redis (2.10.6，原项目没说明)

* asgi_redis (1.4.3，原项目指明最起码是1.0)

  

结果：

​	关于如何训练之类的细节可查看原项目的说明，最后再展示一下训练了一万多轮后模型的表现：

![Screen Shot 2020-03-22 at 10.29.41 am](image/2.png)

​	总的来说看着还行，起码说出来的语句也通顺，只用了几层LSTM构成的seq2seq，没用word2vec的embedding，没有注意力机制，能达到这个效果也算满意了。

​	但是使用web interface进行交互则完全不行，暂时也没空分析哪里出错了，反正在命令行交互是正常的就说明模型是没问题的，可能是web interface的代码在调用训练好的模型的时候失败了之类的问题，之后有空回分析一下：

![screenshot3](image/screenshot3.png)

