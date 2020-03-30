# 基于深度学习的对话系统
### 1. 前言

小时候看钢之炼金术师，印象最深的就是瓶中小人这个角色，不论是它的诞生，还是它想成为神、成为真理的追求都是那么的动人，这也是后来我开始学习深度学习、学习自然语言处理的原因之一，如果能够构建出一个和人类没有区别的对话系统，那该有多美妙。

除此之外，我一直觉得人类的语言是最神奇的东西，我们能够用语言交流，能够用语言记录，用语言表达知识，甚至从现有的知识中推导出新的知识，这一切都是基于语言进行的。可是人类的智慧和生命终究是有上限的，而机器没有这些限制，假如有一天，我们能够让机器，用它们的语言表达知识，推导新的知识，再转化成人类能理解的语言，我们再去验证，这绝对会是人类史上最大的突破。

因此，这里主要讨论一下基于深度学习的对话系统，目前感觉网上还没有一个比较统一或者鲜明的路线，所以这也是我自己的探索过程，或者说一个学习记录。

<br/>

### 2. 前置知识

* 编程基础就不再详细说了，tensorflow、pytorch双修肯定不会错；

* 本项目是在[NLPBeginner]("https://github.com/JesseYule/NLPBeginner")的基础上写的，或者说在写这个项目之前，我学习的所有关于自然语言处理的知识都在NLPBeginner都说了，可以大致看看。

<br/>

### 3. 相关基础理论

下面的大纲是我花了很多时间整理出来的，因为我比较关注的是基于深度学习的对话系统，所以会省略一些基于规则和传统机器学习的方法，没有做到面面俱到，当然大体的方向上应该是比较完整的，方便大家宏观地了解学习对话系统到底需要先了解什么理论概念。

#### 3.1 自然语言理解

* 语义表示

  * Distributional semantics（分布语义表示）

    * word2vec、glove

  * Model-theoretic semantics（模型论语义表示）

  * Frame semantics（框架语义表示）

* 意图检测（分类）

  * 基于规则的方法

    * CFG

  * 基于传统机器学习

       * SVM

  * 基于深度学习

       * CNN、RNN、LSTM

* 语义槽填充（slot filling）

  * 序列标注方法

    * 条件随机场

    * 隐马尔可夫模型（HMM）

    * CNN、RNN

#### 3.2 对话管理

关于对话管理，我选取了几篇经典论文，值得一看。

* Dialogue State Tracking

  * [Dialog State Tracking Challenge (DSTC)](https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/)

  * [Neural Belief Tracker: Data-Driven Dialogue State Tracking](https://arxiv.org/abs/1606.03777)

* Dialogue policy learning

  * [A Network-based End-to-End Trainable Task-oriented Dialogue System](https://arxiv.org/abs/1604.04562)

#### 3.3 自然语言生成

关于自然语言生成，虽然也有不少传统pipeline方法，但目前我觉得比较主流的还是seq2seq这类深度学习模型，当然如果遇到经典的值得学习的传统模型我也会补充上去。

* 基于深度学习

  * seq2seq

  * transformer

  * bert

 <br/>

### 4. 对话系统

* 任务导向型对话系统

  * pipeline方法

    * 主要由自然语言理解、对话管理、自然语言生成构成

  * end-to-end方法

    * seq2seq、transformer、bert

* 非任务导向型对话系统

  * 生成模型

    * Neural Generative Models

  * 基于检索的方法

    * 单轮回复匹配

    * 多轮回复匹配

  * 混合方法

 <br/>

### 5. 当前不足

- Swift Warm-Up：通常来说，对话数据的规模是较大的，然而对于特定领域而言，对话数据往往较少，在实际的对话工程中，在新领域的预热阶段，仍然需要依靠传统的流水线技术。

- Deep Understanding： 目前基于神经网络的对话系统主要依赖于大量不同类型的有标注数据、结构化的知识库和对话数据。这导致回复缺乏多样性，而且有时是没有意义的。因此，对话智能体应当通过对语言和现实世界的深度理解来更加有效地学习。比如，对话智能体能够从人的指导中学习，摆脱反复的训练。由于互联网上有大量的知识，如果对话智能体更聪明一些，就够利用这种非结构化的知识资源来理解。最后但依然很重要的一点，对话智能体应该能够做出合理的推论，找到新的东西，分享跨领域的知识，而不是像鹦鹉学舌。

- Privacy Protection：当下，对话系统服务的对象越来越多，而往往很多人使用的是同一个对话系统。对话系统会无意间存储一些用户的敏感数据。所以在构建更好的对话系统的同时，也应该注意保护用户的隐私。

 <br/>

### 6. 学习笔记

* [自然语言理解概述](https://blog.csdn.net/jesseyule/article/details/104929582)

* [意图识别与语义槽填充](https://blog.csdn.net/jesseyule/article/details/105105886)

* [对话管理简述](https://blog.csdn.net/jesseyule/article/details/105166348)

* [DST与Neural Belief Tracker](https://blog.csdn.net/jesseyule/article/details/105167212)

* [自然语言生成简述](https://blog.csdn.net/jesseyule/article/details/105201148)

 <br/>

### 7. 对话系统实例

#### 7.1 基于seq2seq的自然语言生成模型（聊天机器人）

第一个例子选择seq2seq的聊天机器人的原因很简单，第一，模型简单容易实现容易理解；第二，其实看回对话系统，seq2seq本身可以构建非任务导向型系统，也可以构建end2end的任务导向型对话系统，在pipeline的任务导向型对话系统中还可以构建NLG模块，所以其实用途十分广泛；第三，理解了seq2seq的本质，也方便学习transformer和bert，从而一步步改进现有的模型。

这个模型没有word2vec、没有注意力机制，只是由几层LSTM构成的seq2seq模型，方便快速理解模型的结构和原理，训练之后也有可以接受的效果。


