import tensorflow as tf

from chatbot.textdata import Batch


class ProjectionOp:

    """ Single layer perceptron
    Project input tensor on the output dimension
    """

    '''
    这个算子的作用是把输入的tensor通过线性变换转化为特定维度
    '''

    def __init__(self, shape, scope=None, dtype=None):
        """
        Args:
            shape: a tuple (input dim, output dim)
            scope (str): encapsulate variables
            dtype: the weights type
        """
        assert len(shape) == 2

        self.scope = scope

        # Projection on the keyboard
        with tf.variable_scope('weights_' + self.scope):
            self.W_t = tf.get_variable(
                'weights',
                shape,
                # initializer=tf.truncated_normal_initializer()  # TODO: Tune value (fct of input size: 1/sqrt(input_dim))
                dtype=dtype
            )
            self.b = tf.get_variable(
                'bias',
                shape[0],
                initializer=tf.constant_initializer(),
                dtype=dtype
            )
            self.W = tf.transpose(self.W_t)

    def getWeights(self):
        """ Convenience method for some tf arguments
        """
        return self.W, self.b

    def __call__(self, X):
        """ Project the output of the decoder into the vocabulary space
        Args:
            X (tf.Tensor): input value
        """
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b


class Model:
    """
    Implementation of a seq2seq model.
    """

    '''
    模型结构:
        Encoder/decoder
        2 LTSM layers
    '''

    def __init__(self, args, textData):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        print("Model creation...")

        self.textData = textData  # Keep a reference on the dataset
        self.args = args  # Keep track of the parameters of the model
        self.dtype = tf.float32

        # Placeholders
        self.encoderInputs = None
        self.decoderInputs = None  # Same that decoderTarget plus the <go>
        self.decoderTargets = None
        self.decoderWeights = None  # Adjust the learning to the target sentence size

        # Main operators
        self.lossFct = None
        self.optOp = None
        self.outputs = None  # Outputs of the network, list of probability for each words

        # Construct the graphs
        self.buildNetwork()

    def buildNetwork(self):
        """ Create the computational graph
        """

        # TODO: Create name_scopes (for better graph visualisation)
        # TODO: Use buckets (better perfs)

        # Parameters of sampled softmax (needed for attention mechanism and a large vocabulary size)

        '''
        这里说一下output_projection的作用
        
        output_projection在tf中是一个很重要的机制，对于处理large output vocabulary的问题来讲可以很好的节约空间。
        
        output_projection作为一个重要的参数在tf的seq2seq模型中传播，它的值有两个，
        第一个是None，第二个是一个tuple=(W,B)。下面对这两种情况进行说明：
        
        1. output_projection=None
        
            None代表着不采用output_projection机制，此时从tf给的seq2seq模型中的输出结果
            的维度是output=[batch_size , num_decoder_symbols]。
    
            batch_size是深度学习中训练数据的一种方式，每次选取一个小的batch的数据去训练，然后计算这个batch的损失函数值，
            模型的训练目标让每一个batch的损失函数值逐渐减小。该方法是从SGD和GD中选取的一个折中的方法。
    
            Num_decoder_symbols指decoder的输入的特征数。比如在英法翻译中，法语作为目标语言的输入，
            num_decoder_symbols是选取的法语特征的单词数，即法语词表的数值大小。
    
            在英法翻译例子代码中batch_size=64, num_decoder_symbols=40000。如果不采用output_projection,
            每一个时间步的输出的维度为：output=[64,40000]，需要耗费很大的存储空间。

        2. output_projection=(W,B)
        
            W是一个权重权矩阵，W的维度是[size , num_decoder_symbols]，B是一个偏置向量，B的维度是[num_decoder_symbols]。
            
            采用output_projection=(W,B)时，tf给的seq2seq模型中的输出结果的维度是 output=[batch_size , size]。
            
            在英法翻译例子代码中size=1024,batch_size=64, num_decoder_symbols=40000
            
            
        output_projection的核心思想是：
        
        在模型训练时，当num_decoder_symbols=40000时，值很大，不便于存储和计算，用一个较小的值size=1024来代替。
        这样所需要耗费的存储资源就会少很多，计算速度也会提升。因此在英法翻译例子代码中采用了这种方式。
        当模型测试时，再将output=[batch_size, size]和W=[size,num_decoder_symbols]做矩阵乘，再加上偏置B，
        得到最后的输出final_output=[batch_size, num_decoder_symbols]，
        该过程相当于是做一个维度映射，size映射到num_decoder_symbols。
        最后得到的final_output是每一个词在目标语言词表上的概率分布。

        '''

        outputProjection = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if 0 < self.args.softmaxSamples < self.textData.getVocabularySize():

            outputProjection = ProjectionOp(
                (self.textData.getVocabularySize(), self.args.hiddenSize),
                scope='softmax_projection',
                dtype=self.dtype
            )

            def sampledSoftmax(labels, inputs):
                labels = tf.reshape(labels, [-1, 1])  # Add one dimension (nb of true classes, here 1)

                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                localWt = tf.cast(outputProjection.W_t, tf.float32)
                localB = tf.cast(outputProjection.b, tf.float32)
                localInputs = tf.cast(inputs, tf.float32)

                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        localWt,  # Should have shape [num_classes, dim]
                        localB,
                        labels,
                        localInputs,
                        self.args.softmaxSamples,  # The number of classes to randomly sample per batch
                        self.textData.getVocabularySize()),  # The number of classes
                    self.dtype)

        # Creation of the rnn cell
        def create_rnn_cell():

            # 基本的LSTM单元，这里只有一个参数，表示输出神经元数量
            encoDecoCell = tf.contrib.rnn.BasicLSTMCell(  # Or GRUCell, LSTMCell(args.hiddenSize)
                self.args.hiddenSize,
            )

            # 非测试阶段添加dropout层
            if not self.args.test:  # TODO: Should use a placeholder instead
                encoDecoCell = tf.contrib.rnn.DropoutWrapper(
                    encoDecoCell,
                    input_keep_prob=1.0,
                    output_keep_prob=self.args.dropout
                )
            return encoDecoCell

        # 一个encoder-decoder结构包含了多个rnn_cell
        # 所以说到底在这里一个encoder-decoder就只是一堆LSTM
        encoDecoCell = tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell() for _ in range(self.args.numLayers)],
        )

        # Network input (placeholders)

        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs = [tf.placeholder(tf.int32, [None, ]) for _ in
                                  range(self.args.maxLengthEnco)]  # Batch size * sequence length * input dim

        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs = [tf.placeholder(tf.int32, [None, ], name='inputs') for _ in
                                  range(self.args.maxLengthDeco)]  # Same sentence length for input and output (Right ?)
            self.decoderTargets = [tf.placeholder(tf.int32, [None, ], name='targets') for _ in
                                   range(self.args.maxLengthDeco)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in
                                   range(self.args.maxLengthDeco)]

        # Define the network
        # Here we use an embedding model, it takes integer as input and convert them into word vector for
        # better word representation

        # This model first embeds encoder_inputs by a newly created
        # embedding (of shape [num_encoder_symbols x input_size]).
        # Then it runs an RNN to encode embedded encoder_inputs into
        # a state vector. Next, it embeds decoder_inputs by another
        # newly created embedding (of shape [num_decoder_symbols x input_size]).
        # Then it runs RNN decoder, initialized with the
        # last encoder state, on embedded decoder_inputs.

        '''
        
        https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py

        Args:
            encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
            decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
            
            cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
            
            num_encoder_symbols: Integer; number of symbols on the encoder side.
            num_decoder_symbols: Integer; number of symbols on the decoder side.
            
            embedding_size: Integer, the length of the embedding vector for each symbol.
            
            output_projection: None or a pair (W, B) of output projection weights and
                biases; W has shape [output_size x num_decoder_symbols] and B has
                shape [num_decoder_symbols]; if provided and feed_previous=True, each
                fed previous output will first be multiplied by W and added B.
                
            feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
                of decoder_inputs will be used (the "GO" symbol), and all other decoder
                inputs will be taken from previous outputs (as in embedding_rnn_decoder).
                If False, decoder_inputs are used as given (the standard decoder case).
                
            dtype: The dtype of the initial state for both the encoder and encoder
                rnn cells (default: tf.float32).
                
            scope: VariableScope for the created subgraph; defaults to
                "embedding_rnn_seq2seq"
                
        Returns:
            A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors. The
                output is of shape [batch_size x cell.output_size] when
                output_projection is not None (and represents the dense representation
                of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
                when output_projection is None.
            state: The state of each decoder cell in each time-step. This is a list
                with length len(decoder_inputs) -- one item for each time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].

        
        这里详细讲解一下这个tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq
        
        首先encoder_input和decoder_input只有两个维度，一个是batch，一个是一维tensor表示一个句子
        也就是说，这个函数只是把输入的词用整数表示，比如[1,2,3,0,0]，这就是一个有五个词的句子
        
        num_encoder_symbols表示encoder_inputs中词的数量，因为我们并没有对词做embedding，所以这里其实用序号表示每个词

        在训练阶段，我们除了输入原始序列作为encoder_input，还需要输入target序列作为decoder_input(feed_previous=False)，
        这样会使模型更准确，但是在测试阶段，我们只能把encder的output作为decoder的input(feed_previous=True)，
        这些由参数中的feed_previous控制
        
        所谓target就是正确的输出序列，因为decoder的输入是语义向量state和上一时刻decoder的输出，训练的时候就直接用上一时刻的正确输出
        作为输入，这样训练的模型就robust
        
        '''

        decoderOutputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            encoder_inputs=self.encoderInputs,  # List<[batch=?, inputDim=1]>, list of size args.maxLength
            decoder_inputs=self.decoderInputs,  # For training, we force the correct output (feed_previous=False)
            cell=encoDecoCell,
            num_encoder_symbols=self.textData.getVocabularySize(),
            num_decoder_symbols=self.textData.getVocabularySize(),  # Both encoder and decoder have the same number of class
            embedding_size=self.args.embeddingSize,  # Dimension of each word
            output_projection=outputProjection.getWeights() if outputProjection else None,
            feed_previous=bool(self.args.test)
            # When we test (self.args.test), we use previous output as next input (feed_previous)
        )

        # TODO: When the LSTM hidden size is too big, we should project the LSTM output into a smaller space (4086 => 2046): Should speed up
        # training and reduce memory usage. Other solution, use sampling softmax

        # For testing only
        if self.args.test:
            if not outputProjection:
                self.outputs = decoderOutputs
            else:
                self.outputs = [outputProjection(output) for output in decoderOutputs]

            # TODO: Attach a summary to visualize the output

        # For training only
        else:

            '''
            这里说一下tf.contrib.legacy_seq2seq.sequence_loss
            
            Weighted cross-entropy loss for a sequence of logits, batch-collapsed.
            
            加权的交叉熵损失，不是普通的交叉熵，有意义的值给予较大的权重，这部分应该是损失函数的主要组成
            没意义的权重就小点，我的理解是padding、开始终结符之类的就没什么意义

            '''

            # Finally, we define the loss function
            self.lossFct = tf.contrib.legacy_seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,

                # 我觉得这个参数比较奇怪，和原函数的参数（average_across_timesteps）对应不上，原参数应该应该输入bool值
                # sequence_loss(logits,
                #               targets,
                #               weights,
                #               average_across_timesteps=True,
                #               average_across_batch=True,
                #               softmax_loss_function=None,
                #               name=None):
                # 我尝试去掉这个参数进行训练，结果正常

                # self.textData.getVocabularySize(),

                softmax_loss_function=sampledSoftmax if outputProjection else None  # If None, use default SoftMax
            )

            tf.summary.scalar('loss', self.lossFct)  # Keep track of the cost

            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.lossFct)

    def step(self, batch):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
        Return:
            (ops), dict: A tuple of the (training, loss) operators or (outputs,) in testing mode with the associated feed dictionary
        """

        # Feed the dictionary
        feedDict = {}
        ops = None

        if not self.args.test:  # Training
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]
            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoderInputs[i]] = batch.decoderSeqs[i]
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]

            ops = (self.optOp, self.lossFct)
        else:  # Testing (batchSize == 1)
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]] = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]] = [self.textData.goToken]

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feedDict
