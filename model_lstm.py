import tensorflow as tf
import numpy as np
import pandas as pd
import Data_Processor

class LSTM_net():
    def __init__(self,sess,stock_count,
                 lstm_size,
                 num_layers,
                 num_steps,
                 input_size,
                 embed_size,
                 train_ratio,
                 logs_dir='',
                 plots_dir=''):
        self.sess = sess
        self.stock_count = stock_count
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size
        self.embed_size = embed_size or -1
        self.logs_dir = logs_dir
        self.plots_dir = plots_dir
        self.train_ratio=train_ratio
        self.graph()
    def graph(self):
        '''
        构建数据流图
        '''
        '''定义需要用到的占位符'''
        self.learning_rate = tf.placeholder(tf.float32,None,name='LearningRate')
        self.keep_prob = tf.placeholder(tf.float32,None,name='KeepProb')
        self.symbols_x = tf.placeholder(tf.int32,[None,self.num_steps],name='stock_labels_x')
        self.inputs = tf.placeholder(tf.float32,[None,self.num_steps,2],name='inputs')
        # 2 is for the change and vol,是因为有涨跌率和交易量两组数据
        self.targets = tf.placeholder(tf.float32,[None,1],name='targets')

        '''Embedding 层，用于将每只股票用embed_size大小的向量表示'''
        self.embed_matrix=tf.Variable(tf.random_uniform([self.stock_count,self.embed_size],minval=-0.2,maxval=0.2),name='embed_matrix')
        stock_label_embeds_x = tf.nn.embedding_lookup(self.embed_matrix, self.symbols_x)

        '''将输入和股票的Embed_size向量表示组合起来'''
        self.inputs_with_embeds = tf.concat([self.inputs,stock_label_embeds_x],axis=2)

        '''根据lstm_size、keep_prob、num_layers构建含Dropout包装器的LSTM神经网络'''
        def _create_one_cell():
            cell=tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.lstm_size,state_is_tuple=True),
                output_keep_prob=self.keep_prob
            )
            return cell
        cell=tf.nn.rnn_cell.MultiRNNCell([_create_one_cell() for i in range(self.num_layers)],state_is_tuple=True) if self.num_layers > 1 else _create_one_cell()

        '''获得LSTM网络的输出和状态'''
        val,_ = tf.nn.dynamic_rnn(cell,self.inputs_with_embeds,dtype=tf.float32)

        '''根据LSTM网络的输出计算出与target维度匹配的输出'''
        # Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
        # After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
        val = tf.transpose(val,[1,0,2])
        last = tf.gather(val,int(val.get_shape()[0])-1,name='last_lstm_output')
        weight=tf.Variable(tf.truncated_normal([self.lstm_size,1]))
        bias=tf.Variable(tf.constant(0.1,shape=[1]))
        self.prediction=tf.matmul(last,weight)+bias

        '''模型的代价函数和优化器'''
        self.loss=tf.reduce_mean(tf.square(self.prediction-self.targets))
        self.optimizer=tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self,max_epoch,init_learning_rate,decay_rate,decay_epoch,batch_ratio,keep_prob,interval,future):
        ### 获取数据，设定好batch_size与训练epoch,将数据feed进图开始训练和测试

        '''获取数据，初始化模型参数'''
        stock_code,stock_data,mean_fluctuation=Data_Processor.get_stocks(
            input_size=self.input_size,num_steps=self.num_steps,train_ratio=self.train_ratio,interval=interval,future=future)
        tf.global_variables_initializer().run()

        '''合并每只股票的测试样本为一个统一的测试集'''
        merge_test_x = []
        merge_test_y = []
        merge_test_labels_x = []
        for test_label, test_data in enumerate(stock_data):
            merge_test_x += list(test_data.test_x)
            merge_test_y += list(test_data.test_y)
            merge_test_labels_x += [[test_label] * self.num_steps] * len(test_data.test_x)
        test_feed_dic = {
            self.learning_rate: 0.0,
            self.keep_prob: 1.0,
            self.inputs: np.array(merge_test_x),
            self.targets: np.array(merge_test_y),
            self.symbols_x: np.array(merge_test_labels_x),
        }

        # 开始训练
        for epoch in range(max_epoch):

            '''每轮更新一次学习率'''
            learning_rate = init_learning_rate * (
                decay_rate ** max(float(epoch + 1 - decay_epoch), 0.0)
            )

            for label,data in enumerate(stock_data):
                # 准备训练集
                train_x = tf.placeholder(data.train_x.dtype,data.train_x.shape)
                train_y = tf.placeholder(data.train_y.dtype,data.train_y.shape)

                # 取batch_size个样本的训练集
                batch_size=int(len(data.train_x)*batch_ratio)
                dataset=tf.data.Dataset.from_tensor_slices((train_x,train_y))
                dataset=dataset.batch(batch_size)
                iterator=dataset.make_initializable_iterator()
                self.sess.run(iterator.initializer, feed_dict={train_x: data.train_x, train_y: data.train_y})
                next_batch = iterator.get_next()
                batch_x, batch_y = self.sess.run(next_batch)
                # 构建每个输入对应的标签
                batch_label_x = np.array([[label]*self.num_steps] * batch_x.shape[0])

                train_feed_dic = {
                    self.learning_rate:learning_rate,
                    self.symbols_x:batch_label_x,
                    #self.symbols_y:batch_label_y,
                    self.keep_prob:keep_prob,
                    self.inputs:batch_x,
                    self.targets:batch_y,
                }
                # 训练
                train_loss,optimizer=self.sess.run(
                    [self.loss,self.optimizer],train_feed_dic
                )
            print('After ', epoch, 'the train_loss: ', train_loss)
            # 测试
            test_pred, test_loss = self.sess.run([self.prediction, self.loss], test_feed_dic)
            print('After ',epoch,'the test_loss: ',test_loss)

        #最终再测试一次
        final_pred,final_loss=self.sess.run([self.prediction, self.loss], test_feed_dic)
        print('Final,the test_loss: ',final_loss)

        #预测数据和target分别存入txt方便观察
        np.savetxt('out\pred.txt',final_pred)
        np.savetxt('out\out.txt',merge_test_y)

        #保存一下模型
        Saver=tf.train.Saver()
        Saver.save(sess=self.sess,save_path='.\save\params')

        # 计算平均预测误差
        sum_error=0
        for i in range(final_pred.shape[0]):
            print('final_pred:',final_pred[i][0],' and the target:',merge_test_y[i][0])
            sum_error += (final_pred[i][0] - merge_test_y[i][0])
        mean_error=sum_error/final_pred.shape[0]

        print('所有股票涨幅的平均波动为：',mean_fluctuation)
        print('在测试集上对于涨跌趋势的预测的平均误差为：',mean_error)



def main():
    with tf.Session() as sess:
        lstm_model=LSTM_net(
            sess,
            stock_count=50,
            lstm_size=128,
            num_layers=1,
            num_steps=250,#一年250个交易日
            input_size=10,
            embed_size=3,
            train_ratio=0.9,
            logs_dir='./logs',
            plots_dir='. /plots'
        )
        lstm_model.train(max_epoch=30,
                         init_learning_rate=0.001,
                         decay_rate=0.98,
                         decay_epoch=10,
                         batch_ratio=0.8,
                         keep_prob=0.8,
                         interval=30,
                         future=30
                         )
main()