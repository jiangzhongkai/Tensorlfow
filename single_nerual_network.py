#用tensorflow 来实现简单的单层神经网络

#1.导入必要的编程库
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets   #数据集导入

#2.加载数据集,开始一个计算图会话

iris=datasets.load_iris()  #加载
x_vals=np.array([x[0:3] for x in iris.data])
y_vals=np.array([x[3] for x in iris.data])
sess=tf.Session()  #开始一个计算图的会话

#3.数据集较小，设置一个种子使得返回结果可复现
seed=2
tf.set_random_seed(seed)
np.random.seed(seed)

#4.准备数据集   训练集80%   测试集是20%
train_indices=np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace=False)
test_indices=np.array(list(set(range(len(x_vals)))-set(train_indices)))

x_vals_train=x_vals[train_indices]
x_vals_test=x_vals[test_indices]
y_vals_train=y_vals[train_indices]
y_vals_test=y_vals[test_indices]

def normalize_cols(m):
    col_max=m.max(axis=0)
    col_min=m.min(axis=0)
    return (m-col_min)/(col_max-col_min)
x_vals_train=np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test=np.nan_to_num(normalize_cols(x_vals_test))

#5.为数据集和目标值声明批量大小和占位符
batch_size=50
x_data=tf.placeholder(shape=[None,3],dtype=tf.float32)
y_target=tf.placeholder(shape=[None,1],dtype=tf.float32)

#6.声明有合适的模型变量   ,任意设置隐藏层的大小
hidden_layer_nodes=5
A1=tf.Variable(tf.random_normal(shape=[3,hidden_layer_nodes]))
b1=tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
A2=tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,1]))
b2=tf.Variable(tf.random_normal(shape=[1]))

#7.分两步声明训练模型
#  第一步：创建一个隐藏层输出
# 第二步:创建训练模型的最后输出

hidden_output=tf.nn.relu(tf.add(tf.matmul(x_data,A1),b1))   #f(A1*x_data+b1)
final_output=tf.nn.relu(tf.add(tf.matmul(hidden_output,A2),b2))

#8.定义均方误差作为损失函数
loss=tf.reduce_mean(tf.square(y_target-final_output))   #批量训练取平均损失来进行梯度

#9.声明优化算法，初始化模型变量
my_opt=tf.train.GradientDescentOptimizer(0.01)   #学习率是0.01
train_step=my_opt.minimize(loss)   #最小化误差
init=tf.global_variables_initializer()  # 初始化模型变量
sess.run(init)

#10.遍历迭代模型
#first we initialize the loss vectors for storage
loss_vec=[]
test_loss=[]
for i in range(500):  #迭代500次
    #first we select a random set ofindices for the batch
    rand_index=np.random.choice(len(x_vals_train),size=batch_size)
    #then we select the training values
    rand_x=x_vals_train[rand_index]
    rand_y=np.transpose([y_vals_train[rand_index]])
    #now we run thr train_step
    sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
    #we save the training loss
    temp_loss=sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})
    loss_vec.append(np.sqrt(temp_loss))
    #Finally,we run the test_set loss and save it
    test_temp_loss=sess.run(loss,feed_dict={x_data:x_vals_test,y_target:np.transpose([y_vals_test])})
    test_loss.append(np.sqrt(test_temp_loss))
    if (i+1)% 50 ==0:
        print('Generation:'+str(i+1)+'.Loss='+str(temp_loss))

#11.使用matplotlib绘制损失函数
plt.plot(loss_vec,'k-',label='Train Loss')
plt.plot(test_loss,'r--',label='Test Loss')
plt.title('Loss(MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')   #
plt.show()




