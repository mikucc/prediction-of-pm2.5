import numpy as np
import pandas as pd
def dataprocess(od):
    train_x,train_y=[],[]
    od=od.replace(['NR'],[0.0])#降雨量中的NR代表没下雨，换成0表示没下雨
    datao=np.array(od).astype(float)#转换数据类型
    for i in range(0,4320,18):#一共4320项数据，其中每18个为一组
        for j in range(15):#24小时每9小时一组数据
            mat=datao[i:i+18,j:j+9]#取出数据中的输入数据
            label=datao[i+9,j+9]#取出数据中的输出数据
            train_x.append(mat)
            train_y.append(label)
    t_x=np.array(train_x)
    t_y=np.array(train_y)
    #print(t_x.shape,t_y.shape)
    return t_x,t_y
def train(x,y,times):
    bias=0#偏置值初始化
    weights=np.ones((18,9))#每组数据9小时18个指标，一共162个权重项
    learning_rate=2.5#全局学习量
    regularization_coefficient=0.01#正则项系数
    b_gradients=0;#偏置项梯度累积量
    w_gradients=np.zeros((18,9))#权重项梯度累计变量
    for i in range(times+1):
        b_gradient=0
        w_gradient=np.zeros((18,9))
        for j in range(3200):
            temp=2*(y[j]-np.sum(np.multiply(x[j,:,:],weights))-bias)
            b_gradient+=temp*(-1)
            for sb in range(18):
                for dsb in range(9):
                    w_gradient[sb][dsb]+=temp*(-x[j,sb,dsb])+2*regularization_coefficient*weights[sb][dsb]
        #AdaGrad算法
        b_gradients+=b_gradient**2
        w_gradients+=w_gradient**2
        bias-=(learning_rate/b_gradients**0.5)*b_gradient
        weights-=(learning_rate/w_gradients**0.5)*w_gradient
        if i%50 == 0:#每50轮输出一次loss
            loss=0
            for j in range(3200):
                loss+=(y[j]-np.sum(np.multiply(x[j,:18,:9],weights))-bias)**2
            print('after {} times,the loss on train data is:'.format(i),loss/3200)
    return weights,bias
def validata(x_ver,y_ver,w,b):#验证数据集
    loss=0
    for i in range(400):
        loss+=(y_ver[i]-np.sum(np.multiply(x_ver[i,:,:],w))-b)**2
    return loss/400
def main():
    od=pd.read_csv('train.csv',usecols=range(3,27),encoding='gb18030')#读取数据，范围为3到27列，全部行
    x_t,y_t=dataprocess(od)#加工数据
    x,y=x_t[0:3200],y_t[0:3200]#取3600组中的前3200组进行训练
    x_ver,y_ver=x_t[3200:3600],y_t[3200:3600]#后400组进行验证
    times=1000#训练轮数
    w,b=train(x,y,times)#训练模型，w,b分别为权重项，和偏置系数
    loss=validata(x_ver,y_ver,w,b)
    print('the loss on verify data is:',loss)
main()
    