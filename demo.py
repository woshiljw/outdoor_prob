# !/usr/bin/python3
# Author:ljw
# @Time:

import numpy as np
import tensorflow as tf
from tensorflow import contrib
import math
import file_read

value_type=tf.float64#设定变量值的类型

'''
该部分为需要外部输入的固定值
'''
Tmatrix = tf.placeholder(shape=(3448,360),dtype=value_type)
SGroundTruth = tf.placeholder(shape=(3448,3),dtype=value_type)
rho_sigma = tf.placeholder(shape=(4,3),dtype=value_type)
rho_mu = tf.placeholder(shape=(4,3),dtype=value_type)
rho_ComponentProportion = tf.placeholder(shape=(1,4),dtype=value_type)
theta_sigma = tf.placeholder(shape=(5,10),dtype=value_type)
theta_mu = tf.placeholder(shape=(5,10),dtype=value_type)
theta_componentProportion = tf.placeholder(shape=(1,5),dtype=value_type)


class model(object):
    def __init__(self,x):
        '''
        定义变量，这些变量包含了模型所需的16维参数，将通过模型的多次迭代来对参数进行优化
        :param x: 一个16维的list用于对初始变量赋值
        '''
        self.zenithAngle = tf.Variable(x[0], dtype=value_type)
        self.azimuthAngle = tf.Variable(x[1], dtype=value_type)
        self.wsky = tf.Variable([[x[2],x[3],x[4]]], dtype=value_type)#1x3
        self.wsun = tf.Variable([[x[5],x[6],x[7]]], dtype=value_type)#1x3
        self.t = tf.Variable(x[8], dtype=value_type)
        self.K = tf.Variable(x[9], dtype=value_type)
        self.Rho = tf.Variable([[x[10],x[11],x[12]]], dtype=value_type)#1x3
        self.RhoGnd = tf.Variable([[x[13],x[14],x[15]]], dtype=value_type)#1x3
        self.nGnd = tf.constant([[0., 0., 0.], [0., 0., 0.], [1., 1., 1.]], dtype=tf.double)

    #天空光模型
    def light_sky(self):
        f, sundirection, w = self.psl()
        omigasun_x_omiga = tf.matmul(w, tf.transpose(sundirection))  # 360x1
        C = self.K / (2 * math.pi - 2 * math.pi * tf.exp(-2 * self.K))

        Iss = tf.matmul(f, self.wsky) + tf.matmul(C * tf.exp(self.K * (omigasun_x_omiga - 1)),self.wsun)
        return Iss

    # Preetham sky luminance model(主要用于天空光模型进行调用)
    def psl(self):
        def per(theta, gamma):
            # the Perez model
            a = 0.1787 * self.t - 1.4630
            b = -0.3554 * self.t + 0.4275
            c = -0.0227 * self.t + 5.3251
            d = 0.1206 * self.t - 2.5771
            e = -0.0670 * self.t + 0.3703
            return (1 + a * tf.exp(b / tf.cos(theta))) * (1 + c * tf.exp(d * gamma) + e * tf.square(tf.cos(gamma)))

        #定义太阳方向
        sundirection=[[tf.sin(self.zenithAngle) * tf.cos(self.azimuthAngle),
                                -tf.sin(self.zenithAngle) * tf.sin(self.azimuthAngle),
                                tf.cos(self.zenithAngle)]]
        #在经度上取9个方位，在纬度上取40个方位，一共取得9x40个点
        lat = tf.constant(np.arange(0, np.pi / 2, np.pi / 18).reshape(9, 1))
        lng = tf.constant(np.arange(0, np.pi * 2, np.pi / 20).reshape(1, 40))


        w0=tf.reshape(tf.matmul(tf.sin(lat), tf.cos(lng)),[-1])
        w1=tf.reshape(-tf.matmul(tf.sin(lat), tf.sin(lng)),[-1])
        w2=tf.reshape(tf.matmul(tf.cos(lat), tf.ones_like(lng)),[-1])
        w = tf.stack([w0,w1,w2],axis=1)

        '''
        tf.reshape(tf.stack(, axis=2), (-1, 3))  # 360x3
        '''

        x = (4 / 9 - self.t / 120) * (math.pi - 2 * self.zenithAngle)
        Yz = (4.0453 * self.t - 4.9710) * tf.tan(x) - 0.2155 * self.t + 2.4192
        gamma = tf.acos(tf.matmul(w, tf.transpose(sundirection)))  # 360x1
        theta = tf.reshape(tf.matmul(lat, tf.ones_like(lng)), (-1, 1))  # 360x1
        return Yz * per(theta, gamma) / per(tf.zeros_like(theta), self.zenithAngle * tf.ones_like(gamma)), sundirection, w

    #地面光模型
    def light_gnd(self,Iss):
        lat = tf.constant(np.arange(0, np.pi / 2, np.pi / 18).reshape(9, 1), dtype=tf.double)
        lng = tf.constant(np.arange(0, np.pi * 2, np.pi / 20).reshape(1, 40), dtype=tf.double)

        w0 = tf.reshape(tf.matmul(tf.sin(lat), tf.cos(lng)), [-1])
        w1 = tf.reshape(-tf.matmul(tf.sin(lat), tf.sin(lng)), [-1])
        w2 = tf.reshape(tf.matmul(tf.cos(lat), tf.ones_like(lng)), [-1])
        w = tf.stack([w0, w1, w2], axis=1)

        T_gnd = tf.matmul(w, self.nGnd)  # 360x3
        rho_gnd = tf.matmul(np.ones((360, 1)), self.RhoGnd)  # 360x3
        return tf.multiply(tf.multiply(Iss, T_gnd), rho_gnd) / math.pi

    #着色模型
    def shade_model(self,I):
        r=tf.matmul(tf.matmul(tf.eye(3448, dtype=value_type) * self.Rho[0, 0], Tmatrix), tf.reshape(I[:, 0], (360, 1)))
        g=tf.matmul(tf.matmul(tf.eye(3448, dtype=value_type) * self.Rho[0, 1], Tmatrix), tf.reshape(I[:, 1], (360, 1)))
        b=tf.matmul(tf.matmul(tf.eye(3448, dtype=value_type) * self.Rho[0, 2], Tmatrix), tf.reshape(I[:, 2], (360, 1)))
        return tf.reshape(tf.stack([r, g, b], axis=1), (3448, 3))

    #rho的先验项
    def rho_priori(self):
        rho_sigmaDet = tf.reduce_prod(rho_sigma, axis=1)  # 计算累乘
        rho_formerCoefficient = rho_ComponentProportion / np.power(2 * math.pi, 1.5) / np.power(rho_sigmaDet, 0.5)
        rho_laterCoefficient = - 0.5 / rho_sigma[:, 0] * tf.pow(self.Rho[0,0] - rho_mu[:, 0], 2) \
                               - 0.5 / rho_sigma[:, 1] * tf.pow(self.Rho[0,1] - rho_mu[:, 1], 2) \
                               - 0.5 / rho_sigma[:, 2] * tf.pow(self.Rho[0,2] - rho_mu[:, 2], 2)
        RhoPriori = -0.5 * tf.log(tf.matmul(rho_formerCoefficient, tf.exp(tf.transpose([rho_laterCoefficient]))))
        return RhoPriori

    #theta的先验项
    def theta_priori(self):
        theta_sigmaDet = tf.reduce_prod(theta_sigma, axis=1)
        theta_formerCoefficient = theta_componentProportion[0] / tf.pow(2 * tf.to_double(math.pi), 5) / tf.pow(theta_sigmaDet,0.5)
        theta_laterCoefficient = - 0.5 / theta_sigma[:, 0] * tf.pow(self.zenithAngle - theta_mu[:, 0], 2) \
                                 - 0.5 / theta_sigma[:, 1] * tf.pow(self.azimuthAngle - theta_mu[:, 1], 2) \
                                 - 0.5 / theta_sigma[:, 2] * tf.pow(self.wsky[0,0] - theta_mu[:, 2], 2) \
                                 - 0.5 / theta_sigma[:, 3] * tf.pow(self.wsky[0,1] - theta_mu[:, 3], 2) \
                                 - 0.5 / theta_sigma[:, 4] * tf.pow(self.wsky[0,2] - theta_mu[:, 4], 2) \
                                 - 0.5 / theta_sigma[:, 5] * tf.pow(self.wsun[0,0] - theta_mu[:, 5], 2) \
                                 - 0.5 / theta_sigma[:, 6] * tf.pow(self.wsun[0,1] - theta_mu[:, 6], 2) \
                                 - 0.5 / theta_sigma[:, 7] * tf.pow(self.wsun[0,2] - theta_mu[:, 7], 2) \
                                 - 0.5 / theta_sigma[:, 8] * tf.pow(self.t - theta_mu[:, 8], 2) \
                                 - 0.5 / theta_sigma[:, 9] * tf.pow(self.K - theta_mu[:, 9], 2)

        thetaPriori = -0.5 * tf.log(tf.matmul([theta_formerCoefficient], tf.exp(tf.transpose([theta_laterCoefficient]))))
        return thetaPriori

    def revalue(self):
        tf.assign(self.zenithAngle,tf.clip_by_value(self.zenithAngle,0.,0.5*math.pi))
        tf.assign(self.azimuthAngle,tf.clip_by_value(self.azimuthAngle,-math.pi,math.pi))
        tf.assign(self.wsky,tf.clip_by_value(self.wsky,[[0.,0.,0.]],[[np.infty,np.infty,np.infty]]))
        tf.assign(self.wsun,tf.clip_by_value(self.wsun,[[0.,0.,0.]],[[np.infty,np.infty,np.infty]]))
        tf.assign(self.t,tf.clip_by_value(self.t,3,20))
        tf.assign(self.K,tf.clip_by_value(self.K,8,8192))
        tf.assign(self.Rho,tf.clip_by_value(self.Rho,[[0.,0.,0.]],[[np.infty,np.infty,np.infty]]))
        tf.assign(self.RhoGnd,tf.clip_by_value(self.RhoGnd,[[0.,0.,0.]],[[np.infty,np.infty,np.infty]]))


    def train(self):
        self.revalue()
        Iss=self.light_sky()
        Ignd=self.light_gnd(Iss)
        I=Iss+Ignd
        S_est=self.shade_model(I)
        error = tf.add(tf.square(S_est - SGroundTruth),1e-6)

        RhoPriori=self.rho_priori()
        ThetaPriori=self.theta_priori()
        func = (RhoPriori + ThetaPriori) * 3448 + tf.reduce_sum(tf.sqrt(error))

        return func

x0 = np.array([1, 1.2, 0.05, 0.05, 0.05, 20, 20, 20, 3, 2000, 0.17, 0.09, 0.08, 0.2, 0.2, 0.2], dtype='float32')
m=model(x0)
with tf.Session() as sess:
    cost = m.train()

    opt = tf.train.AdamOptimizer(0.01).minimize(cost)
    sess.run(tf.global_variables_initializer())


    for i in range(10000):

        _,cost1=sess.run([opt,cost],feed_dict={
            Tmatrix:file_read.readTmatrix("./data/tmatrix1.txt"),
            SGroundTruth:file_read.readfaceHdr("./data/faceHdr.txt"),
            rho_sigma:np.matrix(file_read.readRho_sigma("./data/GMMdata/rhoGMModel_sigma.txt"),dtype=np.float64),
            rho_mu:np.matrix(file_read.readRho_mu("./data/GMMdata/rhoGMModel_mu.txt"), dtype=np.float64),
            rho_ComponentProportion:np.matrix(file_read.readRho_componentProportion("./data/GMMdata/rhoGMModel_ComponentProportion.txt"), dtype=np.float64),
            theta_sigma:np.matrix(file_read.readTheta_sigma("./data/GMMdata/thetaGMModel_sigma.txt"),dtype=np.float64),
            theta_mu:np.matrix(file_read.readTheta_mu("./data/GMMdata/thetaGMModel_mu.txt"), dtype=np.float64),
            theta_componentProportion:np.matrix(file_read.readTheta_componentProportion("./data/GMMdata/thetaGMModel_ComponentProportion.txt"), dtype=np.float64)
        })
        print('cost',cost1)
