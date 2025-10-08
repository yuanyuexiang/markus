"""
SigNet签名验证模型 - TensorFlow 2.x兼容版本
基于 https://github.com/luizgh/sigver_wiwd
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from six.moves import cPickle
import six


class SigNetModel:
    """SigNet签名验证模型封装"""
    
    def __init__(self, model_path='models/signet.pkl'):
        """
        加载SigNet预训练模型
        
        Parameters:
            model_path: 模型权重文件路径
        """
        print("📦 正在加载SigNet模型...")
        
        # 加载模型参数
        with open(model_path, 'rb') as f:
            if six.PY2:
                model_params = cPickle.load(f)
            else:
                model_params = cPickle.load(f, encoding='latin1')
        
        self.params = model_params['params']
        self.input_size = model_params['input_size']  # (150, 220)
        
        # 创建TensorFlow图
        self.graph = tf.Graph()
        with self.graph.as_default():
            # 定义输入占位符
            self.x_input = tf.placeholder(tf.float32, (None, 150, 220, 1))
            
            # 构建网络
            self.model = self._build_network(self.x_input)
            
            # 创建session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=self.graph)
            
            # 初始化变量
            self.sess.run(tf.global_variables_initializer())
        
        print("✅ SigNet模型加载完成")
    
    def _build_network(self, input_var):
        """构建SigNet网络架构"""
        # 使用tf.keras.layers替代tf.layers (Keras 3兼容)
        
        # Conv Layer 1: 96@11x11
        conv1 = tf.keras.layers.Conv2D(
            96, (11, 11), strides=4, padding='valid', activation=None,
            kernel_initializer=tf.constant_initializer(self.params['conv1_W']),
            bias_initializer=tf.constant_initializer(self.params['conv1_b']),
            name='conv1'
        )(input_var)
        conv1_bn = self._batch_norm(conv1, 'conv1')
        
        # Conv2 + BN + ReLU + Pool
        conv2 = tf.keras.layers.conv2d(
            pool1, 256, 5, padding='same',
            kernel_initializer=self._get_initializer(params[5]),
            use_bias=False, name='conv2'
        )
        conv2 = self._batch_norm(conv2, 'conv2_bn', params[6], params[7], params[8], params[9])
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.keras.layers.max_pooling2d(conv2, 3, 2, name='pool2')
        
        # Conv3 + BN + ReLU
        conv3 = tf.keras.layers.conv2d(
            pool2, 384, 3, padding='same',
            kernel_initializer=self._get_initializer(params[10]),
            use_bias=False, name='conv3'
        )
        conv3 = self._batch_norm(conv3, 'conv3_bn', params[11], params[12], params[13], params[14])
        conv3 = tf.nn.relu(conv3)
        
        # Conv4 + BN + ReLU
        conv4 = tf.keras.layers.conv2d(
            conv3, 384, 3, padding='same',
            kernel_initializer=self._get_initializer(params[15]),
            use_bias=False, name='conv4'
        )
        conv4 = self._batch_norm(conv4, 'conv4_bn', params[16], params[17], params[18], params[19])
        conv4 = tf.nn.relu(conv4)
        
        # Conv5 + BN + ReLU + Pool
        conv5 = tf.keras.layers.conv2d(
            conv4, 256, 3, padding='same',
            kernel_initializer=self._get_initializer(params[20]),
            use_bias=False, name='conv5'
        )
        conv5 = self._batch_norm(conv5, 'conv5_bn', params[21], params[22], params[23], params[24])
        conv5 = tf.nn.relu(conv5)
        pool5 = tf.keras.layers.max_pooling2d(conv5, 3, 2, name='pool5')
        
        # Flatten
        flat = tf.keras.layers.flatten(pool5)
        
        # FC1 + BN + ReLU
        fc1 = tf.keras.layers.dense(
            flat, 2048,
            kernel_initializer=self._get_initializer(params[25]),
            use_bias=False, name='fc1'
        )
        fc1 = self._batch_norm(fc1, 'fc1_bn', params[26], params[27], params[28], params[29])
        fc1 = tf.nn.relu(fc1)
        
        # FC2 + BN + ReLU
        fc2 = tf.keras.layers.dense(
            fc1, 2048,
            kernel_initializer=self._get_initializer(params[30]),
            use_bias=False, name='fc2'
        )
        fc2 = self._batch_norm(fc2, 'fc2_bn', params[31], params[32], params[33], params[34])
        net['fc2'] = tf.nn.relu(fc2)
        
        return net
    
    def _get_initializer(self, weights):
        """创建权重初始化器"""
        # 转换Lasagne格式(BCHW)到TensorFlow格式(BHWC)
        if len(weights.shape) == 4:
            weights = np.transpose(weights, [2, 3, 1, 0])
        return tf.constant_initializer(weights)
    
    def _batch_norm(self, input_tensor, scope, beta, gamma, mean, inv_std):
        """批标准化层"""
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            beta_var = tf.get_variable('beta', initializer=beta, dtype=tf.float32)
            gamma_var = tf.get_variable('gamma', initializer=gamma, dtype=tf.float32)
            mean_var = tf.get_variable('mean', initializer=mean, dtype=tf.float32)
            inv_std_var = tf.get_variable('inv_std', initializer=inv_std, dtype=tf.float32)
            return (input_tensor - mean_var) * (gamma_var * inv_std_var) + beta_var
    
    def get_feature_vector(self, image):
        """
        提取单张图片的特征向量
        
        Parameters:
            image: numpy array of shape (150, 220)
        
        Returns:
            feature vector of shape (2048,)
        """
        assert len(image.shape) == 2, "Input should have two dimensions: H x W"
        
        # 扩展维度: (H, W) -> (1, H, W, 1)
        input_batch = image[np.newaxis, :, :, np.newaxis]
        
        with self.graph.as_default():
            features = self.sess.run(self.model['fc2'], feed_dict={self.x_input: input_batch})
        
        return features[0]  # 返回第一个特征向量
    
    def compute_similarity(self, img1, img2):
        """
        计算两张签名图片的相似度
        
        Parameters:
            img1, img2: numpy arrays of shape (150, 220)
        
        Returns:
            euclidean distance (float)
        """
        feat1 = self.get_feature_vector(img1)
        feat2 = self.get_feature_vector(img2)
        
        # 计算欧氏距离
        euclidean_dist = np.linalg.norm(feat1 - feat2)
        
        return euclidean_dist
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'sess'):
            self.sess.close()
