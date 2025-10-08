import tensorflow as tf
import numpy as np
import six
try:
    import cPickle
except ImportError:
    import pickle as cPickle

class SigNetModel:
    """SigNet签名验证模型 (TensorFlow 2.x兼容)"""
    
    def __init__(self, model_path):
        print("📦 正在加载SigNet模型...")
        
        # 加载模型参数
        with open(model_path, 'rb') as f:
            if six.PY2:
                model_params = cPickle.load(f)
            else:
                model_params = cPickle.load(f, encoding='latin1')
        
        self.params = model_params['params']
        self.input_size = model_params['input_size']  # (150, 220)
        
        # 创建TensorFlow模型
        tf.compat.v1.disable_v2_behavior()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x_input = tf.compat.v1.placeholder(tf.float32, [None, self.input_size[0], self.input_size[1], 1])
            self.model = self._build_network_keras(self.x_input)
            self.sess = tf.compat.v1.Session()
            self.sess.run(tf.compat.v1.global_variables_initializer())
        
        print(f"✅ SigNet模型加载成功! 输入尺寸: {self.input_size}")
    
    def _build_network_keras(self, input_var):
        """使用Keras Sequential API构建网络"""
        from tensorflow import keras
        
        # 使用Functional API以兼容placeholder
        x = input_var
        
        # Conv1: 96@11x11 strides=4
        x = keras.layers.Conv2D(96, 11, strides=4, padding='valid', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D(3, 2)(x)
        
        # Conv2: 256@5x5
        x = keras.layers.Conv2D(256, 5, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D(3, 2)(x)
        
        # Conv3: 384@3x3
        x = keras.layers.Conv2D(384, 3, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        
        # Conv4: 384@3x3
        x = keras.layers.Conv2D(384, 3, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        
        # Conv5: 256@3x3
        x = keras.layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D(3, 2)(x)
        
        # Flatten
        x = keras.layers.Flatten()(x)
        
        # FC6: 2048
        x = keras.layers.Dense(2048, use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dropout(0.5)(x)
        
        # FC7: 2048 (features)
        x = keras.layers.Dense(2048)(x)
        
        return x
    
    def get_feature_vector(self, image):
        """提取签名特征向量 (2048维)"""
        # 确保输入形状正确: (1, 150, 220, 1)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=(0, 3))
        elif len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        with self.graph.as_default():
            features = self.sess.run(self.model, feed_dict={self.x_input: image})
        
        return features[0]
    
    def compute_similarity(self, template_img, query_img):
        """计算两个签名的欧氏距离"""
        feat1 = self.get_feature_vector(template_img)
        feat2 = self.get_feature_vector(query_img)
        
        # 欧氏距离
        euclidean_dist = np.linalg.norm(feat1 - feat2)
        
        return euclidean_dist
