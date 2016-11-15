import tensorflow as tf

def weight_variable(shape,name='weight',stddev=0.1):
    with tf.name_scope(name):
        init = tf.truncated_normal(shape,stddev=stddev)
        weights = tf.Variable(init,name=name)
        #if len(weights.get_shape()) == [5,5,1,6]:
        #    tmp=tf.transpose(weights,perm=[3,0,1,2], name='tmp')
        return weights
def bias_variable(shape,name='bias'):
    with tf.name_scope(name):
        init = tf.constant(0.1,shape=shape)
        bias = tf.Variable(init,name=name)
        return bias

######################### LAYERS #########################
def conv_layer(bottom,Weights,bias=None,name='conv_layer'):
    with tf.name_scope(name):
        conv_w = weight_variable(Weights,name=name+'/weight')# [cols,rows,channels,n]
        conv = tf.nn.conv2d(bottom,conv_w,strides=[1,1,1,1], padding='SAME')
        #tf.histogram_summary(name+'/conv',conv)
        if bias==None:
            return conv
        else:
            b = bias_variable(bias,name=name+'/bias')
            return conv + b
        
def pooling_layer(bottom,name='pooling_layer'):
    with tf.name_scope(name):
        return tf.nn.max_pool(bottom,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
def relu_layer(bottom,name='relu_layer'):
    with tf.name_scope(name):    
        return tf.nn.relu(bottom)

def fully_connected(bottom,Weights,name='fc'):
    with tf.name_scope(name):
        return tf.matmul(bottom,Weights)

def softmax_layer(bottom,name='softmax'):
    with tf.name_scope(name):
        return tf.nn.softmax(bottom)
        
def upsampling_layer(bottom,new_height=100,new_width=100,name='upsampling'):
    with tf.name_scope(name):
        return tf.image.resize_images(bottom, [new_height, new_width])
        
def deconv_layer(bottom,shape,Weights,bias=None,name='deconv_layer'):
    with tf.name_scope(name):
        deconv_w = weight_variable(Weights,name=name+'/weight')# [cols,rows,channels,n]
        deconv = tf.nn.conv2d_transpose(bottom,deconv_w,shape,strides=[1,1,1,1])
        if bias==None:
            return deconv
        else:
            b = bias_variable(bias,name=name+'/bias')
            return deconv + b