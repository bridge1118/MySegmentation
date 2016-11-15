import tensorflow as tf

from MyNet import conv_layer,pooling_layer,relu_layer
from Readim import MyImages

import scipy.io

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc


batch_size = 1
h = 512
w = 512
c = 1
learning_rate = 0.01

imdir='./SPINE_data'
myImages = MyImages()
myImages.build(imdir,batch_size,'png')


xs = tf.placeholder(tf.float32,[batch_size,h,w,c],name='xs')
ys = tf.placeholder(tf.float32,[batch_size,h,w,c],name='ys')


########## LAYER DEFINITION START ##########
### Encoder ###
# layer 1
conv1_1 = conv_layer(xs, [3,3,1,50], [50], name='conv1_1') #[512*512*1]->[512*512*96]
relu1_1 = relu_layer(conv1_1,name='relu1_1')
pool1_1 = pooling_layer(relu1_1, name='pool1_1') # [512*512*96]->[256*256*96]

# layer 2
conv2_1 = conv_layer(pool1_1, [3,3, 50,200], [200], name='conv2_1') # [256*256*96]->[256*256*256]
relu2_1 = relu_layer(conv2_1,name='relu2_1')
pool2_1 = pooling_layer(relu2_1, name='pool2_1') # [256*256*256]->[128*128*256]

# layer 3
conv3_1 = conv_layer(pool2_1, [3,3,200, 256], [256], name='conv3_1') # [128*128*256]->[128*128*384]
relu3_1 = relu_layer(conv3_1,name='relu3_1')
conv3_2 = conv_layer(relu3_1, [5,5,256, 384], [384], name='conv3_2') # [128*128*384]->[128*128*384]
relu3_2 = relu_layer(conv3_2,name='relu3_2')

# layer 4 (fc)
fc4_1 = conv_layer(relu3_2,[1,1,384,2048],[2048], name='fc4_1') # [128*128*384]->[128*128*4096]
fc4_1 = relu_layer(fc4_1,name='relu4_1')

# layer 5 (fc)
fc5_1 = conv_layer(fc4_1,[1,1,2048,500],[500], name='fc5_1') # [128*128*4096]->[128*128*1000]
fc5_1 = relu_layer(fc5_1,name='relu5_1-fc')

# layer 6 (fc)
fc6_1 = conv_layer(fc5_1,[1,1,500,1],[1], name='fc6_1') # [128*128*4096]->[128*128*1]
fc6_1 = relu_layer(fc6_1,name='relu6_1-fc')

pred = fc6_1

# training solver
with tf.name_scope('loss'):
    
    # autoencoder
    '''
    ys_reshape = tf.image.resize_images(xs, [128, 128])
    cross_entropy = tf.reduce_mean(tf.pow(ys_reshape-pred,2),3)
    #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(pred,ys_reshape)
    prediction = cross_entropy
    cross_entropy = tf.reduce_sum( cross_entropy ) 
    tf.scalar_summary('loss',cross_entropy)
    '''
    
    # classification
    ys_reshape = tf.image.resize_images(ys, [128, 128]) #[-1,128,128,2]
    cross_entropy = tf.reduce_mean(tf.pow(ys_reshape-pred,2),3)
    prediction = cross_entropy
    cross_entropy = tf.reduce_sum( cross_entropy ) 
    tf.scalar_summary('loss',cross_entropy)

with tf.name_scope('solver'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
########## LAYER DEFINITION END ##########


# start training
sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("logs/", sess.graph)
sess.run(tf.initialize_all_variables())

for step in range(5):
    
    print('step '+str(step))
    
    batch_xs, batch_ys = myImages.nextBatch()
    _, loss_value = sess.run([train_step, cross_entropy],
                             feed_dict={xs:batch_xs, ys:batch_ys})
    print(loss_value)
    
    #img = sess.run(xs,feed_dict={xs:batch_xs})
    #scipy.io.savemat('batch.mat', mdict={'batch': img})
    
    results = sess.run(merged,feed_dict={xs:batch_xs, ys:batch_ys})
    writer.add_summary(results,step)
    writer.flush()

    
# Testing
f, axarr = plt.subplots(3, 2)
for step in range(2):
    batch_xs, batch_ys = myImages.nextBatch()
    img,yr = sess.run([prediction,ys_reshape],feed_dict={xs:batch_xs, ys:batch_ys})

    batch_xs = np.squeeze(batch_xs)
    axarr[0,step].imshow(batch_xs)

    scipy.misc.imsave('input'+str(step)+'.jpg', batch_xs)

    yr = np.squeeze(batch_ys[:,:,:,0])
    axarr[1,step].imshow(yr)
    scipy.misc.imsave('groundtruth'+str(step)+'.jpg', yr)

    #img = np.squeeze(img[:,:,:,0])
    img = np.squeeze(img)
    axarr[2,step].imshow(img)
    scipy.misc.imsave('outfile'+str(step)+'.jpg', img)

plt.show()


batch_xs, batch_ys = myImages.nextBatch()
conv11,conv32 = sess.run([conv1_1,conv3_2],feed_dict={xs:batch_xs, ys:batch_ys})
#scipy.io.savemat('conv.mat', mdict={'conv': conv})



sess.close()


