import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim

sess = tf.InteractiveSession()
# load MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])
imgs=mnist.train.images
labels=mnist.train.labels
# MNIST resize and normalization
train_set_imgs = tf.image.resize_images(imgs, [64, 64]).eval()
train_set_imgs = (train_set_imgs - 0.5) / 0.5  # normalization; range: -1 ~ 1
train_set_txt = (labels - 0.5)/0.5

# training parameters

batch_size = 100
lr_d = 0.0003
lr_g =0.0003
#lr_g2= 0.00002
train_epoch = 200

vocab=['0','1','2','3','4','5','6','7','8','9']

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

def KL_loss_txt(mu,sigma,reuse=False):
    with tf.variable_scope("generator_img_KL_divergence",reuse=reuse):
        #loss=tf.distributions.kl_divergence(mu,sigma,allow_nan_stats=False)
        loss=-sigma + .5 * (-1 + tf.exp(2. * sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss
def KL_loss_img(mu,sigma,reuse=False):
    with tf.variable_scope("generator_txt_KL_divergence",reuse=reuse):
        #loss=tf.distributions.kl_divergence(mu,sigma,allow_nan_stats=False)
        loss=-sigma + .5 * (-1 + tf.exp(2. * sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss
def generate_cond_img(txt,reuse=False):
    with tf.variable_scope("generator_img_cond",reuse=reuse):
        f=tf.layers.Flatten()(txt)
        fc=tf.layers.dense(f,20,use_bias=True,bias_initializer=tf.zeros_initializer(),trainable=True,name=None,reuse=reuse)
        lr=lrelu(fc,0.2)
        mean=lr[:,:10]
        stddev=lr[:,10:]
        dist=tf.truncated_normal(tf.shape(mean))
        c_out=mean+ stddev*dist

        return c_out,mean,stddev


def generate_cond_txt(img,reuse=False):
    with tf.variable_scope("generator_txt_cond",reuse=reuse):
        f=tf.layers.Flatten()(img)
        fc=tf.layers.dense(f,2*64*64,use_bias=True,bias_initializer=tf.zeros_initializer(),trainable=True,name=None,reuse=reuse)
        lr=lrelu(fc,0.2)
        mean=lr[:,:(64*64)]
        stddev=lr[:,(64*64):]
        dist=tf.truncated_normal(tf.shape(mean))
        c_out=mean+ stddev*dist

        return c_out,mean,stddev


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)



def decode_embed(array, vocab):
    array=list(array)
    index=array.index(max(array))
    return vocab[index]

def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        conv1 = tf.layers.conv2d_transpose(x, 512, [4, 4], strides=(2, 2), padding='valid')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [3, 3], strides=(1, 1), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        conv3 = tf.layers.conv2d_transpose(lrelu2, 128, [3, 3], strides=(1, 1), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [1, 1], strides=(1, 1), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        conv5 = tf.layers.conv2d_transpose(lrelu4, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu5 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        
        conv6 = tf.layers.conv2d_transpose(lrelu5, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu6 = lrelu(tf.layers.batch_normalization(conv6, training=isTrain), 0.2)

        conv7 = tf.layers.conv2d_transpose(lrelu6, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu7 = lrelu(tf.layers.batch_normalization(conv7, training=isTrain), 0.2)

        conv8 = tf.layers.conv2d_transpose(lrelu7, 64, [4, 4], strides=(2, 2), padding='same')
        lrelu8 = lrelu(tf.layers.batch_normalization(conv8, training=isTrain), 0.2)
        
        conv9 = tf.layers.conv2d_transpose(lrelu8, 1, [4, 4], strides=(2, 2), padding='same')
        
        o_img = tf.nn.tanh(conv9)
        
        lrelu_txt=tf.layers.Flatten()(lrelu(lrelu1,0.2))
        op_txt= tf.layers.dense(lrelu_txt,10,use_bias=True,bias_initializer=tf.zeros_initializer(),trainable=True,name=None,reuse=reuse)
        
        o_txt = tf.nn.tanh(op_txt)

        return o_img,o_txt

# D(x)

def discriminator_combined(x,y, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator_combined', reuse=reuse):
        
        conv1 = tf.layers.conv2d(x, 64, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = lrelu(conv1, 0.2)

        conv2 = tf.layers.conv2d(lrelu1, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        conv3 = tf.layers.conv2d(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        conv4 = tf.layers.conv2d(lrelu3, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        
        conv5 = tf.layers.conv2d(lrelu4, 128, [1, 1], strides=(1, 1), padding='same')
        lrelu5= lrelu(tf.layers.batch_normalization(conv5, training=isTrain), 0.2)

        conv6 = tf.layers.conv2d(lrelu5, 128, [3, 3], strides=(1, 1), padding='same')
        lrelu6= lrelu(tf.layers.batch_normalization(conv6, training=isTrain), 0.2)

        conv7 = tf.layers.conv2d(lrelu6, 512, [3, 3], strides=(1, 1), padding='same')
        lrelu7= lrelu(tf.layers.batch_normalization(conv7, training=isTrain), 0.2)
        
        fc= tf.layers.dense(y,10,use_bias=True,bias_initializer=tf.zeros_initializer(),trainable=True,name=None,reuse=reuse)
        lrelu_fc=lrelu(fc,0.2)
        z=tf.expand_dims(tf.expand_dims(lrelu_fc,1),1)
        txt=tf.tile(z,[1,4,4,1])
        img=tf.concat([lrelu7,txt],axis=3)
        
        conv8 = tf.layers.conv2d(img, 512, [1, 1], strides=(1, 1), padding='same')
        lrelu8= lrelu(tf.layers.batch_normalization(conv8, training=isTrain), 0.2)

        o = tf.layers.conv2d(lrelu8, 1, [4, 4], strides=(4, 4), padding='valid')
        
        return o

def discriminator_img(x,isTrain=True, reuse=False):
    with tf.variable_scope('discriminator_img', reuse=reuse):
        
        conv1 = tf.layers.conv2d(x, 64, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = lrelu(conv1, 0.2)

        conv2 = tf.layers.conv2d(lrelu1, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        conv3 = tf.layers.conv2d(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        conv4 = tf.layers.conv2d(lrelu3, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        
        conv5 = tf.layers.conv2d(lrelu4, 128, [1, 1], strides=(1, 1), padding='same')
        lrelu5= lrelu(tf.layers.batch_normalization(conv5, training=isTrain), 0.2)

        conv6 = tf.layers.conv2d(lrelu5, 128, [3, 3], strides=(1, 1), padding='same')
        lrelu6= lrelu(tf.layers.batch_normalization(conv6, training=isTrain), 0.2)

        conv7 = tf.layers.conv2d(lrelu6, 512, [3, 3], strides=(1, 1), padding='same')
        lrelu7= lrelu(tf.layers.batch_normalization(conv7, training=isTrain), 0.2)
        
        conv8 = tf.layers.conv2d(lrelu7, 512, [1, 1], strides=(1, 1), padding='same')
        lrelu8= lrelu(tf.layers.batch_normalization(conv8, training=isTrain), 0.2)

        o = tf.layers.conv2d(lrelu8, 1, [4, 4], strides=(4, 4), padding='valid')
        
        return o

def discriminator_txt(y, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator_txt', reuse=reuse):
        
        fc= tf.layers.dense(y,10,use_bias=True,bias_initializer=tf.zeros_initializer(),trainable=True,name=None,reuse=reuse)
        lrelu_fc=lrelu(fc,0.2)
        z=tf.expand_dims(tf.expand_dims(lrelu_fc,1),1)
        txt=tf.tile(z,[1,4,4,1])
        conv8 = tf.layers.conv2d(txt, 512, [1, 1], strides=(1, 1), padding='same')
        lrelu8= lrelu(tf.layers.batch_normalization(conv8, training=isTrain), 0.2)

        o = tf.layers.conv2d(lrelu8, 1, [4, 4], strides=(4, 4), padding='valid')
        
        return o
tsi=train_set_imgs[:batch_size]
tst=train_set_txt[: batch_size]
fixed_z_=np.random.normal(0, 1, (batch_size,100))
def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    test_images = sess.run(G_z_img, {z: fixed_z_,x:tsi,y:tst, isTrain: False})
    test_texts = sess.run(G_z_txt, {z: fixed_z_,x:tsi,y:tst, isTrain: False})
    
    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(10, 10))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (64, 64)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
    gen_num=[]
    for i in range(100):
        gen_num.append(decode_embed(test_texts[i,:],vocab))
    f=open('Results/generated_text','a+')
    f.write('epoch{},Generated Nos.{}\n'.format(epoch,gen_num))
    f.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']
    y3 = hist['D_losses_dash']
    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')
    plt.plot(x, y3,label='D_loss_dash')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# variables : input
x = tf.placeholder(tf.float32, shape=(batch_size, 64, 64, 1))
#x_wrong=tf.placeholder(tf.float32, shape=(batch_size, 64, 64, 1))
y=tf.placeholder(tf.float32,shape=(batch_size,10))
#y_wrong=tf.placeholder(tf.float32,shape=(batch_size,10))
z = tf.placeholder(tf.float32, shape=(batch_size, 100))
isTrain = tf.placeholder(dtype=tf.bool)

img_cond, mu1,stddev1=generate_cond_img(y)

txt_cond, mu2,stddev2=generate_cond_txt(x)

kl_loss_img=KL_loss_txt(mu1,stddev1)

kl_loss_txt=KL_loss_img(mu2,stddev2)

z_train=tf.expand_dims(tf.expand_dims(tf.concat([img_cond,txt_cond,z],1),1),1)
#z_txt=tf.expand_dims(tf.expand_dims(tf.concat([txt_cond,z],1),1),1)

# networks : generator
G_z_img,G_z_txt=generator(z_train, isTrain)
#G_z_txt = generator_txt(z_train, isTrain)
# networks : discriminator
D_real_logits = discriminator_combined(x,y, isTrain)

#D_wrong_logits = discriminator(x_wrong,y_wrong,isTrain,reuse=True)
D_fake_logits = discriminator_combined(G_z_img, G_z_txt, isTrain, reuse=True)

D_img_logits_real=discriminator_img(x,isTrain)
D_txt_logits_real=discriminator_txt(y,isTrain)

D_img_logits_fake=discriminator_img(G_z_img,isTrain,reuse=True)
D_txt_logits_fake=discriminator_txt(G_z_txt,isTrain,reuse=True)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
#D_loss_wrong= tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_wrong_logits, labels=tf.zeros_like(D_wrong_logits)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
D_loss = D_loss_real + D_loss_fake #+ D_loss_wrong)/2 


G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))+kl_loss_img+kl_loss_txt
#G_txt_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))+kl_loss_txt

D_img_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_img_logits_real, labels=tf.ones_like(D_img_logits_real)))
D_txt_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_txt_logits_real, labels=tf.ones_like(D_txt_logits_real)))

D_img_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_img_logits_fake, labels=tf.zeros_like(D_img_logits_fake)))
D_txt_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_txt_logits_fake, labels=tf.zeros_like(D_txt_logits_fake)))

D_img_loss=D_img_loss_real+D_img_logits_fake

D_txt_loss=D_txt_loss_real+D_txt_loss_fake

D_loss_dash=tf.reduce_mean(D_loss-(D_img_loss+D_txt_loss))
# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
#D_img_vars = [var for var in T_vars if var.name.startswith('discriminator_img')]
#D_txt_vars = [var for var in T_vars if var.name.startswith('discriminator_txt')]

G_vars = [var for var in T_vars if var.name.startswith('generator')]
#G_txt_vars = [var for var in T_vars if var.name.startswith('generator_txt')]


# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr_d, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr_g, beta1=0.5).minimize(G_loss, var_list=G_vars)
    #G_txt_optim = tf.train.AdamOptimizer(lr_g2, beta1=0.5).minimize(G_txt_loss, var_list=G_txt_vars)


show_all_variables()


# open session and initialize all variables

tf.global_variables_initializer().run()
saver = tf.train.Saver(tf.global_variables())


# results save folder
root = 'MNIST_results/'
model = 'MNIST_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['D_losses_dash'] = []

train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses_dash=[]
    D_losses = []
    epoch_start_time = time.time()
    iter=0
    for iter in range(mnist.train.num_examples // batch_size):
        # update discriminator
        x_ = train_set_imgs[iter*batch_size:(iter+1)*batch_size]
        y_ = train_set_txt[iter*batch_size:(iter+1)*batch_size]
        
        z_ = np.random.normal(0, 1, (batch_size,100))
        
        loss_d_, _,d_dash = sess.run([D_loss, D_optim,D_loss_dash], {x: x_,y:y_,z:z_ ,isTrain: True})
        D_losses.append(loss_d_)
        D_losses_dash.append(d_dash)
        # update generators
        loss_g, _ ,img,txt= sess.run([G_loss, G_optim,G_z_img,G_z_txt], {z: z_, x: x_,y: y_, isTrain: True})
        G_losses.append(loss_g)
        #loss_g_txt, _,txt = sess.run([G_txt_loss, G_txt_optim,G_z_txt], {z: z_,x: x_, y: y_, isTrain: True})
        #G_txt_losses.append(loss_g_txt)




    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.8f, loss_g: %.8f, loss_ddash: %.8f' % ((epoch + 1), train_epoch, per_epoch_ptime, 
        np.mean(D_losses), np.mean(G_losses),np.mean(D_losses_dash)))
    
    
    
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result((epoch + 1), save=True, path=fixed_p)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['D_losses_dash'].append(np.mean(D_losses_dash))

    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

saver.save(sess, "saved/model.ckpt")   
end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")

with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')


images = []
for e in range(train_epoch):
    img_name = root + 'img_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)


sess.close()
