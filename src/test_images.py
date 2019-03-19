
import numpy as np
import tensorflow as tf
from keras import backend as K
import model_unet
import util
import sys
import os

if len(sys.argv) != 3:
    print('Usage: python test.py [model_PATH] [out_dir]')
    exit()

_, model_PATH, out_DIR = sys.argv

IMAGE_SZ = 128

mask = util.load_mask('mask.png')

files = np.load('places/places_128.npz')
imgs = files['imgs_test']
indices = files['idx_test']

tf.reset_default_graph()
K.set_learning_phase(1)
G_Z = tf.placeholder(tf.float32, shape=[None, IMAGE_SZ, IMAGE_SZ, 4], name='G_Z')
G_sample = model_unet.generator(G_Z)
K.set_learning_phase(0)

saver = tf.train.Saver()

with tf.Session() as sess:

    K.set_session(sess)
    saver.restore(sess, model_PATH)

    imgs_p = util.preprocess_images_outpainting(imgs, mask=mask)
    m = imgs_p.shape[0]

    for i in range(m):

        index = indices[i]
        if index not in [11, 146, 407, 761, 943, 2203, 2989, 3019, 3124, 3222, 3762, 3800, 4611, 5106]:
            continue

        output, = sess.run([G_sample], feed_dict={G_Z: imgs_p[i][np.newaxis]})

        img_out = output[0]
        img_in = imgs[i]

        # save output image
        name = f'dev{index:08d}_output'
        img_in_path = os.path.join(out_DIR, f'{name}_original.png')
        img_out_path = os.path.join(out_DIR, f'{name}_output.png')
        util.save_image(img_out, img_out_path)
        util.save_image(img_in, img_in_path)

        # save pasted and blended versions
        util.postprocess_images_outpainting(img_in_path, img_out_path, os.path.join(out_DIR, f'{name}_paste.png'), blend=False,
                                            mask=mask)
        util.postprocess_images_outpainting(img_in_path, img_out_path, os.path.join(out_DIR, f'{name}_blend.png'), blend=True,
                                            mask=mask)

        print(f'Saved images {name}')


