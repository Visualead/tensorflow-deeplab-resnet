import tensorflow as tf


def helloWorld(text):
    print(text)
    return True


def write_png(data, filepath):
    g = tf.Graph()
    with g.as_default():
        data_t = tf.placeholder(tf.uint8)
        op = tf.image.encode_png(data_t)
        init = tf.initialize_all_variables()

    with tf.Session(graph=g) as sess:
        sess.run(init)
        data_np = sess.run(op, feed_dict={data_t: data})

    with open(filepath, 'w') as fd:
        fd.write(data_np)
