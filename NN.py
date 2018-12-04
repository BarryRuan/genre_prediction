import tensorflow as tf
import numpy as np
from data.features import LyricsDataSet
import os
import sys

def restore(sess, checkpoint_path):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    if checkpoint:
        path = checkpoint.model_checkpoint_path
        print('Restoring model parameters from {}'.format(path))
        saver.restore(sess, path)
    else:
        print('No saved model parameters found')
    # Return checkpoint path for call to saver.save()
    save_path = os.path.join(
        checkpoint_path, os.path.basename(os.path.dirname(checkpoint_path)))
    return saver, save_path

def NN(input_layer):
    weights = {
            'w1': tf.Variable(\
                    tf.truncated_normal([5000, 200], stddev=0.03),name='w1'),
            'w2': tf.Variable(\
                    tf.truncated_normal([200, 100], stddev=0.03),name='w2'),
            'w3': tf.Variable(\
                    tf.truncated_normal([100, 15], stddev=0.03),name='w3')
            }
    biases = {
            'b1': tf.Variable(tf.truncated_normal([200]),name='b1'),
            'b2': tf.Variable(tf.truncated_normal([100]),name='b2'),
            'b3': tf.Variable(tf.truncated_normal([15]),name='b3')
            }


    fc1 = tf.add(tf.matmul(input_layer, weights['w1']), biases['b1'])
    fc1 = tf.nn.relu(fc1)
    fc2 = tf.add(tf.matmul(fc1, weights['w2']), biases['b2'])
    fc2 = tf.nn.relu(fc2)
    out = tf.add(tf.matmul(fc2, weights['w3']), biases['b3'])

    return out


def loss(nn_out, y):
    sce = tf.nn.softmax_cross_entropy_with_logits(logits=nn_out, labels=y)
    return tf.reduce_mean(sce)


def optimizer(error, lr=10e-5):
    return tf.train.AdamOptimizer(learning_rate=lr).minimize(error)


def accuracy(pred, labels):
    pred = tf.nn.softmax(pred, axis=1)
    pred = tf.argmax(pred,axis=1)
    result = tf.equal(pred, tf.argmax(labels,axis=1))
    accuracy = tf.reduce_mean(tf.cast(result, tf.float32))
    return accuracy

def prediction(pred):
    pred = tf.nn.softmax(pred, axis=1)
    pred = tf.argmax(pred,axis=1)
    return pred

def train(sess, saver, save_path, input_layer, pred_labels,\
        label_layer, err, opti, acc, lyricsdata):
    print("Start training")
    for batch_i in range(1500):
        if batch_i % 100 == 0:
            saver.save(sess, save_path)
            print("{} batches trained.".format(batch_i))
            batch_x, batch_y = lyricsdata.get_batch('dev',batch_size=1024)
            preds, dev_acc, dev_loss = sess.run([pred_labels, acc, err],\
                    feed_dict={input_layer:batch_x, label_layer:batch_y})
            print(preds)
            print("Accuracy is {}.\n Loss is {}.".format(dev_acc, dev_loss))
        batch_x, batch_y = lyricsdata.get_batch('train',batch_size=248)
        sess.run(opti, feed_dict={input_layer:batch_x, label_layer:batch_y})


def test(sess, input_layer, pred_labels, lyricsdata):
    test_y = lyricsdata.get_test_y()
    preds = np.zeros(1)
    while not lyricsdata.test_done():
        batchx = lyricsdata.get_batch('test')
        preds = np.concatenate([preds,\
                sess.run([pred_labels], {input_layer:batchx})])
    preds = preds[1:]
    return np.sum(preds==test_y)/len(test_y)

def main(feature):
    print("Building model.")
    input_layer = tf.placeholder(tf.float32, shape=(None, 5000))
    label_layer = tf.placeholder(tf.float32, shape=(None, 15))
    nn_out = NN(input_layer)
    err = loss(nn_out, label_layer)
    opti = optimizer(err)
    pred_labels = prediction(nn_out)
    lyricsdata = LyricsDataSet(feature_type=feature)
    acc = accuracy(nn_out, label_layer)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver, save_path = restore(sess, 'nn_checkpoints/{}'.format(feature))
        train(sess, saver, save_path, input_layer, pred_labels,\
            label_layer, err, opti, acc, lyricsdata)
        total_acc = test(sess, input_layer, pred_labels, lyricsdata)
        print("Accuracy on test set is {}".format(total_acc))
        print('saving trained model...\n')
        saver.save(sess, save_path)

if __name__ == '__main__':
    main(sys.argv[1])
