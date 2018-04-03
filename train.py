import numpy as np
import os,sys,inspect
import tensorflow as tf
import time
from datetime import datetime
import os
# import hickle as hkl
import os.path as osp
from glob import glob
import sklearn.metrics as metrics

# from input import Dataset
import globals as g_

import utils.config
import utils.split_fun
import provider


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', osp.dirname(sys.argv[0]) + '/tmp/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('weights', '', 
                            """finetune with a pretrained model""")
tf.app.flags.DEFINE_string('caffemodel', '', 
                            """finetune with a model converted by caffe-tensorflow""")

np.set_printoptions(precision=3)

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(10000)
BN_DECAY_CLIP = 0.99


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*4,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay



def train(cfg, dataset_train, dataset_val, ckptfile='', caffemodel=''):
    print ('train() called')
    is_finetune = bool(ckptfile)
    V = g_.NUM_VIEWS
    batch_size = FLAGS.batch_size

    # dataset_train.shuffle()
    # dataset_val.shuffle()
    data_size, num_batch = dataset_train.get_len()
    # data_size = len(dataset_train)
    # print ('train size:', data_size)

    data_size_test, num_batch_test = dataset_val.get_len()
    print ('train size:', data_size)
    print ('test size:', data_size_test)

    best_eval_acc = 0




    with tf.Graph().as_default():
        # startstep = 0 if not is_finetune else int(ckptfile.split('-')[-1])
        startstep = 0
        global_step = tf.Variable(startstep, trainable=False)
         
        # placeholders for graph input
        view_ = tf.placeholder('float32', shape=(None, V, 224, 224, 3), name='im0')
        y_ = tf.placeholder('int64', shape=(None), name='y')
        is_training_pl = tf.placeholder(tf.bool, shape=())
        bn_decay = get_bn_decay(startstep)

        # graph outputs
        fc8 = model.inference_multiview(view_, g_.NUM_CLASSES, is_training_pl, bn_decay=bn_decay)
        loss = model.loss(fc8, y_)
        train_op = model.train(loss, global_step, data_size)
        prediction = model.classify(fc8)

        # build the summary operation based on the F colection of Summaries
        summary_op = tf.summary.merge_all()


        # must be after merge_all_summaries
        validation_loss = tf.placeholder('float32', shape=(), name='validation_loss')
        validation_summary = tf.summary.scalar('validation_loss', validation_loss)
        validation_acc = tf.placeholder('float32', shape=(), name='validation_accuracy')
        validation_acc_summary = tf.summary.scalar('validation_accuracy', validation_acc)

        # tvars = tf.trainable_variables()
        # print (tvars)
        # print (tf.get_default_graph().as_graph_def())

        saver = tf.train.Saver()

        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        
        if is_finetune:
            # load checkpoint file
            sess.run(init_op)
            optimistic_restore(sess, ckptfile)
            # saver.restore(sess, ckptfile)
            print ('restore variables done')
        elif caffemodel:
            # load caffemodel generated with caffe-tensorflow
            sess.run(init_op)
            model.load_alexnet_to_mvcnn(sess, caffemodel)
            print ('loaded pretrained caffemodel:', caffemodel)
        else:
            # from scratch
            sess.run(init_op)
            print ('init_op done')

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                               graph=sess.graph) 

        step = startstep


        for epoch in range(100):
            total_correct_mv = 0
            loss_sum_mv = 0
            total_seen = 0

            val_correct_sum = 0
            val_seen = 0
            loss_val_sum = 0
            print ('epoch:', epoch)

            for i in range(num_batch):
                # st = time.time()
                batch_x, batch_y = dataset_train.get_batch(i)
                # print (time.time()-st)
                step += 1

                start_time = time.time()
                feed_dict = {view_: batch_x,
                             y_ : batch_y,
                             is_training_pl: True }

                _, pred, loss_value = sess.run(
                        [train_op, prediction,  loss,],
                        feed_dict=feed_dict)

                duration = time.time() - start_time

                correct_mv = np.sum(pred == batch_y)
                total_correct_mv += correct_mv
                total_seen += g_.BATCH_SIZE
                loss_sum_mv += (loss_value * g_.BATCH_SIZE)

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                # print training information
                if step % 500 == 0 :
                    # print (pred)
                    # print (batch_y)
                    sec_per_batch = float(duration)
                    print ('%s: step %d, loss=%.2f, acc=%.4f (%.1f examples/sec; %.3f sec/batch)' \
                         % (datetime.now(), step, loss_sum_mv / float(total_seen), total_correct_mv / float(total_seen),
                                    FLAGS.batch_size/duration, sec_per_batch))

                    # for i in range(num_batch_test):
                    #     val_batch_x, val_batch_y = dataset_val.get_batch(i)
                    #     val_feed_dict = {view_: val_batch_x,
                    #                      y_: val_batch_y,
                    #                      is_training_pl: False}
                    #     val_loss, pred = sess.run([loss, prediction], feed_dict=val_feed_dict)
                    #
                    #     correct_mv_val = np.sum(pred == val_batch_y)
                    #     val_correct_sum += correct_mv_val
                    #     val_seen += g_.BATCH_SIZE
                    #     loss_val_sum += (val_loss * g_.BATCH_SIZE)
                    #
                    #     if i == 10:
                    #         print (pred)
                    #         print (val_batch_y)
                    #         print ('val loss=%.4f, acc=%.4f' % ((loss_val_sum / float(val_seen)), (val_correct_sum / float(val_seen))))


                if step % 1000 == 0:
                    # print 'running summary'
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                        
            # validation
            # val_losses = []
            # predictions = np.array([])
            # val_y = []

            for i in range(num_batch_test):
                val_batch_x, val_batch_y = dataset_val.get_batch(i)
                val_feed_dict = {view_: val_batch_x,
                                 y_  : val_batch_y,
                                 is_training_pl: False }
                val_loss, pred = sess.run([loss, prediction], feed_dict=val_feed_dict)

                correct_mv_val = np.sum(pred == val_batch_y)
                val_correct_sum += correct_mv_val
                val_seen += g_.BATCH_SIZE
                loss_val_sum += (val_loss * g_.BATCH_SIZE)

            val_mean_loss = (loss_val_sum / float(val_seen))
            acc = (val_correct_sum / float(val_seen))
            if acc > best_eval_acc:
                best_eval_acc = acc
                checkpoint_path = os.path.join(cfg.ckpt_folder, 'best_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            print ('%s: epoch %d, validation loss=%.4f, acc=%f, best_acc=%f' %\
                    (datetime.now(), epoch, val_mean_loss, acc, best_eval_acc))
            # validation summary
            val_loss_summ = sess.run(validation_summary,
                    feed_dict={validation_loss: val_mean_loss})
            val_acc_summ = sess.run(validation_acc_summary,
                    feed_dict={validation_acc: acc})
            summary_writer.add_summary(val_loss_summ, step)
            summary_writer.add_summary(val_acc_summ, step)
            summary_writer.flush()

            # if epoch % g_.SAVE_PERIOD == 0 and step > startstep:
            #     checkpoint_path = os.path.join(cfg.ckpt_folder, str(epoch)+'model.ckpt')
            #     saver.save(sess, checkpoint_path, global_step=step)


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    tvars = tf.global_variables()

    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def main(argv):
    st = time.time() 
    print ('start loading data')

    cfg = utils.config.config()

    dataset_train = provider.view_data(cfg, state='train', batch_size=g_.BATCH_SIZE, shuffle=True)
    dataset_val = provider.view_data(cfg, state='test', batch_size=g_.BATCH_SIZE, shuffle=True)

    # listfiles_train, labels_train = read_lists(g_.TRAIN_LOL)
    # listfiles_val, labels_val = read_lists(g_.VAL_LOL)
    # dataset_train = Dataset(listfiles_train, labels_train, subtract_mean=False, V=g_.NUM_VIEWS)
    # dataset_val = Dataset(listfiles_val, labels_val, subtract_mean=False, V=g_.NUM_VIEWS)

    print ('done loading data, time=', time.time() - st)

    train(cfg, dataset_train, dataset_val, FLAGS.weights, FLAGS.caffemodel)

#
# def read_lists(list_of_lists_file):
#     listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
#     listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
#     return listfiles, labels
    


if __name__ == '__main__':
    main(sys.argv)

