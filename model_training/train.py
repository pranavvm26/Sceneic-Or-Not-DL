import copy
import glob
import os
import re
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
import model_training.input as mip
import model_training.model as mod
import model_training.utils as utl

slim = tf.contrib.slim


def _tower_loss(images, labels, dropout_keep_rate, scope):
    """
    Calculate the total loss on a single tower
    :param images: 4-D Tensor of size [singlegpu_batchsize, FLAGS.image_size,
                                       FLAGS.image_size, 3]
    :param labels: 1-D Tensor of size [singlegpu_batchsize
    :param dropout_keep_rate: Keep rate for dropout layers in training models
    :param scope: Prefix string identifying the tower.
    :return: Total Loss Value
    """

    # Build inference Graph.
    logits = mod.inference(images, dropout_keep_rate, is_training=True)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    mod.loss(logits, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection(mod.LOSSES_COLLECTION, scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='loss')

    # Compute the moving average of all individual losses and the total loss.
    # TODO: Need path for Tensorflow models issue #698
    # loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    # loss_averages_op = loss_averages.apply(losses + [total_loss])
    # acc_averages_op = loss_averages.apply(accuracy)

    # Attach a scalar summary to all individual losses and the total loss
    for l in losses:
        loss_name = re.sub('%s_[0-9]*/' % mod.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)
        # tf.summary.scalar(loss_name + '_MovingAverage', loss_averages.average(l))

    return total_loss, images, labels, logits[0]


def _average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    :param tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.

    :return: List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """

    with tf.name_scope('gradient_average'):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            #  Return the first tower's pointer to the  redundant Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


def train(logdir,
          logdir_train,
          train_files,
          batch_size,
          epochs,
          num_gpus,
          dropout,
          learning_rate,
          resume_train):

    # with a graph that created and pinned to a CPU
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # global step counter
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        with tf.name_scope('Input'):

            images_batch, labels_batch, image_name_batch = mip.input_pipeline(filenames=train_files,
                                                                              threads=12,
                                                                              dequeue_leftover=500,
                                                                              batchsize=batch_size,
                                                                              epochs=epochs)

            img_splits = tf.split(axis=0, num_or_size_splits=num_gpus, value=images_batch)
            label_splits = tf.split(axis=0, num_or_size_splits=num_gpus, value=labels_batch)

        with tf.name_scope('dropout'):
            dropout_keep_rate = tf.placeholder(tf.float32)

        # variable used to host reconstruction images at runtime
        with tf.name_scope('debug-images'):
            x_decoded = tf.placeholder(tf.float32, [None,
                                                    299,
                                                    299,
                                                    3])
            tf.summary.image('images', x_decoded, max_outputs=100)

        input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Create an optimizer that performs gradient descent.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:{0}'.format(i)):
                    with tf.name_scope('{0}_{1}'.format(mod.TOWER_NAME, i)) as scope:
                        # Pin variables defined using tf.contrib.slim, to CPU. See ISSUE#1 on this repo for more details
                        with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
                            # Calculate the loss for one tower
                            loss, imgs, labels, preds = _tower_loss(img_splits[i], label_splits[i], dropout_keep_rate,
                                                                    scope=scope)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this tower.
                        grads = optimizer.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # Gradient Synchronization
        # grads = _average_gradients(tower_grads)
        grads = tower_grads[0]

        # Add summaries
        summaries.extend(input_summaries)

        # If new logit layer exclude from restore vars
        variables_to_restore = slim.get_variables_to_restore()

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Create a saver
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)) as sess:

            sess.run(init_op)

            if resume_train:
                ckpt = tf.train.get_checkpoint_state(logdir_train)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Pre-trained network_type restored...')
                print('Restored Global Step:', sess.run(global_step))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)

            # write the meta graph
            tf.train.write_graph(sess.graph_def, logdir, "output_graph_raw.pb",
                                 as_text=True)

            trainstart_time = time.time()

            try:
                while not coord.should_stop():
                    step = int(sess.run(global_step))
                    batchstart_time = time.time()
                    _, loss_val, img, lab, pred = sess.run([train_op, loss, imgs, labels, preds], feed_dict={
                        dropout_keep_rate: dropout})
                    batch_duration = time.time() - batchstart_time

                    assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

                    if step % 10 == 0:
                        examples_per_sec = batch_size / float(batch_duration)
                        format_str = ('%s: step %d, loss = %.9f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print(format_str % (datetime.now(), step, loss_val,
                                            examples_per_sec, batch_duration))

                    if step % 100 == 0:
                        print('Saving Events...')
                        burned_image_stack = utl.labels_on_images(images=img,
                                                                  groundtruth_label=lab,
                                                                  predicted_label=pred)

                        summary_str = sess.run(summary_op, feed_dict={x_decoded: burned_image_stack,
                                                                      dropout_keep_rate: 1.0})
                        train_writer.add_summary(summary_str, step)

                    if step % 1000 == 0:
                        checkpoint_path = os.path.join(logdir_train, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

            except tf.errors.OutOfRangeError:
                train_duration = float(time.time() - trainstart_time)
                print('Done Training!')
                print('Train Duration: {0:.2f}'.format(train_duration))

                print('Saving final model...')
                checkpoint_path = os.path.join(logdir_train, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            except KeyboardInterrupt:
                train_duration = float(time.time() - trainstart_time)
                print('Training Interrupted!')
                print('Train Duration: {0:.2f}'.format(train_duration))

                print('Saving final model...')
                checkpoint_path = os.path.join(logdir_train, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()

    print("freezing graph ...")

    # freeze graph
    utl.create_freeze_graph(global_step=step,
                            logdir=logdir,
                            logdir_train=logdir_train,
                            epochs=epochs,
                            model_type="inception",
                            dropout_rate=dropout,
                            input_graph="output_graph_raw.pb")

    return 0


def run_train(logdir, records_dir, batch_size_pergpu, num_gpus, resume=False):

    # batch size can be user selectable, but per environment config
    # recommendation is to leave this as a default value
    batch_size = batch_size_pergpu * num_gpus

    # training log dir is derived from user supplied temp logdir
    logdir_train = os.path.join(logdir, 'train')

    # create these dir if they do not exist
    if not tf.gfile.Exists(logdir):
        tf.gfile.MakeDirs(logdir)
        tf.gfile.MakeDirs(logdir_train)

    # collect all training files
    train_files = glob.glob(os.path.join(records_dir, '*.tfrecords'))
    assert len(train_files) > 0, 'run() error, no tfrecords available under {0}'.format(records_dir)

    # begin training
    train(logdir=logdir,
          logdir_train=logdir_train,
          train_files=train_files,
          batch_size=batch_size,
          epochs=50,
          num_gpus=num_gpus,
          dropout=0.8,
          learning_rate=0.0001,
          resume_train=resume)

    return 0


if __name__ == "__main__":
    run_train(logdir="/data/f6ds/F6DS-PROBE/test-random/Sceneic-Or-Not-DL/logdir",
              records_dir="/data/f6ds/F6DS-PROBE/test-random/Sceneic-Or-Not-DL/rawdata/tfrecords_cleaned",
              batch_size_pergpu=30,
              num_gpus=4,
              resume=True)
    print("Done!")

