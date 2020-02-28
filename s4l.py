# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Self-Supervised Semi-Supervised Learning (S^4L) as described in
    [1] S4L: Self-Supervised Semi-Supervised Learning
        Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, Lucas Beyer
        https://arxiv.org/abs/1905.03670
    It uses the VAT implementation of vat.py and the rotation functions and the
    rotation loss of remixmatch_no_cta_py.
"""

import functools
import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from libml import utils, data, layers, models
from libml.utils import EasyDict
from third_party import vat_utils

FLAGS = flags.FLAGS


class S4L(models.MultiModel):
    """
    Self-Supervised Semi-Supervised Learning (S^4L) as described in
        [1] S4L: Self-Supervised Semi-Supervised Learning
            Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, Lucas Beyer
            https://arxiv.org/abs/1905.03670
    It uses the VAT implementation of vat.py and the rotation loss of
    remixmatch_no_cta_py. The rotation function of remixmatch was replaced with
    a more intuitive implementation.
    All hyperparameters (optimizer, learning rate, weights of the different loss
    terms) follow the original implementation. Instead of decreasing the
    learning rate by a constant factor after a number of iterations, we use a
    cosine learning rate decay.
    """

    def classifier_rot(self, x):
        """
        Adds a second dense output layer for the four different rotation classes
        to a model.
        Args:
            x: Input layer
        Returns: Dense output layer for four classes.
        """
        with tf.variable_scope('classify_rot', reuse=tf.AUTO_REUSE):
            return tf.layers.dense(x, 4, kernel_initializer=tf.glorot_normal_initializer())

    def model(self, batch, lr, wd, ema, warmup_pos, w_super, w_vat, vat_eps,
              w_rot, w_entmin, momentum, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # Labeled training data
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')     # Validation or test data
        y_in = tf.placeholder(tf.float32, [batch] + hwc, 'y')    # Unlabeled training data
        l_in = tf.placeholder(tf.int32, [batch], 'labels')       # Labels for xt
        wd *= lr

        # Start with a warmup, by gradually increasing the VAT loss
        warmup = tf.clip_by_value(tf.to_float(self.step) / (warmup_pos * (FLAGS.train_kimg << 10)), 0, 1)

        # Ramp up learning rate and use cosine learning rate decay
        lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
        lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
        tf.summary.scalar('monitors/lr', lr)

        gpu = utils.get_gpu()

        def random_rotate(x):
            """
            Rotates a batch of images by 0, 90, 180 or 270 degrees.
            Args:
                x: A batch of images with shape [n_batch, n_width, n_height, n_channels]
            Returns: (x_rot, rotation_label)
                X_rot: The images in the batch x, but transformed with a
                perpendicular rotation.
                rotation_label:
            """
            b4 = batch // 4

            # Rotate the images by 0, 90, 180 and 270 degrees and take a quarter
            # of the batchsize each
            x_0 = x[:b4]
            x = tf.image.rot90(x)
            x_90 = x[b4:2 * b4]
            x = tf.image.rot90(x)
            x_180 = x[2 * b4:3 * b4]
            x = tf.image.rot90(x)
            x_270 = x[3 * b4:]
            x_rot = tf.concat([x_0, x_90, x_180, x_270], axis=0)

            # final rotation to restore original x
            x = tf.image.rot90(x)

            # Construct labels as classes from 0 to 3
            l = np.zeros(b4, np.int32)
            labels = tf.constant(np.concatenate([l, l + 1, l + 2, l + 3], axis=0))

            return x_rot, labels

        # Compute rotation loss
        if w_rot > 0:
            # The rotation is randomised, because the batches are shuffled
            rot_y, rot_l = random_rotate(y_in)
            with tf.device(next(gpu)):
                rot_logits = self.classifier_rot(self.classifier(rot_y, training=True, **kwargs).embeds)
            loss_rot = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(rot_l, 4), logits=rot_logits)
            loss_rot = tf.reduce_mean(loss_rot)
            tf.summary.scalar('losses/rot', loss_rot)
        else:
            loss_rot = 0

        # Compute VAT and EntMin losses
        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
        l = tf.one_hot(l_in, self.nclass)
        logits_x = classifier(xt_in, training=True)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Take only first call to update batch norm.
        logits_y = classifier(y_in, training=True)
        delta_y = vat_utils.generate_perturbation(y_in, logits_y, lambda x: classifier(x, training=True), vat_eps)
        logits_student = classifier(y_in + delta_y, training=True)
        logits_teacher = tf.stop_gradient(logits_y)
        loss_vat = layers.kl_divergence_from_logits(logits_student, logits_teacher)
        loss_vat = tf.reduce_mean(loss_vat)
        loss_entmin = tf.reduce_mean(tf.distributions.Categorical(logits=logits_y).entropy())

        # Compute supervised loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l, logits=logits_x)
        loss = tf.reduce_mean(loss)

        tf.summary.scalar('losses/xe', loss)
        tf.summary.scalar('losses/vat', loss_vat)
        tf.summary.scalar('losses/entmin', loss_entmin)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        total_loss = w_super * loss + loss_vat * warmup * w_vat + w_rot * loss_rot + w_entmin * loss_entmin
        train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(total_loss,
                                                                     colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        return EasyDict(
            xt=xt_in, x=x_in, y=y_in, label=l_in, train_op=train_op,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))


def main(argv):
    utils.setup_main()
    del argv  # Unused.
    dataset = data.DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = S4L(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        momentum=FLAGS.momentum,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        warmup_pos=FLAGS.warmup_pos,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        smoothing=FLAGS.smoothing,
        w_super=FLAGS.w_super,
        w_vat=FLAGS.w_vat,
        vat_eps=FLAGS.vat_eps,
        w_rot=FLAGS.w_rot,
        w_entmin=FLAGS.w_entmin,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 2e-4, 'Weight decay.')
    flags.DEFINE_float('momentum', 0.9, 'Momentum used for the optimizer.')
    flags.DEFINE_float('w_super', 0.3, 'Supervised weight.')
    flags.DEFINE_float('w_vat', 0.3, 'VAT weight.')
    flags.DEFINE_float('vat_eps', 6, 'VAT perturbation size.')
    flags.DEFINE_float('w_rot', 0.7, 'Self-supervised rotation weight.')
    flags.DEFINE_float('w_entmin', 0.3, 'Entropy minimization weight.')
    flags.DEFINE_float('warmup_pos', 0.4, 'Relative position at which constraint loss warmup ends.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('smoothing', 0.1, 'Label smoothing.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('batch', 128)
    FLAGS.set_default('lr', 0.1)
    FLAGS.set_default('train_kimg', 1 << 14)
    app.run(main)
