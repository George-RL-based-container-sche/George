"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.
Policy Gradient, Reinforcement Learning.
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate,
                 suffix):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.suffix = suffix

        """
        self.ep_obs, self.ep_as, self.ep_rs: observation, action, reward, recorded for each batch
        """
        self.ep_obs, self.ep_as, self.ep_rs, self.ep_ss = [], [], [], []

        """
        self.tput_batch: record throughput for each batch, used to show progress while training
        self.tput_persisit, self.episode: persist to record throughput, used to be stored and plot later

        """
        self.tput_batch = []
        self.tput_persisit = []
        self.ss_batch = []
        self.ss_persisit = []
        self.episode = []
        self.ss_perapp_persisit = []
        self.ss_coex_persisit = []
        self.ss_sum_persisit = []
        # TODO self.vio = []: violation

        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        restore = ['Actor' + self.suffix + '/fc1' + self.suffix + '/kernel:0', 'Actor' + self.suffix + '/fc1' + self.suffix + '/bias:0', 'Actor' + self.suffix + '/fc2' + self.suffix + '/kernel:0',
                   'Actor' + self.suffix + '/fc2' + self.suffix + '/bias:0']
        restore_var = [v for v in tf.all_variables() if v.name in restore]
        self.saver = tf.train.Saver(var_list=restore_var)
        # self.saver = tf.train.Saver()

    def _build_net(self):
        with tf.variable_scope("Actor" + self.suffix):
            with tf.name_scope('inputs' + self.suffix):
                self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name='observation' + self.suffix)
                self.tf_acts = tf.placeholder(tf.int32, [None, ], name='actions_num' + self.suffix)
                self.tf_vt = tf.placeholder(tf.float32, [None, ], name='actions_value' + self.suffix)
                self.tf_safe = tf.placeholder(tf.float32, [None, ], name='safety_value' + self.suffix)
                self.entropy_weight = tf.placeholder(tf.float32, shape=(), name='entropy_weight_clustering' + self.suffix)

            # layer1 = tf.layers.dense(
            #      inputs=self.tf_obs,
            #      units=128,
            #      activation= tf.nn.tanh,
            #      kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.3),
            #      bias_initializer= tf.constant_initializer(0.1),
            #      name='fc0'+self.suffix
            #  )
            layer = tf.layers.dense(inputs=self.tf_obs, units=128, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1), name='fc1' + self.suffix)
            # layer3 = tf.layers.dense(
            #     inputs=layer,
            #     units=128 * 1,
            #     activation=tf.nn.tanh,
            #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            #     bias_initializer=tf.constant_initializer(0.1),
            #     name='fc3' + self.suffix
            # )
            all_act = tf.layers.dense(inputs=layer, units=self.n_actions, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                bias_initializer=tf.constant_initializer(0.1), name='fc2' + self.suffix)

            self.all_act_prob = tf.nn.softmax(all_act, name='act_prob'+self.suffix)

            with tf.name_scope('loss'+self.suffix):
                neg_log_prob = tf.reduce_sum(-tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * tf.one_hot(indices=self.tf_acts, depth=self.n_actions), axis=1)
                loss = tf.reduce_mean(neg_log_prob * self.tf_vt)
                loss += self.entropy_weight * tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * self.all_act_prob, axis=1))
                self.entro = self.entropy_weight * tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(self.all_act_prob, 1e-30, 1.0)) * self.all_act_prob, axis=1))
                self.loss = loss

            with tf.name_scope('train'+self.suffix):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
                # decay_rate =0.99999 # 0.999
                # learning_rate = 1e-1
                # self.train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation})  # (4,) ->(1,4)
        if np.isnan(prob_weights).any():
            test=1
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action, prob_weights

    def choose_action_determine(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation})  # (4,) ->(1,4)
        action = np.argmax(prob_weights.ravel())
        return action, prob_weights

    def store_training_samples_per_episode(self, s, a, r, ss):
        self.ep_obs.extend(s)
        self.ep_as.extend(a)
        self.ep_rs.extend(r)
        self.ep_ss.extend(ss)

    def store_tput_per_episode(self, tput, safe, episode, list_check_per_app, list_check_coex, list_check_sum):
        self.tput_batch.append(tput)
        self.tput_persisit.append(tput)
        self.ss_batch.append(safe)
        self.ss_persisit.append(safe)
        self.episode.append(episode)
        self.ss_perapp_persisit.append(list_check_per_app)
        self.ss_coex_persisit.append(list_check_coex)
        self.ss_sum_persisit.append(list_check_sum)

    def learn(self, epoch_i, entropy_weight, IfPrint=False):
        discounted_ep_rs_norm_reward = self._discount_and_norm_rewards()
        discounted_ep_rs_norm = discounted_ep_rs_norm_reward

        if np.isnan(discounted_ep_rs_norm).any():
            test=1
        if np.isnan(self.ep_obs).any():
            test=1
        if np.isnan(self.ep_as).any():
            test=1

        _, loss, all_act_prob, entropy = self.sess.run([self.train_op, self.loss, self.all_act_prob, self.entro], feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),
            self.tf_acts: np.array(self.ep_as),
            self.tf_vt: discounted_ep_rs_norm,
            self.entropy_weight: entropy_weight
        })
        if np.isnan(loss).any():
            test=1
        if IfPrint:
            print("epoch: %d, safety: %f,, tput: %f, entropy: %f, loss: %f" % (
                epoch_i, np.mean(self.ep_rs), np.mean(self.tput_batch), np.mean(entropy), np.mean(loss)))

        self.ep_obs, self.ep_as, self.ep_rs, self.ep_ss = [], [], [], []
        self.tput_batch = []
        self.ss_batch = []

    def _discount_and_norm_rewards(self):
        """
        Normalize reward per batch
        :return:
        """
        discounted_ep_rs = np.array(self.ep_rs)
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        if np.std(discounted_ep_rs) != 0:
            discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


    def _discount_and_norm_rewards_ss(self):
        """
        Normalize reward per batch
        :return:
        """
        discounted_ep_ss = np.array(self.ep_ss)
        discounted_ep_ss -= np.mean(discounted_ep_ss)
        if np.std(discounted_ep_ss) != 0:
            discounted_ep_ss /= np.std(discounted_ep_ss)
        return discounted_ep_ss

    def save_session(self, ckpt_path):
        self.saver.save(self.sess, ckpt_path)

    def restore_session(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)
        # self.saver = tf.train.Saver()
