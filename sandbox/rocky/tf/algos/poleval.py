import os
from rllab.misc import ext
import numpy as np
import rllab.misc.logger as logger
from rllab.core.serializable import Serializable
import tensorflow as tf
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer

class Poleval():
    """
    Base class defining methods for policy evaluation.
    """
    def init_critic(self,
            min_pool_size=10000,
            replay_pool_size=1000000,
            replacement_prob=1.0,
            qf_batch_size=32,
            qf_weight_decay=0.,
            qf_update_method='adam',
            qf_learning_rate=1e-3,
            qf_use_target=True,
            qf_mc_ratio = 0,
            qf_residual_phi = 0,
            soft_target_tau=0.001,
            **kwargs):
        self.soft_target_tau = soft_target_tau
        self.min_pool_size = min_pool_size
        self.replay_pool_size = replay_pool_size
        self.replacement_prob = replacement_prob
        self.qf_batch_size = qf_batch_size
        self.qf_weight_decay = qf_weight_decay
        self.qf_update_method = \
            FirstOrderOptimizer(
                update_method=qf_update_method,
                learning_rate=qf_learning_rate,
            )
        self.qf_learning_rate = qf_learning_rate
        self.qf_use_target = qf_use_target

        self.qf_mc_ratio = qf_mc_ratio
        if self.qf_mc_ratio > 0:
            self.mc_y_averages = []

        self.qf_residual_phi = qf_residual_phi
        if self.qf_residual_phi > 0:
            self.residual_y_averages = []
            self.qf_residual_loss_averages = []

        self.qf_loss_averages = []
        self.q_averages = []
        self.y_averages = []
    

    def init_phi(self, 
            pf_batch_size=32,
            pf_weight_decay=0.,
            pf_update_method='adam',
            **kwargs):
        
        self.pf_batch_size = pf_batch_size
        self.pf_weight_decay = 0.
        self.pf_update_method = \
            FirstOrderOptimizer(
                update_method=pf_update_method,
                learning_rate=self.pf_learning_rate
            )

    def log_critic_training(self):
        if self.qf is None: return
        if len(self.q_averages) == 0: return
        all_qs = np.concatenate(self.q_averages)
        all_ys = np.concatenate(self.y_averages)
        average_q_loss = np.mean(self.qf_loss_averages)
        logger.record_tabular('AverageQLoss', average_q_loss)
        logger.record_tabular('AverageQ', np.mean(all_qs))
        logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
        logger.record_tabular('AverageY', np.mean(all_ys))
        logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))
        logger.record_tabular('AverageAbsQYDiff',
                              np.mean(np.abs(all_qs - all_ys)))
        if self.qf_mc_ratio > 0:
            all_mc_ys = np.concatenate(self.mc_y_averages)
            logger.record_tabular('AverageMcY', np.mean(all_mc_ys))
            logger.record_tabular('AverageAbsMcY', np.mean(np.abs(all_mc_ys)))
            self.mc_y_averages = []
        if self.qf_residual_phi > 0:
            all_residual_ys = np.concatenate(self.residual_y_averages)
            average_q_residual_loss = np.mean(self.qf_residual_loss_averages)
            logger.record_tabular('AverageResQLoss', average_q_residual_loss)
            logger.record_tabular('AverageResY', np.mean(all_residual_ys))
            logger.record_tabular('AverageAbsResY', np.mean(np.abs(all_residual_ys)))
            logger.record_tabular('AverageAbsResQYDiff',
                              np.mean(np.abs(all_qs - all_residual_ys)))
            self.residual_y_averages = []
            self.qf_residual_loss_averages = []
        self.qf_loss_averages = []
        self.q_averages = []
        self.y_averages = []
    
    #TODO: add log of phi training
    def log_phi_training(self):
        if self.pf is None: return
        if len(self.pf_loss_averages) == 0: return
        
        average_phi_loss = np.mean(self.pf_loss_averages)
        logger.record_tabular('AveragePhiLoss', average_phi_loss)

        self.pf_loss_averages = []
        

    def init_opt_critic(self):
        if self.qf is None: logger.log("qf is None"); return

        if self.qf_use_target:
            logger.log("[init_opt] using target qf.")
            target_qf = Serializable.clone(self.qf, name="target_qf")
        else:
            logger.log("[init_opt] no target qf.")
            target_qf = self.qf

        obs = self.qf.env_spec.observation_space.new_tensor_variable(
            'qf_obs',
            extra_dims=1,
        )
        action = self.qf.env_spec.action_space.new_tensor_variable(
            'qf_action',
            extra_dims=1,
        )
        yvar = tf.placeholder(dtype=tf.float32, shape=[None], name='ys')

        qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
                               sum([tf.reduce_sum(tf.square(param)) for param in
                                    self.qf.get_params(regularizable=True)])

        qval = self.qf.get_qval_sym(obs, action)

        qf_loss = tf.reduce_mean(tf.square(yvar - qval))
        qf_input_list = [yvar, obs, action]
        qf_output_list = [qf_loss, qval]

        # set up residual gradient method
        if self.qf_residual_phi > 0:
            next_obs = self.qf.env_spec.observation_space.new_tensor_variable(
                'qf_next_obs',
                extra_dims=1,
            )
            rvar = tf.placeholder(dtype=tf.float32, shape=[None], name='rs')
            terminals = tf.placeholder(dtype=tf.float32, shape=[None], name='terminals')
            discount = tf.placeholder(dtype=tf.float32, shape=(), name='discount')
            qf_loss *= (1. - self.qf_residual_phi)
            next_qval = self.qf.get_e_qval_sym(next_obs, self.policy)
            residual_ys = rvar + (1.-terminals)*discount*next_qval
            qf_residual_loss = tf.reduce_mean(tf.square(residual_ys-qval))
            qf_loss += self.qf_residual_phi * qf_residual_loss
            qf_input_list += [next_obs, rvar, terminals, discount]
            qf_output_list += [qf_residual_loss, residual_ys]

        # set up monte carlo Q fitting method
        if self.qf_mc_ratio > 0:
            mc_obs = self.qf.env_spec.observation_space.new_tensor_variable(
                'qf_mc_obs',
                extra_dims=1,
            )
            mc_action = self.qf.env_spec.action_space.new_tensor_variable(
                'qf_mc_action',
                extra_dims=1,
            )
            mc_yvar = tf.placeholder(dtype=tf.float32, shape=[None], name='mc_ys')
            mc_qval = self.qf.get_qval_sym(mc_obs, mc_action)
            qf_mc_loss = tf.reduce_mean(tf.square(mc_yvar - mc_qval))
            qf_loss = (1.-self.qf_mc_ratio)*qf_loss + self.qf_mc_ratio*qf_mc_loss
            qf_input_list += [mc_yvar, mc_obs, mc_action]

        qf_reg_loss = qf_loss + qf_weight_decay_term
        self.qf_update_method.update_opt(
            loss=qf_reg_loss, target=self.qf, inputs=qf_input_list)
        qf_output_list += [self.qf_update_method._train_op]

        f_train_qf = tensor_utils.compile_function(
            inputs=qf_input_list,
            outputs=qf_output_list,
        )

        self.opt_info_critic = dict(
            f_train_qf=f_train_qf,
            target_qf=target_qf,
        )
    
    def init_opt_phi(self):
        if self.pf is None: return
        is_recurrent = int(self.policy.recurrent)
        obs = self.pf.env_spec.observation_space.new_tensor_variable(
            'phi_obs',
            extra_dims=1,
        )
        action = self.pf.env_spec.action_space.new_tensor_variable(
            'phi_action',
            extra_dims=1,
        )

        origin_advantages = tensor_utils.new_tensor(
            name="origin_adv_var",
            ndim = 1+ is_recurrent,
            dtype=tf.float32,
        )
        
        eta_var = tensor_utils.new_tensor(
            name="eta_var",
            ndim=1 + is_recurrent,
            dtype=tf.float32
        )


        pf_weight_decay_term = .5 * self.pf_weight_decay * \
                                sum([tf.reduce_sum(tf.square(param)) for param in
                                    self.pf.get_params(regularizable=True)])
        
        self.pf_loss_averages = []
        
        # TODO: Add multiple choice for Variance Reduction

        if self.use_gradient_vr:
            logger.log("using gradient as variance reduction")
            phi_mse = self.pf.get_gradient_cv_sym(obs, action, 
                        origin_advantages, eta_var, self.policy)
            phi_mse = phi_mse["mu_mse"] + phi_mse["var_mse"]
        else:
            logger.log("using reward as variance reduction")
            phi_mse = self.pf.get_adv_cv_sym(obs, action, 
                        origin_advantages, eta_var, self.policy)
        
        pf_loss = tf.reduce_mean(phi_mse)
         
        pf_input_list = [obs, action, origin_advantages, eta_var]
        pf_output_list = [pf_loss]
        pf_reg_loss = pf_loss + pf_weight_decay_term
        self.pf_update_method.update_opt(loss=pf_reg_loss, 
                        target=self.pf, inputs=pf_input_list)
        
        # parameters of phi
        params = self.pf.get_params(trainable=True)
        
        for param in params:
            logger.log("parameter of phi %s, shape=%s"%(param.name, param.shape))

        pf_output_list += [self.pf_update_method._train_op]

        f_train_pf = tensor_utils.compile_function(
            inputs=pf_input_list,
            outputs=pf_output_list)
        
        self.opt_train_phi = dict(
            f_train_pf=f_train_pf
        )

        # optimal eta calculation
        
        opt_eta = self.pf.get_opt_eta_sym(obs, action, 
                origin_advantages, self.policy)
        
        self.opt_eta_phi = tensor_utils.compile_function(
            inputs=[obs, action, origin_advantages],
            outputs=opt_eta)


    
    def do_critic_training(self, itr, batch, samples_data=None):

        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        target_qf = self.opt_info_critic["target_qf"]
        if self.qf_dqn:
            next_qvals = target_qf.get_max_qval(next_obs)
        else:
            target_policy = self.opt_info["target_policy"]
            next_qvals = target_qf.get_e_qval(next_obs, target_policy)

        ys = rewards + (1. - terminals) * self.discount * next_qvals
        inputs = (ys, obs, actions)

        if self.qf_residual_phi:
            inputs += (next_obs, rewards, terminals, self.discount)
        if self.qf_mc_ratio > 0:
            mc_inputs = ext.extract(
                samples_data,
                "qvalues", "observations", "actions"
            )
            inputs += mc_inputs
            self.mc_y_averages.append(mc_inputs[0])

        qf_outputs = self.opt_info_critic['f_train_qf'](*inputs)
        qf_loss = qf_outputs.pop(0)
        qval = qf_outputs.pop(0)
        if self.qf_residual_phi:
            qf_residual_loss = qf_outputs.pop(0)
            residual_ys = qf_outputs.pop(0)
            self.qf_residual_loss_averages.append(qf_residual_loss)
            self.residual_y_averages.append(residual_ys)

        if self.qf_use_target:
            target_qf.set_param_values(
                target_qf.get_param_values() * (1.0 - self.soft_target_tau) +
                self.qf.get_param_values() * self.soft_target_tau)

        self.qf_loss_averages.append(qf_loss)
        self.q_averages.append(qval)
        self.y_averages.append(ys)
    
    def do_phi_training(self, itr, indices=None, samples_data=None):
        
        batch_samples = samples_data
        '''
        dict(
            observations=samples_data["observations"][indices],
            actions=samples_data["actions"][indices],
            origin_advantages=samples_data["origin_advantages"][indices],)
        '''
        inputs = ext.extract(
            batch_samples, 
            "observations", 
            "actions", 
            "origin_advantages", 
            "etas",)

        # the following code is useless
        # FIXME: write a better version of this
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]

        inputs += tuple(state_info_list)

        #TODO: add recurrent
        if self.policy.recurrent:
            inputs += (samples_data["valid"], )
        
        pf_outputs = self.opt_train_phi['f_train_pf'](*inputs)
        pf_loss = pf_outputs.pop(0)
        self.pf_loss_averages.append(pf_loss)
        
