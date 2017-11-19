

import os 
from rllab.misc import logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
import tensorflow as tf
import gc


class VPG(BatchPolopt, Serializable):
    """
    Vanilla Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
        self.optimizer_args = optimizer_args

        self.opt_info = None
        super(VPG, self).__init__(env=env, policy=policy, baseline=baseline, **kwargs)

    @overrides
    def init_opt(self):

                
        if self.pf is not None:
            optimizer = [FirstOrderOptimizer(**self.optimizer_args) for i in range(2)]
        else:
            optimizer = FirstOrderOptimizer(**self.optimizer_args)
        
        self.optimizer = optimizer          

        is_recurrent = int(self.policy.recurrent)

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = tensor_utils.new_tensor(
            name='advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )

        advantage_bar = None
        if self.phi:
            advantage_bar = tensor_utils.new_tensor(
            name='advantage_bar',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )

        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        if is_recurrent:
            surr_obj = - tf.reduce_sum(logli * advantage_var * valid_var) / tf.reduce_sum(valid_var)
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            max_kl = tf.reduce_max(kl * valid_var)
        else:
            surr_obj = - tf.reduce_mean(logli * advantage_var)
            mean_kl = tf.reduce_mean(kl)
            max_kl = tf.reduce_max(kl)

        input_list = [obs_var, action_var, advantage_var] + state_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        vars_info = {
            "mean_kl": mean_kl,
            "input_list": input_list,
            "obs_var": obs_var,
            "action_var": action_var,
            "advantage_var": advantage_var,
            "advantage_bar": advantage_bar,
            "surr_loss": surr_obj,
            "dist_info_vars": dist_info_vars,
            "lr": logli,
        }

        if self.qprop:
            eta_var = tensor_utils.new_tensor(
                'eta',
                ndim=1 + is_recurrent,
                dtype=tf.float32,
            )
            qvalue = self.qf.get_e_qval_sym(vars_info["obs_var"], self.policy, deterministic=True)
            qprop_surr_loss = - tf.reduce_mean(vars_info["lr"] *
                vars_info["advantage_var"]) - tf.reduce_mean(
                qvalue * eta_var)
            input_list += [eta_var]
            self.optimizer.update_opt(
                loss=qprop_surr_loss,
                target=self.policy,
                inputs=input_list,
            )
            # calculate covariance between \hat A and \ba A
            control_variate = self.qf.get_cv_sym(obs_var,
                    action_var, self.policy)
            f_control_variate = tensor_utils.compile_function(
                inputs=[obs_var, action_var],
                outputs=control_variate,
            )
            self.opt_info_qprop = dict(
                f_control_variate=f_control_variate,
            )
        elif self.phi:
            # Here we use two unified versions...
            # First we get gradient w.r.t \theta of \mu
            # Then we get gradient w.r.t \theta of \sigma
            # TODO: fix gradwrtmu parameters, find a more elegant method..
            # change to adaptive eta
            eta_var = tensor_utils.new_tensor(
                'eta',
                ndim=1 + is_recurrent,
                dtype=tf.float32,
            )

            phival, mean_var = self.pf._get_e_phival_sym(vars_info["obs_var"], 
                    self.policy, gradwrtmu=True, deterministic=True)
            
            scv_surr_mu_loss = - tf.reduce_mean(vars_info['lr'] * 
                vars_info["advantage_var"]) - tf.reduce_mean(eta_var * phival)
            
            self.optimizer[0].update_opt(
                loss = scv_surr_mu_loss,
                target=self.policy.mean_network,
                inputs=input_list)

            # gradient w.r.t \theta of \sigma
            # using stein gradient variance reduction methods
            self.pf.phi_sigma_full = False
            if self.pf.phi_sigma_full:
                logger.log("Use full stein sigma variance reduction methods")
                # using standard stein variance reduction w.r.t \sigma
                grad_info, dist_info = self.policy.get_grad_info_sym(vars_info['obs_var'],
                                    vars_info['action_var'])
                
                # FIXME: Check if it is a list or something
                phi_primes = tf.gradients(phival, mean_var)[0]
                

                var_gradient = grad_info["logpi_dvar"] * tf.expand_dims(vars_info["advantage_var"], axis=1) - \
                                tf.expand_dims(vars_info['advantage_bar'], axis=1) * grad_info["logpi_dvar"] + \
                                2. * grad_info['logpi_dmu'] * phi_primes
                

                var_loss = - tf.reduce_mean(tf.reduce_sum(tf.stop_gradient(
                            var_gradient) * tf.exp(2.*dist_info["log_std"]), axis=1))

                self.optimizer[1].update_opt(
                    loss=var_loss,
                    target=self.policy.std_network,
                    inputs=input_list + [vars_info['advantage_bar']]
                )

            # the same as Q-prop for sigma updates
            else:
                logger.log("Did not use full stein variance reduction for sigma")
                surr_sigma_loss = - tf.reduce_mean(vars_info['lr'] * 
                vars_info["advantage_var"])

                self.optimizer[1].update_opt(
                    loss=surr_sigma_loss,
                    target=self.policy.std_network,
                    inputs=input_list
                )

            stein_phi = self.pf.get_phi_bar_sym(obs_var, 
                        action_var, self.policy)
            f_stein_phi = tensor_utils.compile_function(
                inputs=[obs_var, action_var],
                outputs=stein_phi,
            )
                
            self.opt_info_phi = dict(
                f_stein_phi=f_stein_phi, 
            )
                
        
        else:
            self.optimizer.update_opt(loss=surr_obj, target=self.policy, inputs=input_list)

        f_kl = tensor_utils.compile_function(
            inputs=input_list + old_dist_info_vars_list,
            outputs=[mean_kl, max_kl],
        )
        self.opt_info = dict(
            f_kl=f_kl,
            target_policy=self.policy,
        )


        self.init_opt_critic()
        # init optimization for phi training
        self.init_opt_phi()

        logger.log("Parameters...")
        for param in self.policy.mean_network.get_params(trainable=True):
            logger.log('mean with name=%s, shape=%s'% (param.name, param.shape))
        
        for param in self.policy.std_network.get_params(trainable=True):
            logger.log('std with name=%s, shape=%s'% (param.name, str(param.shape)))


    # TODO: This need to change for stien variational reduction
    @overrides
    def optimize_policy(self, itr, samples_data):
        logger.log("optimizing policy")
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        )
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        inputs += tuple(state_info_list)
        if self.policy.recurrent:
            inputs += (samples_data["valids"],)
        if self.qprop:
            inputs += (samples_data["etas"], )
            # Qprop for vpg
            logger.log("Using Qprop optimizer")
        
        if self.phi:
            logger.log("Using Stein CV optimizers")

            mu_optimizer = self.optimizer[0]
            dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
            mu_loss_before = mu_optimizer.loss(inputs)
            gc.collect()
            mu_optimizer.optimize(inputs)
            gc.collect()
            mu_loss_after = mu_optimizer.loss(inputs)
            logger.record_tabular("Mu_LossBefore", mu_loss_before)
            logger.record_tabular("Mu_LossAfter", mu_loss_after)
            
            # add extra stein identity for sigma optimization
            if self.pf.phi_sigma_full:
                tmp_inputs = inputs + (samples_data["advantage_bar"],)
                sigma_optimizer = self.optimizer[1]
                sigma_loss_before = sigma_optimizer.loss(tmp_inputs)
                # gc.collect()
                sigma_optimizer.optimize(tmp_inputs)
                # gc.collect()
                sigma_loss_after = sigma_optimizer.loss(tmp_inputs)
                logger.record_tabular("Sigma_LossBefore", sigma_loss_before)
                logger.record_tabular("Sigma_LossAfter", sigma_loss_after)
            else:
                sigma_optimizer = self.optimizer[1]
                sigma_loss_before = sigma_optimizer.loss(inputs)
                # gc.collect()
                sigma_optimizer.optimize(inputs)
                # gc.collect()
                sigma_loss_after = sigma_optimizer.loss(inputs)
                logger.record_tabular("Sigma_LossBefore", sigma_loss_before)
                logger.record_tabular("Sigma_LossAfter", sigma_loss_after)
        else:        

            optimizer = self.optimizer
            dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
            loss_before = optimizer.loss(inputs)
            gc.collect()
            optimizer.optimize(inputs)
            gc.collect()
            loss_after = optimizer.loss(inputs)
            logger.record_tabular("LossBefore", loss_before)
            logger.record_tabular("LossAfter", loss_after)

        mean_kl, max_kl = self.opt_info['f_kl'](*(list(inputs) + dist_info_list))
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
