from sandbox.rocky.tf.phi_functions.base import PhiFunction
from rllab.core.serializable import Serializable
import numpy as np 

from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.layers import batch_norm
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.misc.mpi_running_mean_std import RunningMeanStd

import tensorflow as tf 
import sandbox.rocky.tf.core.layers as L 

class ContinuousLinearPhiFunction(PhiFunction, LayersPowered, Serializable):
    
    def __init__(
            self,
            env_spec, 
            name='Phinet',
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu, 
            action_merge_layer=-2,
            output_nonlinearity=None,
            bn=False):
        Serializable.quick_init(self, locals())

        assert not env_spec.action_space.is_discrete
        self._env_spec = env_spec
        
        with tf.variable_scope(name):
            l_obs = L.InputLayer(shape=(None,env_spec.observation_space.flat_dim), name="obs")
            l_action = L.InputLayer(shape=(None, env_spec.action_space.flat_dim), name="action")

            n_layers = len(hidden_sizes) + 1

            if n_layers > 1:
                action_merge_layer = \
                    (action_merge_layer % n_layers + n_layers) % n_layers
            else:
                action_merge_layer = 1

            # self.obs_rms = RunningMeanStd(shape=(env_spec.observation_space.flat_dim, ))

            # obz = L.NormalizeLayer(l_obs, rms=self.obs_rms, clip_min=-5., clip_max=5.)
            obz = l_obs
            obs_hidden = L.DenseLayer(
                        obz, 
                        num_units=hidden_sizes[0],
                        nonlinearity=hidden_nonlinearity,
                        name="obs_h%d"%(0))

            act_hidden = L.DenseLayer(
                        l_action, 
                        num_units=hidden_sizes[0],
                        nonlinearity=hidden_nonlinearity,
                        name="act_h%d"%(0))
            
            merge_hidden = L.OpLayer(obs_hidden, 
                    op=lambda x, y: x + y,
                    shape_op=lambda x, y: x,
                    extras=[act_hidden])
        
            l_hidden = merge_hidden
        
            for idx, size in enumerate(hidden_sizes[1:]):
                if bn:
                    l_hidden = batch_norm(l_hidden)
                
                l_hidden = L.DenseLayer(
                    l_hidden,
                    num_units=size,
                    nonlinearity=hidden_nonlinearity,
                    name="h%d" % (idx + 1)
                )

            
            # for idx, size in enumerate(hidden_sizes):
            #     if bn:
            #         l_hidden = batch_norm(l_hidden)
                
            #     if idx == action_merge_layer:
            #         l_hidden = L.ConcatLayer([l_hidden, l_action])

            #     l_hidden = L.DenseLayer(
            #         l_hidden,
            #         num_units=size,
            #         nonlinearity=hidden_nonlinearity,
            #         name="h%d" % (idx + 1)
            #     )

            # if action_merge_layer == n_layers:
            #     l_hidden = L.ConcatLayer([l_hidden, l_action])
            
            l_output = L.DenseLayer(
                l_hidden, 
                num_units=1,
                nonlinearity=output_nonlinearity,
                name="output"
            )

            output_var = L.get_output(l_output, deterministic=True)
            output_var = tf.reshape(output_var, (-1,))

            self._f_phival = tensor_utils.compile_function([l_obs.input_var, l_action.input_var], output_var)
            self._output_layer = l_output
            self._obs_layer = l_obs
            self._action_layer = l_action
            self.output_nonlinearity = output_nonlinearity


            LayersPowered.__init__(self, [l_output])
    

    def get_phival(self, observations, actions):
        return self._f_phival(observations, actions)

    def get_phival_sym(self, obs_var, action_var, **kwargs):
        phival = L.get_output(
            self._output_layer,
            {self._obs_layer: obs_var, self._action_layer:action_var},
            **kwargs)

        return tf.reshape(phival, (-1, ))
    
    def get_e_phival(self, observations, policy):
        if isinstance(policy, StochasticPolicy):
            agent_info = policy.dist_info(observations)
            means, log_stds = agent_info['mean'], agent_info['log_std']
            
            #NOTE: that we use stochastic policy to sample action
            randoms = np.random.normal(size=means.shape)
            
            actions = means + np.exp(log_stds) * randoms
            phivals = self.get_phival(observations, actions)
        
        else:
            actions, _ = policy.get_actions(observations)
            phivals = self.get_phival(observations, actions)

        return phivals
    
    # compute \phi 
    def _get_e_phival_sym(self, obs_var, policy, gradwrtmu=False, **kwargs):
        if isinstance(policy, StochasticPolicy):
            agent_info = policy.dist_info_sym(obs_var)
            mean_var, log_std_var = agent_info['mean'], agent_info['log_std']
            
            if gradwrtmu:
                return self.get_phival_sym(obs_var, mean_var, **kwargs), mean_var

            else:
                action_var = tf.random_normal(shape=tf.shape(mean_var)) * tf.exp(log_std_var) + mean_var
                return self.get_phival_sym(obs_var, action_var, **kwargs), action_var

        else:
            action_var = policy.get_action_sym(obs_var)
            return self.get_phival_sym(obs_var, action_var, **kwargs), action_var
        
    def get_e_phival_sym(self, obs_var, policy, **kwargs):
        return self._get_e_phival_sym(obs_var, policy, **kwargs)[0]

    def get_phi_bar_sym(self, obs_var, action_var, policy, **kwargs):
        phival, action0 = self._get_e_phival_sym(obs_var, policy, gradwrtmu=True, deterministic=True, **kwargs)
        qprimes = tf.gradients(phival, action0)[0]
        deltas = action_var - action0
        
        return tf.reduce_sum(deltas * qprimes, 1)
    

    def get_opt_eta_sym(self, obs_var, action_var, 
            origin_adv_var, policy, **kwargs):
        grad_info, _ = policy.get_grad_info_sym(obs_var, action_var)
        
        phival, action0 = self._get_e_phival_sym(obs_var, policy, gradwrtmu=True, deterministic=True, **kwargs)
        phi_primes = tf.gradients(phival, action0)[0]
        deltas = action_var - action0
        
        phi_bar = tf.reduce_sum(deltas * phi_primes, 1)
        
        origin_grad_mu = grad_info["logpi_dmu"] * tf.expand_dims(origin_adv_var, axis=1)

        phi_grad_mu = - grad_info['logpi_dmu'] * \
                    tf.expand_dims(phi_bar, axis=1) + \
                    phi_primes

        # We only consider the relation of mu to caluclate lambda
        covar = tf.reduce_sum(tf.reduce_mean((origin_grad_mu - \
                tf.reduce_mean(origin_grad_mu, axis=0)) * \
                (phi_grad_mu - tf.reduce_mean(phi_grad_mu, axis=0)), axis=0))

        phi_mse = tf.reduce_sum(tf.reduce_mean(
                    tf.square(phi_grad_mu - \
                    tf.reduce_mean(phi_grad_mu, axis=0)), axis=0))
        
        return covar / (phi_mse + 1e-8)

   
    # FIXME: The advantage var should be the original advantages...
    # rather than the one already minus advantage bar
    # FIXME: There might be some problem: we use the variance of each samles

    # compute control variate 
    def get_adv_cv_sym(self, obs_var, action_var, 
            origin_adv_var, eta_var, policy, **kwargs):
        phi_bar = self.get_phi_bar_sym(obs_var, action_var, policy, **kwargs)
        
        reward = origin_adv_var - eta_var * phi_bar
        loss = tf.reduce_mean(tf.square(reward))

        return loss

    # Here advantage var should be original advantage_var
    def get_gradient_cv_sym(self, obs_var, action_var, 
            origin_ad_var, eta_var, policy, **kwargs):

        phival, action0 = self._get_e_phival_sym(obs_var, policy, gradwrtmu=True, deterministic=True, **kwargs)
        phi_primes = tf.gradients(phival, action0)[0]
        deltas = action_var - action0
        
        adv_bar = tf.reduce_sum(deltas * phi_primes, 1)

        # get gradient information w.r.t \mu \Sigma
        grad_info, _ = policy.get_grad_info_sym(obs_var, action_var)

        mu_stein_grad = grad_info["logpi_dmu"] * \
                        tf.expand_dims(origin_ad_var \
                        - eta_var * adv_bar, axis=1) +  \
                        tf.expand_dims(eta_var,axis=1) * phi_primes
        
        var_stein_grad = grad_info['logpi_dvar'] * \
                        tf.expand_dims(origin_ad_var \
                        - eta_var * adv_bar, axis=1)

        Mu_MSE = tf.reduce_sum(tf.reduce_mean(
                    tf.square(mu_stein_grad - \
                    tf.reduce_mean(mu_stein_grad, axis=0)), axis=0))
                    
        Var_MSE = tf.reduce_sum(tf.reduce_mean(
                    tf.square(var_stein_grad - \
                    tf.reduce_mean(var_stein_grad, axis=0)), axis=0))
        
        return dict(mu_mse=Mu_MSE, 
                    var_mse=Var_MSE)