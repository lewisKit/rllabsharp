from sandbox.rocky.tf.phi_functions.base import PhiFunction
from rllab.core.serializable import Serializable
import numpy as np 

from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.core.layers import batch_norm
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.misc.mpi_running_mean_std import RunningMeanStd

import tensorflow as tf
import sandbox.rocky.tf.core.layers as L 

'''
    Quardratic function for Phi with formula
    Q(a, s) = -(a - f(s))A(a - f(s)) 
'''
class ContinuousQuadraticPhiFunction(PhiFunction, LayersPowered, Serializable):
    
    def __init__(
            self,
            env_spec,
            name='QuadraticPhinet',
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=None,
            vs_form=None,
            bn=False,
            A=None,
            init_a=1.0,
            a_parameterization='exp'):
        Serializable.quick_init(self, locals())

        assert not env_spec.action_space.is_discrete
        self._env_spec = env_spec
        self.vs_form = vs_form
        with tf.variable_scope(name):
            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            l_act = L.InputLayer(shape=(None, action_dim), name="action")
            action_var =l_act.input_var
            l_obs = L.InputLayer(shape=(None, obs_dim), name="obs")
            
            self.obs_rms = RunningMeanStd(shape=(obs_dim, ))
            
            obz = L.NormalizeLayer(l_obs, rms=self.obs_rms)
            l_hidden = l_obs
            hidden_sizes += (action_dim, )
            
            for idx, size in enumerate(hidden_sizes):
                if bn:
                    l_hidden = batch_norm(l_hidden)

                l_hidden = L.DenseLayer(
                    l_hidden,
                    num_units=size,
                    nonlinearity=hidden_nonlinearity,
                    name="h%d" % (idx + 1)
                )

            obs_var = l_obs.input_var 
            fs = l_hidden # fs_network.output_layer

            if A is not None:
                l_A_param = A.output_layer
            else:
                if a_parameterization == 'exp':
                    init_a_param = np.log(init_a) - .5
                elif a_parameterization == 'softplus':
                    init_a_param = np.log(np.exp(init_a) - 1)
                else:
                    raise NotImplementedError

                l_log_A = L.ParamLayer(
                    l_obs,
                    num_units=action_dim,
                    param=tf.constant_initializer(init_a_param),
                    name="diagonal_a_matrix",
                    trainable=True)
            if vs_form is not None:
                raise NotImplementedError
            
            
            self._l_log_A = l_log_A
            self.a_parameterization = a_parameterization
            self.fs = fs
            
            if vs_form is not None:
                self._output_vs = vs
                LayersPowered.__init__(self, [self.fs, 
                            self._l_log_A, self._output_vs])
            else:
                LayersPowered.__init__(self, [self.fs, self._l_log_A])

            output_var = self.get_phival_sym(obs_var, action_var)

            self._f_phival = tensor_utils.compile_function(
                    inputs=[obs_var, action_var],
                    outputs=output_var
            )

    def get_phival(self, observations, actions):
        return self._f_phival(observations, actions)

    def get_phival_sym(self, obs_var, action_var, **kwargs):
        if self.vs_form is not None:
            fs, l_log_A, vs =L.get_output(
                [self.fs, self._l_log_A, 
                self._output_vs],
                obs_var)
            phival = - tf.reduce_sum(
                tf.exp(l_log_A) * tf.square(
                action_var - fs), axis=1, keep_dims=True)
            phival += vs
        else:
            fs, l_log_A = L.get_output(
                [self.fs, self._l_log_A],
                obs_var)
        
            phival = - tf.reduce_sum(
                tf.exp(l_log_A) * tf.square(
                action_var - fs), axis=1)
        
        return tf.reshape(phival, (-1, ))

    
    def get_e_phival(self, observations, policy):
        if isinstance(policy, StochasticPolicy):
            agent_info = policy.dist_info(observations)
            means, log_stds = agent_info['mean'], agent_info['log_std']

            randoms = n.random.normal(size=means.shape)
            actions = means + np.exp(log_stds) * randoms
        else:
            actions, _ = policy.get_actions(observations)
        
        phivals = self.get_phival(observations, actions)
        return phivals    
    
    def get_e_phival_sym(self, obs_var, policy, gradwrtmu=False, **kwargs):
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
            
    def get_phi_derive_sym(self, obs_var, action_var, **kwargs):
        fs, l_log_A = L.get_output(
            [self.fs, self._l_log_A],
            obs_var, **kwargs)
        
        phival = tf.reduce_sum(
            -tf.exp(l_log_A) * tf.square(
            action_var - fs), axis=1)
        phival = tf.reshape(phival, (-1, ))
        
        # Derivative
        phi_prime = -2. * tf.exp(l_log_A) * (action_var - fs)

        phi_double_prime = -2 * tf.exp(l_log_A)

        return dict(phival=phival, phi_prime=phi_prime,
                phi_double_prime=phi_double_prime)
    
    def get_adv_cv_sym(self, obs_var, action_var,
            origin_adv_var, eta_var, policy, **kwargs):
            phival = self.get_phival_sym(obs_var, 
                            action_var, **kwargs)
            loss = tf.reduce_mean(tf.square(origin_adv_var - eta_var * phival))

            return loss
    
    def get_opt_eta_sym(self, obs_var, action_var, 
                        origin_adv_var, policy, **kwargs):
        grad_info, _ = policy.get_grad_info_sym(obs_var, action_var)

        phi_derives = self.get_phi_derive_sym(obs_var, action_var)

        origin_grad_mu = grad_info["logpi_dmu"] * tf.expand_dims(origin_adv_var, axis=1)

        phi_grad_mu = - grad_info['logpi_dmu'] * \
                    tf.expand_dims(phi_derives['phival'], axis=1) + \
                    phi_derives['phi_prime']
        
        # We only consider the relation of mu to caluclate lambda
        covar = tf.reduce_sum(tf.reduce_mean((origin_grad_mu - \
                tf.reduce_mean(origin_grad_mu, axis=0)) * \
                (phi_grad_mu - tf.reduce_mean(phi_grad_mu, axis=0)), axis=0))

        phi_mse = tf.reduce_sum(tf.reduce_mean(
                    tf.square(phi_grad_mu - \
                    tf.reduce_mean(phi_grad_mu, axis=0)), axis=0))
        
        return covar / (phi_mse + 1e-8)       

    
    def get_gradient_cv_sym(self, obs_var, action_var, 
            origin_ad_var, eta_var, policy, **kwargs):
        dist_info = policy.dist_info_sym(obs_var)
        mean = dist_info["mean"]
        log_std = dist_info["log_std"]

        grad_info, _ = policy.get_grad_info_sym(obs_var, action_var)

        phi_derives =self.get_phi_derive_sym(obs_var, action_var)
 
        mu_stein_grad = grad_info["logpi_dmu"] * \
                        tf.expand_dims(origin_ad_var \
                        - eta_var * phi_derives['phival'], axis=1) +  \
                        tf.expand_dims(eta_var,axis=1) * phi_derives['phi_prime']

        var_stein_grad = grad_info["logpi_dvar"] * \
                        tf.expand_dims(origin_ad_var \
                        - eta_var * phi_derives['phival'], axis=1)  \
                        + .5 *tf.expand_dims(eta_var, axis=1)  \
                        * phi_derives['phi_double_prime']
        
        Mu_MSE = tf.reduce_sum(tf.reduce_mean(
                    tf.square(mu_stein_grad - \
                    tf.reduce_mean(mu_stein_grad, axis=0)), axis=0))
                    
        Var_MSE = tf.reduce_sum(tf.reduce_mean(
                    tf.square(var_stein_grad - \
                    tf.reduce_mean(var_stein_grad, axis=0)), axis=0))
        
        return dict(mu_mse=Mu_MSE, 
                    var_mse=Var_MSE)


