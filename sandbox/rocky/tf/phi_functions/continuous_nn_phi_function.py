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
Neural Network Phi function
'''
class ContinuousMLPPhiFunction(PhiFunction, LayersPowered, Serializable):

    def __init__(
            self,
            env_spec,
            name='MLPPhinet',
            hidden_sizes=(100, 100),
            hidden_nonlinearity=tf.nn.relu,
            action_merge_layer=-2,
            output_nonlinearity=None,
            vs_form=None,
            bn=False):
        Serializable.quick_init(self, locals())

        assert not env_spec.action_space.is_discrete
        self._env_spec = env_spec
        self.vs_form = vs_form
        with tf.variable_scope(name):
            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            l_obs = L.InputLayer(shape=(None, obs_dim), name="obs")
            l_action = L.InputLayer(shape=(None, action_dim), name="action")

            self.obs_rms = RunningMeanStd(shape=(obs_dim, ))

            obz = L.NormalizeLayer(l_obs, rms=self.obs_rms, clip_min=-5., clip_max=5.)

            obs_hidden = L.DenseLayer(
                        obz, 
                        num_units=hidden_sizes[0],
                        nonlinearity=hidden_nonlinearity,
                        name="obs_h%d"%(0))
            print("hidden sizes...", hidden_sizes[0], hidden_sizes[1:])
            act_hidden = L.DenseLayer(
                        l_action, 
                        num_units=hidden_sizes[0],
                        nonlinearity=hidden_nonlinearity,
                        name="act_h%d"%(0))
            merge_hidden = L.OpLayer(obs_hidden, 
                    op=lambda x, y: x + y,
                    shape_op=lambda x, y: y,
                    extras=[act_hidden])

            l_hidden = merge_hidden

            for idx, size in enumerate(hidden_sizes[1:]):
                if bn:
                    l_hidden = batch_norm(l_hidden)
                
                l_hidden = L.DenseLayer(
                    l_hidden,
                    num_units=size,
                    nonlinearity=hidden_nonlinearity,
                    name="h%d" % (idx + 1))
            
            l_output = L.DenseLayer(
                l_hidden,
                num_units=1,
                nonlinearity=output_nonlinearity,
                name="output")
            
            if vs_form is not None:
                if vs_form == 'linear':
                    vs = L.DenseLayer(
                        l_obs, 
                        num_units=1,
                        nonlinearity=None,
                        name='vs')

                elif vs_form == 'mlp':
                    vs = L.DenseLayer(
                        l_obs,
                        num_units=64,
                        nonlinearity=tf.nn.relu,
                        name='hidden_vs')
                    vs = L.DenseLayer(
                        vs,
                        num_units=1,
                        nonlinearity=None,
                        name='vs')
                else:
                    raise NotImplementedError
            
                output_var = L.get_output(l_output, deterministic=True) + \
                                    L.get_output(vs, deterministic=True)
                output_var = tf.reshape(output_var, (-1, ))
            else:
                output_var = L.get_output(l_output, deterministic=True)
                output_var = tf.reshape(output_var, (-1, ))
            
            self._f_phival = tensor_utils.compile_function(
                inputs=[l_obs.input_var, l_action.input_var],
                outputs=output_var
            )
            self._output_layer = l_output
            self._obs_layer = l_obs
            self._action_layer = l_action
            self.output_nonlinearity = output_nonlinearity
        
            if vs_form is not None:
                self._output_vs = vs
                LayersPowered.__init__(self, [l_output,
                                     self._output_vs])
            else:
                LayersPowered.__init__(self, [l_output])
    
    def get_phival(self, observations, actions):
        return self._f_phival(observations, actions)
    
    def get_phival_sym(self, obs_var, action_var, **kwargs):
        if self.vs_form is not None:
            phival, vs = L.get_output(
            [self._output_layer, self._output_vs],
            {self._obs_layer:obs_var, self._action_layer:action_var},
            **kwargs)
            phival = phival + vs
        else:
            phival = L.get_output(
                self._output_layer,
                {self._obs_layer:obs_var, self._action_layer:action_var},
                **kwargs)
        phival = tf.reshape(phival, (-1, ))
        return phival

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

                
    def get_phi_derive_sym(self, obs_var, action_var, **kwargs):
        phival = self.get_phival_sym(obs_var, action_var)
        
        # derivative
        phi_prime = tf.gradients(phival, action_var)[0]
        
        # FIXME: Since this is a nn structure, it is very inefficient to calculate 
        # second order dervative

        return dict(phival=phival, phi_prime=phi_prime)


    def get_gradient_cv_sym(self, obs_var, action_var,
            origin_ad_var, eta_var, policy, **kwargs):
        
        dist_info = policy.dist_info_sym(obs_var)
        mean = dist_info["mean"]
        log_std = dist_info["log_std"]

        grad_info, _ = policy.get_grad_info_sym(obs_var, action_var)

        phi_derives = self.get_phi_derive_sym(obs_var, action_var)

        mu_stein_grad = grad_info["logpi_dmu"] * \
                        tf.expand_dims(origin_ad_var \
                        - eta_var * phi_derives['phival'], axis=1) +  \
                        tf.expand_dims(eta_var,axis=1) * phi_derives['phi_prime']
        
        var_stein_grad = grad_info["logpi_dvar"] * \
                        tf.expand_dims(origin_ad_var \
                        - eta_var * phi_derives['phival'], axis=1)  \
                        + .5 *tf.expand_dims(eta_var, axis=1) \
                        * tf.stop_gradient(grad_info["logpi_dmu"]) * \
                        phi_derives['phi_prime']

        Mu_MSE = tf.reduce_sum(tf.reduce_mean(
                    tf.square(mu_stein_grad - \
                    tf.reduce_mean(mu_stein_grad, axis=0)), axis=0))
                    
        Var_MSE = tf.reduce_sum(tf.reduce_mean(
                    tf.square(var_stein_grad - \
                    tf.reduce_mean(var_stein_grad, axis=0)), axis=0))
        
        return dict(mu_mse=Mu_MSE, 
                    var_mse=Var_MSE)


    def get_adv_cv_sym(self, obs_var, action_var, 
            origin_adv_var, eta_var, policy, **kwargs):
        phival = self.get_phival_sym(obs_var, action_var, **kwargs)
        loss = tf.reduce_mean(tf.square(origin_adv_var - eta_var * phival))

        return loss
    