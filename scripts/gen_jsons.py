import numpy as np
import json
import os


# seed = 2
set_seeds = [250, 47, 59, 97]
# set_seeds = [213, 131, 157, 731, 191]
batch_size=5000
env_name = 'Walker2d-v1'
pf_hidden_sizes='100x50'
set_pf_iters=[250, 400, 800]
set_pf_learning_rate=[1e-3, 1e-4, 4e-5]
set_pf_cls=["quadratic"]
use_gradient_vrs=[False]
vs_form=None
pf_phi_lam_option="ones"
qprop_eta_option='adapt1'

#set_step_size = [1e-1, 1e-2, 1e-3]
#set_learning_rate=[1e-3, 1e-4]
#set_qf_learning_rate=[1e-3, 1e-4]

# learning_rate = 0.001
# qf_learning_rate = 0.0001
# step_size = 0.1



index = 1

exp_prefixes=['mergerllab_reward']
algo_name='cfpo'
for exp_prefix, use_gradient_vr, pf_cls in zip(exp_prefixes, use_gradient_vrs, set_pf_cls):
    for seed in set_seeds:
        if not os.path.exists('cfpo_%s_seed=%d'%(env_name, seed)):
            os.makedirs('cfpo_%s_seed=%d'%(env_name, seed))
        for pf_learning_rate in set_pf_learning_rate:
            for pf_iters in set_pf_iters:
                data = {
                    'exp_prefix': exp_prefix,
                    'use_gradient_vr':use_gradient_vr,
                    'seed':seed,
                    'pf_learning_rate':pf_learning_rate,
                    'batch_size':batch_size,
                    'env_name':env_name,
                    'algo_name':algo_name,
                    'pf_iters':pf_iters,
                    'vs_form':vs_form,
                    'pf_phi_lam_option':pf_phi_lam_option,
                    'pf_cls':pf_cls,
                    'pf_hidden_sizes':pf_hidden_sizes,
                    }
                with open('cfpo_%s_seed=%d/cfpo_%s_%d.json'%(env_name, seed, env_name, index), 'w') as outfile:
                    json.dump(data, outfile)

                index += 1


# exp_prefix='mergerllab_qprop'
# algo_name='qprop'

# for seed in set_seeds:
#     data = {
#         'exp_prefix':exp_prefix,
#         'seed':seed,
#         'qprop_eta_option':qprop_eta_option,
#         'batch_size':batch_size,
#         'env_name':env_name,
#         'algo_name':algo_name}

#     with open('../configurations/qprop_%s_%d.json'%(env_name, index), 'w') as outfile:
#                 json.dump(data, outfile)
        
#     index += 1



# for step_size in set_step_size:
#     for learning_rate in set_learning_rate:
#         for qf_learning_rate in set_qf_learning_rate:
#             data = {
#                 'exp_prefix':exp_prefix,
#                 'seed':seed,
#                 'batch_size':batch_size,
#                 'env_name':env_name,
#                 'algo_name':algo_name,
#                 'qprop_eta_option':qprop_eta_option,
#                 'step_size':step_size,
#                 'learing_rate':learning_rate,
#                 'qf_learning_rate':qf_learning_rate
#             }

#             with open('../configurations/qprop_%s_%d.json'%(env_name, index), 'w') as outfile:
#                 json.dump(data, outfile)
        
#             index += 1
