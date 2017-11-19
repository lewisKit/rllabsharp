import tensorflow as tf
import numpy as np
import os
import multiprocessing
import subprocess
import sys

def worker_function(i):
    subprocess.check_output(i, shell = True)

index = int(sys.argv[1])

commands = []

envs = ['Humanoid-v1', 'HalfCheetah-v1',  'Ant-v1']

meta_com = 'python algo_gym_stub.py --seed 250 --algo_name %s  --env_name %s --batch_size %d  '

# Q-prop
command = meta_com%('qprop', envs[0], 5000)+'--qprop_eta_option adapt1'
commands.append(command)

command = meta_com%('qprop', envs[1], 5000)+'--qprop_eta_option adapt1'
commands.append(command)

command = meta_com%('qprop', envs[2], 5000)+'--qprop_eta_option adapt1'
commands.append(command)

print(commands[index])

commands = [commands[index]]

pool = multiprocessing.Pool(len(commands))
result = pool.imap(worker_function, commands)
pool.close()
pool.join()
