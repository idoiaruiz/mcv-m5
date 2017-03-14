#!/usr/bin/env python

import subprocess, shlex, os
import argparse
import imp


def run(command, output = None, error_output = None):
    proc = subprocess.Popen(shlex.split(command), stdout = output, stderr = error_output)
    proc.communicate()
    if proc.returncode != 0:
        raise ValueError, "Command '%s' has failed." % command

def main():
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description = 'Model training')
    parser.add_argument('-c', '--config_path', type = str,
                        default = None, help = 'Configuration file')
    parser.add_argument('-e', '--exp_name', type = str,
                        default = None, help = 'Name of the experiment')
    parser.add_argument('-s', '--shared_path', type=str,
                        default = '/home/master/M5/Onofre/', help = 'Path to shared data folder')
                        #default='/share/mastergpu/module5', help='Path to shared data folder')
    parser.add_argument('-l', '--local_path', type=str,
                        default = '/home/master/M5/Onofre/', help='Path to local data folder')
                        #default='/share/mastergpu/module5', help='Path to local data folder')
    arguments = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    for experiment in range(2):
        #Call and train all the networks of the ensemble
        cmd = "python train.py -c " + 'config/tt100k_classif' + str(experiment) + '.py' + " -e " + arguments.exp_name + "/da_" + str(experiment)
        run(cmd)

    

if __name__ == "__main__":
    main()