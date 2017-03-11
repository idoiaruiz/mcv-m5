#!/usr/bin/env python

import subprocess, shlex, os
import argparse
import imp
from bagging_code.copy_images import createDataPaths
from bagging_code.change_line import modify_line_file

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
    
    cf = imp.load_source('config', arguments.config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cmd  = "python train.py -c config/tt100k_classif.py -e pruebaonofre"
    # Create datasets
    if cf.copy_files:
        print ('Creating ' + str(cf.number_bootstraps) + ' of ' + str(cf.samples_per_bootstrap) + 'files')
        createDataPaths(cf.samples_per_bootstrap, cf.number_bootstraps, cf.dataset_name)
    if cf.train_model:  
        for bootstrap in range(cf.number_bootstraps):
            modify_line_file(19, arguments.config_path, 'num_bootstrap = ' + str(bootstrap))
            #Call and train all the networks of the ensemble
            cmd = "CUDA_VISIBLE_DEVICES=0 python train_bagging.py -c " + arguments.config_path + " -e " + arguments.exp_name + "/bootstrap_" + str(bootstrap)
            run(cmd)
    if cf.test_model:
        #Get the weights from training and build the ensemble
        cmd  = "CUDA_VISIBLE_DEVICES=0 python test_bagging.py -c " + arguments.config_path + " -e " + arguments.exp_name
        run(cmd)
    

if __name__ == "__main__":
    main()