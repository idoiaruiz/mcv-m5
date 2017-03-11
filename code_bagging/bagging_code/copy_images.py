import numpy as np
import os
from shutil import copyfile, copytree, rmtree
from random import shuffle
from change_line import modify_line_file

def createDataPaths(samples_per_bootstrap, number_bootstraps, dataset):
    #Path where the dataset is
    data_dir = '/share/mastergpu/module5/Datasets/classification/' + dataset
    #New path for data
    dst_dir = '/home/master/M5/Onofre/Datasets/classification/' + dataset
    
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    copyData(data_dir, dst_dir, samples_per_bootstrap, number_bootstraps)

def copyData(data_dir, dst_dir, samples_per_bootstrap, number_bootstraps):
    
    train_dir = data_dir + '/train'                
    n_images_train  = samples_per_bootstrap
    classFolders = os.listdir(train_dir)
    
    #Create a dataset for each bootstrap
    for i in range(number_bootstraps):
       
        dst_dir_bootstrap = dst_dir + '/bootstrap_' + str(i)

        all_images = []
        for folder in classFolders:
            #Create folder for each class(previously erase if necessary)
            if os.path.exists(dst_dir_bootstrap + '/train/' + folder):
                rmtree(dst_dir_bootstrap + '/train/' + folder)
            
            os.makedirs(dst_dir_bootstrap + '/train/' + folder)
            
            #Take name of all images in class
            images_class = os.listdir(train_dir + '/' + folder)
            images_class = [folder + '/' + image for image in images_class]
            #Append names of all classes
            all_images = np.append(all_images, images_class)
        
        shuffle(all_images)
        #Take samples for a bootstrap uniformly random with replacement
        images_bootstrap = np.random.choice(all_images, size = samples_per_bootstrap, replace = True)
        
        #Copy images
        for trainim in images_bootstrap:
            copyfile(train_dir + '/' + trainim, dst_dir_bootstrap + '/train/' + trainim)
        #Copy validation and test sets
        if os.path.exists(dst_dir_bootstrap + '/valid'):
            rmtree(dst_dir_bootstrap + '/valid')
        if os.path.exists(dst_dir_bootstrap + '/test'):
            rmtree(dst_dir_bootstrap + '/test')        
        copytree(data_dir + '/valid', dst_dir_bootstrap + '/valid')
        copytree(data_dir + '/test', dst_dir_bootstrap + '/test')
        
        copyfile(data_dir + '/config.py', dst_dir_bootstrap + '/config.py')
        new_location_config = dst_dir_bootstrap + '/config.py'
        modify_line_file(10, new_location_config, 'n_images_train  = ' + str(n_images_train))
        
if __name__ == "__main__":
    samples_per_bootstrap = 400
    number_bootstraps = 1
    dataset = 'TT100K_trafficSigns'
    createDataPaths(samples_per_bootstrap, number_bootstraps, dataset)
    
    
    
    
    