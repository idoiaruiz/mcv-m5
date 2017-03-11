
import numpy as np

def test_ensemble(models, valid_gen, test_gen, cf):
#Doing
#predictions = model.predict_generator(validation_generator, val_samples=total_samples)
#will go trough the whole dataset once and only once (assuming there are total_samples in the folder). 
#To each prediction the 'real class' is validation_generator.classes.
#I want to use this information to build a confusion matrix


    # Compute validation metrics
    prediction_labels_valid = models[0].test(valid_gen)
    # Compute test metrics
    prediction_labels_test = models[0].test(test_gen)
    for bootstrap in range(1, cf.number_bootstraps):
            # Compute validation prediction
            prediction_labels_valid_bootstrap = models[bootstrap].test(valid_gen)
            # Compute test prediction
            prediction_labels_test_bootstrap = models[bootstrap].test(test_gen)
            
            prediction_labels_valid = np.add(prediction_labels_valid_bootstrap, prediction_labels_valid)
            prediction_labels_test = np.add(prediction_labels_test_bootstrap, prediction_labels_test)
     
    prediction_labels_valid = np.divide(prediction_labels_valid, np.float(cf.number_bootstraps))
    prediction_labels_test = np.divide(prediction_labels_test, np.float(cf.number_bootstraps))   
    
    gt_labels_valid = valid_gen.classes()
    gt_labels_test = test_gen.classes()