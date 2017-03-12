
import numpy as np

def test_ensemble(models, valid_gen, test_gen, cf):
#Doing
#predictions = model.predict_generator(validation_generator, val_samples=total_samples)
#will go trough the whole dataset once and only once (assuming there are total_samples in the folder). 
#To each prediction the 'real class' is validation_generator.classes.
#I want to use this information to build a confusion matrix


    # Compute validation metrics
    prediction_prob_valid = models[0].test_validation(valid_gen)
    # Compute test metrics
    prediction_prob_test = models[0].test_test(test_gen)
    for bootstrap in range(1, cf.number_bootstraps):
            # Compute validation prediction
            prediction_prob_valid_bootstrap = models[bootstrap].test_validation(valid_gen)
            # Compute test prediction
            prediction_prob_test_bootstrap = models[bootstrap].test_test(test_gen)
            
            prediction_prob_valid = np.add(prediction_prob_valid_bootstrap, prediction_prob_valid)
            prediction_prob_test = np.add(prediction_prob_test_bootstrap, prediction_prob_test)
     
    prediction_prob_valid = np.divide(prediction_prob_valid, np.float(cf.number_bootstraps))
    prediction_prob_test = np.divide(prediction_prob_test, np.float(cf.number_bootstraps))   
    
    predicted_label_valid = np.argmax(prediction_prob_valid, axis = 1)
    predicted_label_test = np.argmax(prediction_prob_test, axis = 1)
    gt_labels_valid = valid_gen.classes
    gt_labels_test = test_gen.classes
    
    positive_valid = len(gt_labels_valid[gt_labels_valid == predicted_label_valid])
    length_valid = len(gt_labels_valid)
    accuracy_valid = np.float(positive_valid)/np.float(length_valid)
    print 'Validation accuracy: ' + str(accuracy_valid) + '%'
    positive_test = len(gt_labels_test[gt_labels_test == predicted_label_test])
    length_test = len(gt_labels_test)
    accuracy_test = np.float(positive_test)/np.float(length_test)
    print 'Validation test: ' + str(accuracy_test) + '%'
    