# Dataset
problem_type                 = 'detection'     # ['classification' | 'detection' | 'segmentation']
dataset_name                 = 'TT100K_detection' # Dataset name
dataset_name2                = None            # Second dataset name. None if not Domain Adaptation
perc_mb2                     = None            # Percentage of data from the second dataset in each minibatch

# Model
model_name                   = 'yolo'          # Model to use: one of 'yolo' or 'tiny-yolo'
freeze_layers_from           = None            # Freeze layers from 0 to this layer during training (Useful for finetunning) [None | 'base_model' | Layer_id]
show_model                   = False           # Show the architecture layers
load_imageNet                = False            # Load Imagenet weights and normalize following imagenet procedure
load_pretrained              = True           # Load a pretrained model for doing finetuning
weights_file                 = '/home/master/M5/results/Experiments/TT100K_detection/yolo_training/weights.hdf5'  # Training weight file name
different_datasets           = False
# Parameters
train_model                  = True            # Train the model
test_model                   = True            # Test the model
pred_model                   = True            # Predict using the model

# Debug
debug                        = False           # Use only few images for debuging
debug_images_train           = 16              # N images for training in debug mode (-1 means all)
debug_images_valid           = 16              # N images for validation in debug mode (-1 means all)
debug_images_test            = 16              # N images for testing in debug mode (-1 means all)
debug_n_epochs               = 2              # N of training epochs in debug mode

# Batch sizes
batch_size_train             = 16             # Batch size during training
batch_size_valid             = 32             # Batch size during validation
batch_size_test              = 32             # Batch size during testing
crop_size_train              = None            # Crop size during training (Height, Width) or None
crop_size_valid              = None            # Crop size during validation
crop_size_test               = None            # Crop size during testing
resize_train                 = (320, 320)      # Resize the image during training (Height, Width) or None
resize_valid                 = (320, 320)      # Resize the image during validation
resize_test                  = (320, 320)      # Resize the image during testing

# Data shuffle
shuffle_train                = True            # Whether to shuffle the training data
shuffle_valid                = False           # Whether to shuffle the validation data
shuffle_test                 = False           # Whether to shuffle the testing data
seed_train                   = 1924            # Random seed for the training shuffle
seed_valid                   = 1924            # Random seed for the validation shuffle
seed_test                    = 1924            # Random seed for the testing shuffle

# Training parameters
optimizer                    = 'rmsprop'       # Optimizer
learning_rate                = 0.00001         # Training learning rate
weight_decay                 = 0.              # Weight decay or L2 parameter norm penalty
n_epochs                     = 5              # Number of epochs during training

# Callback save results
save_results_enabled         = False           # Enable the Callback
save_results_nsamples        = 5               # Number of samples to save
save_results_batch_size      = 5               # Size of the batch
save_results_n_legend_rows   = 1               # Number of rows when showwing the legend

# Callback early stoping
earlyStopping_enabled        = False           # Enable the Callback
earlyStopping_monitor        = 'avg_recall'    # Metric to monitor
earlyStopping_mode           = 'max'           # Mode ['max' | 'min']
earlyStopping_patience       = 100             # Max patience for the early stopping
earlyStopping_verbose        = 0               # Verbosity of the early stopping

# Callback model check point
checkpoint_enabled           = True            # Enable the Callback
checkpoint_monitor           = None            # Metric to monitor
checkpoint_mode              = 'max'           # Mode ['max' | 'min']
checkpoint_save_best_only    = False           # Save best or last model
checkpoint_save_weights_only = True            # Save only weights or also model
checkpoint_verbose           = 1               # Verbosity of the checkpoint

# Callback plot
plotHist_enabled             = True            # Enable the Callback
plotHist_verbose             = 0               # Verbosity of the callback

# Callback LR decay scheduler
lrDecayScheduler_enabled     = False           # Enable the Callback
lrDecayScheduler_epochs      = [5, 10, 20]     # List of epochs were decay is applied or None for all epochs
lrDecayScheduler_rate        = 2               # Decay rate (new_lr = lr / decay_rate). Usually between 2 and 10.

# Data augmentation for training and normalization
norm_imageNet_preprocess           = False     # Normalize following imagenet procedure
norm_fit_dataset                   = False     # If True it recompute std and mean from images. Either it uses the std and mean set at the dataset config file
norm_rescale                       = 1/255.    # Scalar to divide and set range 0-1
norm_featurewise_center            = True     # Substract mean - dataset
norm_featurewise_std_normalization = True     # Divide std - dataset
norm_samplewise_center             = False     # Substract mean - sample
norm_samplewise_std_normalization  = False     # Divide std - sample
norm_gcn                           = False     # Global contrast normalization
norm_zca_whitening                 = False     # Apply ZCA whitening
cb_weights_method                  = None      # Label weight balance [None | 'median_freq_cost' | 'rare_freq_cost']

# Data augmentation for training
da_rotation_range                  = 0          # Rnd rotation degrees 0-180
da_width_shift_range               = 0.       # Rnd horizontal shift
da_height_shift_range              = 0.       # Rnd vertical shift
da_shear_range                     = 0.       # Shear in radians
da_zoom_range                      = 0.       # Zoom
da_channel_shift_range             = 0.         # Channecf.l shifts
da_fill_mode                       = 'nearest'  # Fill mode
da_cval                            = 0.         # Void image value
da_horizontal_flip                 = True      # Rnd horizontal flip
da_vertical_flip                   = False      # Rnd vertical flip
da_spline_warp                     = False      # Enable elastic deformation
da_warp_sigma                      = 10         # Elastic deformation sigma
da_warp_grid_size                  = 3          # Elastic deformation gridSize
da_save_to_dir                     = False      # Save the images for debuging
