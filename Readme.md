# Multiclass Classification 
## analyzing the dataset :
 
	- it is imbalanced --> possible solutions: 
				- traditional machine learning such as RandomForest which also can consider imbalance dataset (needs finetuning hyperparameter)
				- upsampling or downsampling
				- SMOTE (for images it may not work->SMRT has been proposed for images )	
				- normal data augmentation (doesnt seem good idea)
				- just fit the model with class weights ( augmenting the rare-sample classes is better choice)
				** what i did at last: augment rare sample class with augmentor library, add those images to train data , and fit the model
	- data is noisy and different size 
			 --> possible solutions: 
				- just resize and let the model find the best features 
				- help the model by preprocessing the images((sharp enhancement, median blur, bilateral))[(contour, dilate, erose, threshold is tried but not applied  )] 
				- use a pretrained denoising AE(as feature extractor) so the reconstructed images can be considered as input images for Deep model((Not At all! noise is not normal, or with defined distributuon))
				- whatever chosen, should also applied to test test
				- i create a function to chooose the input[different options] for the deep model
				- sizes changes between H: constant=32 W: (9~28)
##  Model Architecture :
	- Given for intermediate layer
	- preprocessing : just as said above
	- postprocessing :
		 - Global average pooling (less parameters) or Flatten (more parameters)
		 - Dense layer with 10 neurons
		 - but it was said dont use any trainable layer : i dont know if Dense layer is accepted or not
## Model compile and fit:
	- loss function : possible solutions: 
		- CategoricalCrossentropy 
		- focal loss (better for imbalanced dataset)
	- optimization  : I used Adam  (both with constant learning rate and schedule)
	- metric: 
		- Categoricalaccuracy
		- since it is imbalance data : precision , auc, ... 
	- i create a function to choose between different options (normal fitting, fit by class weight , fit in kfold cross val , fit with imagedatagenerator)
## First try :
    - choose raw model 
    - choose raw dataset (just resized and normalized)
    - split train dataset to train and val
    - fit the model until achieving 100% accuracy (overfitting on train data)
    - results: train acc :100% - train loss: 1.21e-4 
    - now try the model on validation dataset :
	ressult : 77% accuracy , 1.95 loss--> sign of overfitting 
					     ---> different solutions :

						  - maybe data is not shuffled (increase buffersize in tf.data.dataset.)
						  - getting more data (natural or artificial[data augmentations(image data generator, imgaug, augmentor, albumentation, ...])
						  - fixing optimization algorithm(train for more epoch- change lr- SVM[hinge loss]
						  - Dropout , BN, weight decay 
now lets try different approaches based on some tricks mentioned as the following:

## tricks:

 ### one time setup:  
 	             1) activation functions: Given for all the layers , expect the last : which I chose implicitely as softmax (from_logits=True in CategoricalCrossentropy loss)
		     2) data preprocess: 
			- handcrafted preprocess
			- normaliazation (zero_centered , unit variance , scaling to [0-1]
			- same normalization should be applied to test
		
		     3) weight initialization (and bias)
			- (W):with Relu --> Hu initialization is better
			- (W):within skip connection : (better to appy Hu for first conv and Zeros for the other conv) [ i appled: hu-zero-zero] (i mean block A)
			- (W): I applied : Hu uniform for all ReLUs- Hu normal for Linear activation by turning of kernel constraint to avoid vanishing or exploding gradient - Glorot uniform for last Dense layer
			- (B):since it is imbalanced : final layer init bias =log (class_weight) : here 200/100=0.2 [weight_decay=0]
			- (B):if balanced data and multicalss: -log(C) : C number of classes
		     4) Regularization
			- BN 
			      - but since we are not allowed to add trainable layers, we should whether use it in (trainable=False) mode or 
        			not using it since when it is not trainable it avoids learning necessary features !
			      - usually the order of applying dropout in CONV layer:  conv-BN-Relu (after conv and before relu! but not a strict rule)	
			      - it increases speed
			


			- Data augmentation
			      - help by giving more data(artificially generated) for model fitting
			      - as my problem is imbalances I used augmentor library to generate new samples and increment the size of rare samples (2, 5, 8)
			- L2 (weight decay)
			      - I didnt apply it
			- Dropout: 
				- (usually after high parametric layers )
				- it is said: for instance in Resnet after GAP layer it is not used!
				- usually after activation 
				- it causes training learning curve more noisy

### Training Dynamics: 
		      1) learning rate scheduler:
			  - if it is tried: better to turn on weight decay (and epochs~100)
			  - usually not applied for ADAM (but i applied) :start from 0.001 and decreases
		          - high learning rate leads to nan loss , and low learing rate causes a plateau
		      2) Batchsize:
			  - if small : will not hurt but long training time
   			  - if large : maybe cant find best weights
			  - when you double it you can double learning rate as well
			  - i chose 64
		      3) Epochs:
			  - early stopping callback
		      4) we can also use grid search[more efficeintly: random search] to find best hyperparameter (using KerasClassifier in sckicit-learn wrapper)
			  in this case train the model for epoches 1 ~ 5 in Grid search, and chooose the best parameter and train the model
	                  using this set of hyperparameters for longer epochs (without LR decay)
		      5) cross validation:
			  train several models in parallel, and save the best model checkpoint 
### after training 
		      1) model ensembles:
			  - we can save several checkpoints of model during train and average them
		      2) transfer learning
			 
 
## analyzing learning curves:
   1) if loss is in plateau:
	- maybe learning rate is so slow
	- maybe inappropriate wight initialization (it slows down training process) [if so high : explode - if so low: vanish]
	- maybe lr schedule can help
	- if train and val loss are both high --> maybe higher layer weights ~0
   2) if loss just ossilate around a limit : training data is not representative in comparision to validation data
   3) if loss start to decrease and then stopped: valid data is not representative -->increase test size
   4) high oscilation in val loss : 1) batch size 2) non-scale data

* underfitting: 1) maybe too simple architecture
		2) not good features
		3) we applied so many regularization
* if val loss < train loss --> not a good split of data
	 
	
