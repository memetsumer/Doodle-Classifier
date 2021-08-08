# Simple-Doodle-Classifier
Predicting doodles using Python, Flask, p5.js


* To run code, you should download at least 3 dataset from [doodle datasets](https://github.com/googlecreativelab/quickdraw-dataset "quickdraw-dataset"), In this project, _numpy bitmap files (.npy)_ are used to work easily with Python.
  * Download datasets which can be any kind of doodle,
  * Put dataset files into Doodle Classifier directory.  
  * Open **doodle_classifier.py** then change .npy file names and variable names as you want.
  ```python
     # for example, change each of these names to what your doodle name is.
     full_numpy_bitmap_ice_cream.npy
     self.icecream_dataset
     self.label_icecream
     self.icecream_training_set
  ``` 
  * Open **main.py** and change each prediction sentence to what your doodle name is.
     ```python
     # for example,
     guess = "It's an ice cream!"
     ``` 
  * Run **main.py** and it's ready!
