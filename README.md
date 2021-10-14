# Image recognition system
This is the template code for an image recognition system 

# What does the code in this repo do
The code in this repo in particular performs data exporting, data preprocessing and model training, and perform inferences with the saved model. 

# Set up environment 
* Requirement: ```anaconda```, ```pip```, ```python 3```
* It is a good practice to create a new conda environment and here is how to create one:
  ```
  conda create -n "name_of_new_environment" python==3.8.5
  conda activate "name_of_new_environment"
  ```
* Install dependencies (use Anaconda Prompt): 
  ```
  pip install --upgrade pip 
  pip install -r requirements.txt 
  ```
  
# How to train the model

Ensure your current directory has all the files from this repo.
```
python image-training-keras.py

```

Alternatively, one can run the notebook "image classification - low level API.ipynb"


# How to do inference

Ensure your current directory has all the files from this repo.
```
python image-inference.py

```

  

