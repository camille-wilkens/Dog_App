[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"

# Dog Breed Classifier Capstone Project
Convolutional Neural Networks (CNN) Algorithm for a Dog Breed Classifier

See my Medium [Blog](https://medium.com/p/3379a873fe45/edit) for more details about this project.

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#file)
4. [Instructions](#instructions)
5. [Findings](#findings)
6. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation <a name="installation"></a>
1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/udacity/dog-project.git
cd dog-project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

6. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

7. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
8. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
9. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

10. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

11. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

12. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.

__NOTE:__ While some code has already been implemented to get you started, you will need to implement additional functionality to successfully answer all of the questions included in the notebook. __Unless requested, do not modify code that has already been included.__


## Project Motivation<a name="motivation"></a>
This project uses Convolutional Neural Networks (CNNs) and involves creating a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed. The image below displays sample output of the finished project.
 

![Sample Output][image1]


## File Descriptions<a name="file"></a>

This is the directory structure of this project:  The main file that you will working with is the jupyter notebook - dog_app.ipynb

		│   CODEOWNERS
		│   dog_app.html
		│   dog_app.ipynb
		│   extract_bottleneck_features.py
		│   LICENSE.txt
		│   README.md
		│
		├───bottleneck_features
		│
		├───haarcascades
		│       haarcascade_frontalface_alt.xml
		│
		├───images
		│
		├───requirements
		│       dog-linux-gpu.yml
		│       dog-linux.yml
		│       dog-mac-gpu.yml
		│       dog-mac.yml
		│       dog-windows-gpu.yml
		│       dog-windows.yml
		│       requirements-gpu.txt
		│       requirements.txt
		│
		├───sample_images
		└───saved_models
			weights.best.from_scratch.hdf5
			weights.best.Resnet50.hdf5
			weights.best.VGG16.hdf5
			

The dog_app.ipynb notebook contains the following steps to create and test the CNN Algorithm:

		Step 0: Import Datasets

		Step 1: Detect Humans

		Step 2: Detect Dogs

		Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

		Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)

		Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

		Step 6: Write your Algorithm

		Step 7: Test Your Algorithm

## Instructions <a name="instructions"></a>

1. Create a sample_images directory on the project's root directory and upload any image files that you would like to perform a dog breed classification using this algorithm.

2. Open the jupyter notebook - dog_app.ipynb on the project's root directory.

	

## Findings<a name="findings"></a>

I was able to achieve a test accuracy score of 81.5% with my CNN Algorithm.   I was expecting to see some incorrect classifications and I believe due to the quality of the pictures may have led to some of the incorrect classifications, my sample images included a horse that was labeled as a dog and human picture not being detected correctly.

Possible Improvements:

	1) Try another bottleneck features from a different pre-trained model such as Xception 
	
	2) Adjust weight initialization parameters 
	
	3) Rescale Images with a variety of scaling factors to achieve higher classification confidence

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Copyright (c) 2017 Udacity, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rightsto use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


