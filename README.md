# CPPND: Capstone - Cat and Dog Classifier using OpenCV and Bag-of-Features SIFT with SVM!

This is my capstone project for Udacity's C++ Nanodegree, featuring the use of C++14 features like move semantics, smart pointers, lambda expressions, and multithreading.

These features are combined in creating an image classifier using the OpenCV C++ library. The algorithm used is the Bag-of-Features (BoF) algorithm, an image variant of the Bag-of-Words algorithm. Features are extracted from images using SIFT and used to build the BoF vocabulary. This vocabulary is used to extract training data from the images and then passed into a Support Vector Machine (SVM). The label for a cat is 1 and a dog is -1. Images are included in the PetImages folder with 2500 images for each. More images can be downloaded from https://www.microsoft.com/en-us/download/details.aspx?id=54765

When running the built executable, it expects 3 parameters; sample size, test size, number of threads. 

Sample size should be 2000, test size should be 200, and number of threads should be 4.

Unfortunately, classification performance is terrible, with only ~30% of test images correctly classified.

# Classes

## BagTrainer

Responsible for feature extraction and building the BoF vocabulary. Prints a message for every 100 images computed and descriptor extracted.

## ImageContainer

Responsible for loading and storing images. Converts them to grayscale and resizes them to 300x300 to save on memory. Prints a message for every 100 images loaded.

## SVMTrainer

Responsible for creating the SVM, training, and predicting.

## ScopeTimer

Responsible for tracking how long a task takes. Uses a template to modify the time unit. Specifically, it measures the lifetime of a specific scope by creating a timestamp on creation and on destruction it prints the time since. Child classes like ScopeTimerSec and ScopeTimerMin automatically use second and minute units automatically and append the appropriate suffix. Since the output is an integer, values less than 1 get truncated to 0.

# Expected Output

Images are loaded. Loading cat images use it's own thread, while loading dog images uses another thread.

![image](https://user-images.githubusercontent.com/4165980/132085729-904270f9-a506-4dc8-ab2c-1b049c26abda.png)

Descriptors are extracted from the images/samples. Images are divided per thread. The BoF is trained. 

![image](https://user-images.githubusercontent.com/4165980/132086210-a6fd8848-a93c-42c0-9e72-8cbd9d86a750.png)

Using the trained BoF, training data is extracted from the images and used to train the SVM. Test images are similarly processed.

![image](https://user-images.githubusercontent.com/4165980/132086237-fe806174-050d-4819-9412-ae9e19f655fe.png)

Test results are printed.

![image](https://user-images.githubusercontent.com/4165980/132086265-de52ac6a-dd9c-4fa0-b477-ade185c5a8c6.png)


## Dependencies for Running Locally
* cmake >= 3.7
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.4
  * The OpenCV 4.4.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.4.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./CatDogClassifier 2000 200 4`.

## Other

All rubric points are addressed.

