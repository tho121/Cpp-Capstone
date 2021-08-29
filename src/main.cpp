#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    std::cout << "Hello World!" << "\n";

    Mat image = imread("./PetImages/Cat/0.jpg");

    return 0;
}