#include "SVMTrainer.h"
#include "ImageContainer.h"
#include "BagTrainer.h"

#include <iostream>
#include <string>

using namespace std;

int main()
{
    std::cout << "Hello World!" << "\n";

    const int sampleSize = 2000;
    const int maxBagIterations = 1000;
    const int maxSVMIterations = 1000;
    const int testSize = 100;

    vector<string> paths = {"../PetImages/Cat/%d.jpg", "../PetImages/Dog/%d.jpg"};

    ImageContainer train_ic(paths, sampleSize);

    BagTrainer bt;
    bt.calcFeatures(train_ic.getImages());
    bt.setVocab(maxBagIterations);

    vector<Mat> catImages = train_ic.getImages(0);
    Mat catData = bt.getDescriptors(catImages);

    vector<Mat> dogImages = train_ic.getImages(1);
    Mat dogData = bt.getDescriptors(dogImages);

    Mat labels;
    for(int i = 0; i < catData.rows; ++i) {labels.push_back(1);}
    for(int i = 0; i < dogData.rows; ++i) {labels.push_back(-1);}

    Mat trainData;
    vconcat(catData, dogData, trainData);

    SVMTrainer st(maxSVMIterations);
    st.train(trainData, labels);

    ImageContainer test_ic(paths, testSize, sampleSize);

    Mat cat_testData = bt.getDescriptors(test_ic.getImages(0));
    Mat dog_testData = bt.getDescriptors(test_ic.getImages(1));

    cout << "Testing Cat labels" << endl;

    Mat catLabels(cat_testData.rows, 1, CV_32F, 1);
    st.test(cat_testData, catLabels);

    cout << "Testing Dog labels" << endl;

    Mat dogLabels(dog_testData.rows, 1, CV_32F, -1);
    st.test(dog_testData, dogLabels);

    return 0;
}