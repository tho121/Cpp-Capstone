#include "SVMTrainer.h"
#include "ImageContainer.h"
#include "BagTrainer.h"
#include "ScopeTimer.h"

#include <iostream>
#include <string>
#include <chrono>

using namespace std;

int main()
{
    std::cout << "Hello World!" << "\n";

    const int sampleSize = 5000;
    const Size imageSize(300, 300);
    const int dictionarySize = 100;
    const int threadCount = 4;
    const int maxBagIterations = 1000;
    const int maxSVMIterations = 10000;
    const int testSize = 500;

    vector<string> paths = {"../PetImages/Cat/%d.jpg", "../PetImages/Dog/%d.jpg"};

    ImageContainer train_ic(paths, sampleSize, imageSize);

    BagTrainer bt(dictionarySize, threadCount);
    {
        ScopeTimer t("Computed descriptors: ");
        bt.computeDescriptors(train_ic.getImages());
    }
    
    {
        ScopeTimer t("Computed vocab: ");
        bt.setVocab(maxBagIterations);
    }

    Mat labels;
    Mat trainData;
    {
        ScopeTimer t("Prepared training data: ");
        vector<Mat> catImages = train_ic.getImages(0);
        Mat catData = bt.getDescriptors(catImages);

        vector<Mat> dogImages = train_ic.getImages(1);
        Mat dogData = bt.getDescriptors(dogImages);

        for(int i = 0; i < catData.rows; ++i) {labels.push_back(1);}
        for(int i = 0; i < dogData.rows; ++i) {labels.push_back(-1);}

        vconcat(catData, dogData, trainData);
    }
    
    SVMTrainer st(maxSVMIterations);
    {
        ScopeTimer t("Training SVM Done: ");
        st.train(trainData, labels);
    }
    
    ImageContainer test_ic(paths, testSize, imageSize, sampleSize);

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