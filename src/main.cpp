#include "SVMTrainer.h"
#include "ImageContainer.h"
#include "BagTrainer.h"
#include "ScopeTimer.h"

#include <iostream>
#include <condition_variable>
#include <string>
#include <thread>

using namespace std;

int main(int argc, char *argv[])
{
    cout << "Cat and Dog Classifier using OpenCV and BoF Sift with SVM!" << endl;

    const int sampleSize = atoi(argv[1]);// 5000;
    const int testSize = atoi(argv[2]);// 500;
    const int threadCount = atoi(argv[3]); //4;

    const Size imageSize(0, 0);
    const int dictionarySize = 100;
    const int maxBagIterations = 1000;
    const int maxSVMIterations = 10000;
    

    vector<string> paths = {"../PetImages/Cat/%d.jpg", "../PetImages/Dog/%d.jpg"};

    //load images
    ImageContainer train_ic(paths, sampleSize, imageSize);

    //set up Bag-of-Features with SIFT trainer
    BagTrainer bagTrainer(dictionarySize, threadCount);

    {   //compute features
        ScopeTimer<chrono::milliseconds> t("Computed descriptors: ");
        bagTrainer.computeDescriptors(train_ic.getImages());
    }
    
    {   //set vocab and train
        ScopeTimer<chrono::milliseconds> t("Computed vocab: ");
        bagTrainer.setVocab(maxBagIterations);
    }

    Mat trainData, labels;
    {   //extract features and assign labels
        ScopeTimerMin t("Prepared training data: ");
        vector<Mat> catImages = train_ic.getImages(0);
        Mat catData = bagTrainer.getDescriptors(catImages);

        vector<Mat> dogImages = train_ic.getImages(1);
        Mat dogData = bagTrainer.getDescriptors(dogImages);

        for(int i = 0; i < catData.rows; ++i) {labels.push_back(1);}
        for(int i = 0; i < dogData.rows; ++i) {labels.push_back(-1);}

        vconcat(catData, dogData, trainData);
    }
    
    //create SVM
    SVMTrainer svmTrainer(maxSVMIterations);

    {   //train SVM
        ScopeTimerMin t("Training SVM Done: ");
        svmTrainer.train(trainData, labels);
    }
    
    //get test images
    ImageContainer test_ic(paths, testSize, imageSize, sampleSize);

    std::condition_variable condition;
    std::mutex mutex;

    Mat testMat;
    int testIndex;
    
    //setup testing which waits for available data
    auto testData = [&testMat, &svmTrainer, &testIndex, &condition, &mutex]() mutable 
    {
        int testCount = 0;

        do{
            std::unique_lock<std::mutex> lck(mutex);
            condition.wait(lck, [&testMat]{ return testMat.rows > 0; });

            testIndex == 0 ? cout << "Testing Cat labels" << endl
                : cout << "Testing Dog labels" << endl;

            Mat labels(testMat.rows, 1, CV_32F, testIndex == 0 ? 1 : -1 );
            svmTrainer.test(testMat, labels);
            testMat = Mat();    //reset
            testCount++;
            
        } while (testCount < 2);
    };

    //setup test data processing, notifies test cycle when data is ready
    auto getTestData = [&testMat, &bagTrainer, &test_ic, &testIndex, &condition, &mutex] (int index) mutable 
    {
        Mat testData = bagTrainer.getDescriptors(test_ic.getImages(index));

        std::unique_lock<std::mutex> lck(mutex);
        testMat = testData;
        testIndex = index;

        condition.notify_one();
    };

    std::thread testThread(testData);
    std::thread catThread(getTestData, 0);
    std::thread dogThread(getTestData, 1);

    testThread.join();
    catThread.join();
    dogThread.join();

    return 0;
}