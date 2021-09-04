#include "SVMTrainer.h"
#include "ImageContainer.h"
#include "BagTrainer.h"
#include "ScopeTimer.h"

#include <iostream>
#include <condition_variable>
#include <string>
#include <thread>

using namespace std;

int main()
{
    std::cout << "Hello World!" << "\n";

    const int sampleSize = 200;
    const Size imageSize(300, 300);
    const int dictionarySize = 100;
    const int threadCount = 4;
    const int maxBagIterations = 1000;
    const int maxSVMIterations = 10000;
    const int testSize = 500;

    vector<string> paths = {"../PetImages/Cat/%d.jpg", "../PetImages/Dog/%d.jpg"};

    ImageContainer train_ic(paths, sampleSize, imageSize);

    BagTrainer bagTrainer(dictionarySize, threadCount);

    {
        ScopeTimer<chrono::milliseconds> t("Computed descriptors: ");

        bagTrainer.computeDescriptors(train_ic.getImages());
    }
    
    {
        ScopeTimer<chrono::milliseconds> t("Computed vocab: ");
        bagTrainer.setVocab(maxBagIterations);
    }

    Mat labels;
    Mat trainData;
    {
        ScopeTimerMin t("Prepared training data: ");
        vector<Mat> catImages = train_ic.getImages(0);
        Mat catData = bagTrainer.getDescriptors(catImages);

        vector<Mat> dogImages = train_ic.getImages(1);
        Mat dogData = bagTrainer.getDescriptors(dogImages);

        for(int i = 0; i < catData.rows; ++i) {labels.push_back(1);}
        for(int i = 0; i < dogData.rows; ++i) {labels.push_back(-1);}

        vconcat(catData, dogData, trainData);
    }
    
    SVMTrainer svmTrainer(maxSVMIterations);

    {
        ScopeTimerMin t("Training SVM Done: ");
        svmTrainer.train(trainData, labels);
    }
    
    ImageContainer test_ic(paths, testSize, imageSize, sampleSize);

    //////////////////////

    /* Mat cat_testData = bagTrainer.getDescriptors(test_ic.getImages(0));
    Mat dog_testData = bagTrainer.getDescriptors(test_ic.getImages(1));

    cout << "Testing Cat labels" << endl;

    Mat catLabels(cat_testData.rows, 1, CV_32F, 1);
    svmTrainer.test(cat_testData, catLabels);

    cout << "Testing Dog labels" << endl;

    Mat dogLabels(dog_testData.rows, 1, CV_32F, -1);
    svmTrainer.test(dog_testData, dogLabels); */

    //////////////////////

    std::condition_variable condition;
    std::mutex mutex;

    Mat testMat;
    int testIndex;
    

    auto testData = [&testMat, &svmTrainer, &testIndex, &condition, &mutex]() mutable {
        
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

    auto getTestData = [&testMat, &bagTrainer, &test_ic, &testIndex, &condition, &mutex] (int index) mutable {
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