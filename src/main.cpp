#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/features2d.hpp>
#include <random>

using namespace std;
using namespace cv;

vector<Mat> getImages(const vector<string>& paths, const int sampleSize)
{
    vector<Mat> images;
    char path[128];
    for(int p = 0; p < paths.size(); ++p)
    {
        int failureOffset = 0;
        for(int i = 0; i < sampleSize + failureOffset; ++i)
        {
            if(i % 100 == 0)
                cout << "Sample " << i << std::endl;

            sprintf(path, paths[p].c_str(), i);

            Mat image = imread(path, IMREAD_GRAYSCALE);

            // Check for failure
            if (image.empty()) 
            {
                std::cout << "Could not open or find the image for " << path << std::endl;
                failureOffset++;
                continue;
            }

            images.emplace_back(image);
        }
    }
    
    return images;
}

Mat initialDataExtraction(vector<Mat> images, Mat& labels, int groupSize, BOWImgDescriptorExtractor& bowDE, Ptr<FeatureDetector> train_detector)
{
    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;
    //To store the SIFT descriptor of current image
    Mat descriptor;
    //To store all the descriptors that are extracted from all the images.
    Mat featuresUnclustered;
    //The SIFT feature extractor and descriptor
    Ptr<SiftDescriptorExtractor> detector = SiftDescriptorExtractor::create();

    for(int i = 0; i < images.size(); ++i)
    {
        //detect feature points
        detector->detect(images[i], keypoints);
        //compute the descriptors for each keypoint
        detector->compute(images[i], keypoints, descriptor);

        featuresUnclustered.push_back(descriptor);
    }

    int dictionarySize=2000;
    //define Term Criteria
    TermCriteria tc(TermCriteria::MAX_ITER,1000,0.001);
    //retries number
    int retries=1;
    //necessary flags
    int flags=KMEANS_PP_CENTERS;
    //Create the BoW (or BoF) trainer
    BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
    //cluster the feature vectors
    Mat dictionary=bowTrainer.cluster(featuresUnclustered);

    bowDE.setVocabulary(dictionary);

    //Mat train, response;

    Mat outMat;
    for(int i = 0; i < images.size(); ++i)
    {
        vector<KeyPoint> keypoints;
        train_detector->detect(images[i], keypoints);
        Mat descriptors;
        bowDE.compute(images[i], keypoints, descriptors);

        if (!descriptors.empty())
        {
            outMat.push_back(descriptors);     // update training data
            labels.push_back( i < groupSize ? 1 : -1);        // update response data
        }
    }

    return outMat;
}

void testSVM(Ptr<ml::SVM> svm, int offset, int testCount, BOWImgDescriptorExtractor& bowDE, Ptr<FeatureDetector> detector )
{
    vector<float> totalCatScores;
    vector<float> totalDogScores;

    char path[128];
    for(int i = offset; i < offset + testCount; ++i)
    {
        Mat test;

        sprintf(path, "../PetImages/Cat/%d.jpg", i);
        Mat imCatTest = imread(path, IMREAD_GRAYSCALE);

        // Check for failure
        if (imCatTest.empty()) 
        {
            cout << "Could not open or find the image for " << path << std::endl;
            continue;;
        }

        {
            vector<KeyPoint> keypoints;
            detector->detect(imCatTest, keypoints);
            Mat descriptors;
            bowDE.compute(imCatTest, keypoints, descriptors);
            if (!descriptors.empty())
            {
                test.push_back(descriptors);     // update training data
            }
            else
            {
                continue;
            }
        }
        

        sprintf(path, "../PetImages/Dog/%d.jpg", i);
        Mat imDogTest = imread(path, IMREAD_GRAYSCALE);

        // Check for failure
        if (imDogTest.empty()) 
        {
            cout << "Could not open or find the image for " << path << std::endl;
            continue;;
        }

        {
            vector<KeyPoint> keypoints;
            detector->detect(imDogTest, keypoints);
            Mat descriptors;
            bowDE.compute(imDogTest, keypoints, descriptors);
            if (!descriptors.empty())
            {
                test.push_back(descriptors);     // update training data
            }
            else
            {
                continue;
            }
        }
        
        Mat scores;
        svm->predict(test, scores, ml::StatModel::RAW_OUTPUT);

        cout << scores.at<float>(0,0) << " " << scores.at<float>(1,0) << endl;

        totalCatScores.emplace_back(scores.at<float>(0,0));
        totalDogScores.emplace_back(scores.at<float>(1,0));
    }

    float catAverage = accumulate( totalCatScores.begin(), totalCatScores.end(), 0.0)/totalCatScores.size();
    float dogAverage = accumulate( totalDogScores.begin(), totalDogScores.end(), 0.0)/totalDogScores.size();

    cout << "The catAverage is " << catAverage << " The dogAverage is " << dogAverage << endl;  

    int catCount = 0;
    for(int i = 0; i < totalCatScores.size(); ++i)
    {
        if(totalCatScores[i] > 0.0f)
        {
            catCount++;
        }
    }

    int dogCount = 0;
    for(int i = 0; i < totalDogScores.size(); ++i)
    {
        if(totalDogScores[i] <= 0.0f)
        {
            dogCount++;
        }
    }

    cout << "The cat score is " << catCount << " out of " << totalCatScores.size() << " " << ((float)catCount)/totalCatScores.size() << endl; 
    cout << "The dog score is " << dogCount << " out of " << totalDogScores.size() << " " << ((float)dogCount)/totalDogScores.size() << endl; 
}

int main()
{
    std::cout << "Hello World!" << "\n";

    const int sampleSize = 2000;
    const int maxIterations = 500;
    const int testSize = 100;

    Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
    Ptr<DescriptorExtractor> extractor = SiftDescriptorExtractor::create();
    BOWImgDescriptorExtractor bowDE(extractor, matcher);
    Ptr<FeatureDetector> detector = SiftFeatureDetector::create();

    vector<string> paths = {"../PetImages/Cat/%d.jpg", "../PetImages/Dog/%d.jpg"};
    vector<Mat> images = getImages(paths, sampleSize);

    Mat labels;
    Mat train_data = initialDataExtraction(images, labels, sampleSize, bowDE, detector);

    int totalSize = train_data.rows;
 
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, maxIterations, 1e-6));
    svm->trainAuto(train_data, ml::ROW_SAMPLE, labels);

    testSVM(svm, sampleSize + testSize, testSize, bowDE, detector);

    waitKey(0);

    return 0;
}