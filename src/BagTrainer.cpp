#include "BagTrainer.h"

#include <thread>
#include <future>
#include <mutex>

BagTrainer::BagTrainer(int dictionarySize, int threads)
:   detector_(SiftFeatureDetector::create())
,   dictionarySize_(dictionarySize)
,   threadCount_(threads)
{
    Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
    Ptr<DescriptorExtractor> extractor = SiftDescriptorExtractor::create();
    bowDE_ = make_unique<BOWImgDescriptorExtractor>(extractor, matcher);

    //ensure at least 1 thread
    if(threadCount_ < 1)
    {
        threadCount_ = 1;
    }
}

void BagTrainer::computeDescriptors(const vector<Mat>& images)
{
    int splitSize = images.size() / threadCount_;
    int offset = 0;

    vector<future<Mat>> futures;
    for(int i = 0; i < threadCount_; ++i)
    {
        futures.emplace_back(async(launch::async, &BagTrainer::computeDescriptorsAsync, this, images, offset, offset + splitSize));
        offset += splitSize;
    }

    Mat featuresUnclustered;

    //wait for data to return, in order
    for(int i = 0; i < futures.size(); ++i)
    {
        Mat features = futures[i].get();
        featuresUnclustered.push_back(features);
    }
    
    featuresUnclustered_.emplace_back(featuresUnclustered);
}

Mat BagTrainer::computeDescriptorsAsync(const vector<Mat>& images, int startPos, int endPos)
{
    vector<KeyPoint> keypoints;
    Mat descriptor;
    Ptr<SiftDescriptorExtractor> detector = SiftDescriptorExtractor::create();
    Mat featuresUnclustered;

    for(int i = startPos; i < endPos && i < images.size(); ++i)
    {
        if(i % 100 == 0)
            cout << "Computing Sample " << i << endl;

        detector->detect(images[i], keypoints);
        detector->compute(images[i], keypoints, descriptor);

        featuresUnclustered.push_back(descriptor);
    }

    return featuresUnclustered;
}

void BagTrainer::setVocab(int maxIterations)
{
    int flags=KMEANS_PP_CENTERS;
    BOWKMeansTrainer bowTrainer(dictionarySize_, TermCriteria(TermCriteria::MAX_ITER, maxIterations ,0.001), 1, flags);

    Mat allFeatures;

    for(int i = 0; i < featuresUnclustered_.size(); ++i)
    {
        allFeatures.push_back(featuresUnclustered_[i]);
    }

    Mat dict = bowTrainer.cluster(allFeatures);

    bowDE_->setVocabulary(dict);
}

Mat BagTrainer::getDescriptors(const vector<Mat>& images)
{
    int splitSize = images.size() / threadCount_;
    int count = 0;

    vector<future<Mat>> futures;
    for(int i = 0; i < threadCount_; ++i)
    {
        futures.emplace_back(async(launch::async, &BagTrainer::getDescriptorsAsync, this, images, count, count + splitSize));
        count += splitSize;
    }

    Mat outMat;
    //wait for data to return, in order
    for(int i = 0; i < futures.size(); ++i)
    {
        Mat features = futures[i].get();

        outMat.push_back(features);
    }
    
    return std::move(outMat);
}

Mat BagTrainer::getDescriptorsAsync(const vector<Mat>& images, int startPos, int endPos)
{
    Mat outMat;
    for(int i = startPos; i < endPos && i < images.size(); ++i)
    {
        if(i % 100 == 0)
            cout << "Computing descriptor " << i << endl;

        vector<KeyPoint> keypoints;
        detector_->detect(images[i], keypoints);

        Mat descriptors;
        bowDE_->compute(images[i], keypoints, descriptors);

        if (!descriptors.empty())
        {
            outMat.push_back(descriptors);
        }
    }

    return std::move(outMat);
}