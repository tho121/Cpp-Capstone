#include "BagTrainer.h"

BagTrainer::BagTrainer()
:   detector_(SiftFeatureDetector::create())
{
    Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
    Ptr<DescriptorExtractor> extractor = SiftDescriptorExtractor::create();
    bowDE_ = make_unique<BOWImgDescriptorExtractor>(extractor, matcher);

    dictionarySize_ = 2000;
}

void BagTrainer::calcFeatures(const vector<Mat>& images)
{
    vector<KeyPoint> keypoints;
    Mat descriptor;
    Ptr<SiftDescriptorExtractor> detector = SiftDescriptorExtractor::create();
    Mat featuresUnclustered;

    for(int i = 0; i < images.size(); ++i)
    {
        detector->detect(images[i], keypoints);
        detector->compute(images[i], keypoints, descriptor);

        featuresUnclustered.push_back(descriptor);
    }

    featuresUnclustered_.push_back(featuresUnclustered);
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
    Mat outMat;
    for(int i = 0; i < images.size(); ++i)
    {
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