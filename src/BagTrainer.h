#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class BagTrainer
{
public:

    BagTrainer();

    void computeDescriptors(const vector<Mat>& images);
    void setVocab(int maxIterations);
    Mat getDescriptors(const vector<Mat>& images);

private:

    vector<Mat> featuresUnclustered_;
    int dictionarySize_;
    unique_ptr<BOWImgDescriptorExtractor> bowDE_;
    Ptr<FeatureDetector> detector_;

};