#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class BagTrainer
{
public:

    BagTrainer(int dictionarySize, int threads = 1);

    void computeDescriptors(const vector<Mat>& images);
    void setVocab(int maxIterations);
    Mat getDescriptors(const vector<Mat>& images);
    
private:

    Mat computeDescriptorsAsync(const vector<Mat>& images, int startPos, int endPos);
    Mat getDescriptorsAsync(const vector<Mat>& images, int startPos, int endPos);

    vector<Mat> featuresUnclustered_;
    int dictionarySize_;
    int threadCount_;
    unique_ptr<BOWImgDescriptorExtractor> bowDE_;
    Ptr<FeatureDetector> detector_;

    

};