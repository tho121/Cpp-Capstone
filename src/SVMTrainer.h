#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;


class SVMTrainer
{
public:
    SVMTrainer(int maxIterations);

    void train(Mat& train_data, Mat& labels);
    void test(const Mat& test_data, const Mat& labels);

private:

    Ptr<ml::SVM> svm_;

};