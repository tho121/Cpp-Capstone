#include "SVMTrainer.h"
#include <numeric>

SVMTrainer::SVMTrainer(int maxIterations)
{
    svm_ = ml::SVM::create();
    svm_->setType(ml::SVM::C_SVC);
    svm_->setKernel(ml::SVM::LINEAR);
    svm_->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, maxIterations, 1e-6));
}

void SVMTrainer::train(Mat& train_data, Mat& labels)
{
    svm_->trainAuto(train_data, ml::ROW_SAMPLE, labels);
}

void SVMTrainer::test(const Mat& test_data, const Mat& labels)
{
    Mat scores;
    svm_->predict(test_data, scores, ml::StatModel::RAW_OUTPUT);

    vector<float> totalScores(test_data.rows);

    for(int i = 0; i < test_data.rows; ++i)
    {
        totalScores[i] = scores.at<float>(i,0);
    }

    float avg = std::accumulate( totalScores.begin(), totalScores.end(), 0.0f)/totalScores.size();

    cout << "The average is " << avg << endl;  

    int count = 0;
    for(int i = 0; i < totalScores.size(); ++i)
    {
        if(scores.at<float>(i,0) > 0 && labels.at<float>(i,0) > 0)
        {
            count++;
        }
        else if(scores.at<float>(i,0) < 0 && labels.at<float>(i,0) < 0)
        {
            count++;
        }
    }

    cout << "The score is " << count << " out of " << test_data.rows << " " << ((float)count)/test_data.rows << endl; 
}