#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>
#include <random>

using namespace std;
using namespace cv;
using namespace cv::ximgproc::segmentation;

const Size targetSize(250, 250);

Mat transformImage(const Mat& image)//, Size targetSize)
{
    Mat imGrey, imBlur, imCanny, imErode, imDil, imResized;

    cv::cvtColor(image, imGrey, cv::COLOR_BGR2GRAY);

    GaussianBlur(imGrey, imBlur, Size(5,5), 3);
    Canny(imBlur, imCanny, 50, 75);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));
    dilate(imCanny, imDil, kernel);
    erode(imDil, imErode, kernel);

    resize(imErode, imResized, targetSize);

    return imResized;
}

void flatten(const Mat& image, Mat& outMat, int index)
{
    int ii = 0;
    for (int i = 0; i<image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            outMat.at<float>(index, ii++) = image.at<uchar>(i,j);
        }
    }
}

vector<Rect> getRects(const Mat& im, vector<float>& centerScores)
{
    vector<Rect> outRects;

    Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();

    ss->setBaseImage(im);
    ss->switchToSelectiveSearchFast();

    // run selective search segmentation on input image
    ss->process(outRects);

    auto size = im.size();
    Point imMid(size.width/2, size.height/2);

    //filter rects 
    for(int i = outRects.size() - 1; i >= 0; --i)
    {
        Rect& r = outRects[i];

        //filter out small boxes
        if((float)r.width / size.width  < 0.5 || (float)r.height / size.height < 0.5)
        {
            outRects.erase(outRects.begin() + i);
            continue;
        }
        else    //get a score approximation while we're here
        {
            Point boxMid(r.x + r.width/2, r.y + r.height/2);
            Point score(imMid - boxMid);
            float x = (float)score.x / size.width;
            float y = (float)score.y / size.height;

            //squared norm as the score
            //heavily discriminates against boxes far from the center
            centerScores.emplace_back(1.0f - (x*x + y*y));
        }
    }

    reverse(centerScores.begin(), centerScores.end());

    return outRects;
}

Mat getTransformedRects(const Mat& im, const vector<Rect>& rects)
{
    vector<Mat> croppedImages;
    vector<float> scores;

    auto size = im.size();
    Point imSize(size.width, size.height);
    Mat flatten_mat(rects.size(), targetSize.area(), CV_32FC1);

    for(int i = 0; i < rects.size(); ++i)
    {
        const Rect& r = rects[i];

        Mat croppedImage = im(r);
        croppedImage = transformImage(croppedImage);//, targetSize);
        flatten(croppedImage, flatten_mat, i);
    }

    return flatten_mat;
}

vector<float> getSVMScores(const Mat& sampleMat, Ptr<ml::SVM> svm)
{
    Mat score;
    svm->predict(sampleMat, score, ml::StatModel::RAW_OUTPUT);

    int rows = sampleMat.size().height;
    vector<float> scores;
    scores.reserve(rows);

    for(int i = 0; i < rows; ++i)
    {
        scores.emplace_back(score.at<float>(i, 0));
    }

    return scores;
}

vector<int> getNMSSelection(const vector<Rect>& rects, const vector<float>& scores, float confidenceThresh, float nmsThresh)
{
    vector<int> nmsIndices;
    dnn::NMSBoxes(rects, scores, confidenceThresh, nmsThresh, nmsIndices);
    
    return nmsIndices;
}

vector<int> getNMSSelection(const vector<Rect>& rects, const Mat& scores, float confidenceThresh, float nmsThresh)
{
    vector<int> nmsIndices;
    dnn::NMSBoxes(rects, scores, confidenceThresh, nmsThresh, nmsIndices);
    
    return nmsIndices;
}

void initialDataExtraction(string pathFormat, int sampleSize, vector<Rect>& outRects, Mat& outMat)
{
    char path[128];
    for(int i = 0; i < sampleSize; ++i)
    {
        sprintf(path, pathFormat.c_str(), i);

        Mat image = imread(path);

        // Check for failure
        if (image.empty()) 
        {
            std::cout << "Could not open or find the image for " << path << std::endl;
            continue;;
        }

        vector<float> scores;
        vector<Rect> rects = getRects(image, scores);
        vector<int> indices = getNMSSelection(rects, scores, 0.95f, 0.4f);
        vector<Rect> candidateRects;
        candidateRects.reserve(indices.size());

        for(int i = 0; i < indices.size(); ++i)
        {
            int j = indices[i];
            candidateRects.emplace_back(rects[j]);
        }

        Mat transformedRects = getTransformedRects(image, candidateRects);
        vconcat(outMat, transformedRects, outMat);

        outRects.insert(outRects.end(), candidateRects.begin(), candidateRects.end());
    }
}

void testSVM(Ptr<ml::SVM> svm, int offset, int testCount)
{
    vector<float> totalCatScores;
    vector<float> totalDogScores;

    char path[128];
    for(int i = offset; i < offset + testCount; ++i)
    {
        sprintf(path, "../PetImages/Cat/%d.jpg", i);
        Mat imCatTest = imread(path);

        // Check for failure
        if (imCatTest.empty()) 
        {
            std::cout << "Could not open or find the image for " << path << std::endl;
            continue;;
        }

        sprintf(path, "../PetImages/Dog/%d.jpg", i);
        Mat imDogTest = imread(path);

        // Check for failure
        if (imDogTest.empty()) 
        {
            std::cout << "Could not open or find the image for " << path << std::endl;
            continue;;
        }

        imCatTest = transformImage(imCatTest);//, targetSize);
        imDogTest = transformImage(imDogTest);//, targetSize);

        Mat test_mat(2, targetSize.area(), CV_32FC1);

        flatten(imCatTest, test_mat, 0);
        flatten(imDogTest, test_mat, 1);
        
        Mat scores;
        svm->predict(test_mat, scores, ml::StatModel::RAW_OUTPUT);

        cout << scores.at<float>(0,0) << " " << scores.at<float>(1,0) << endl;

        totalCatScores.emplace_back(scores.at<float>(0,0));
        totalDogScores.emplace_back(scores.at<float>(1,0));
    }

    float catAverage = accumulate( totalCatScores.begin(), totalCatScores.end(), 0.0)/totalCatScores.size();
    float dogAverage = accumulate( totalDogScores.begin(), totalDogScores.end(), 0.0)/totalDogScores.size();         
    cout << "The catAverage is " << catAverage << " The dogAverage is " << dogAverage << endl;  
}

int main()
{
    std::cout << "Hello World!" << "\n";

    const int sampleSize = 5;
    const float confidenceStep = 0.05f;
    const int maxIterations = 5;
    const int testSize = 5;

    vector<Rect> catRects;
    Mat catTransformedRects(0, targetSize.area(), CV_32FC1);

    initialDataExtraction("../PetImages/Cat/%d.jpg", sampleSize, catRects, catTransformedRects);

    vector<Rect> dogRects;
    Mat dogTransformedRects(0, targetSize.area(), CV_32FC1);

    initialDataExtraction("../PetImages/Dog/%d.jpg", sampleSize, dogRects, dogTransformedRects);

    //temp mid scores calculated, images flattened, nms selection done

    vector<int> catIndices(catRects.size());
    vector<int> dogIndices(dogRects.size());

    //assign initial value to be same as index, with dog indices offset by size of cat array
    for(int i = 0; i < catRects.size(); ++i){ catIndices[i] = i;}
    for(int i = 0; i < dogRects.size(); ++i){ dogIndices[i] = i + catRects.size();}

    for(int count = 0; count < maxIterations; ++count)
    {
        int totalCount = catIndices.size() + dogIndices.size();

        vector<int> order;
        order.reserve(totalCount);
        order.insert(order.end(), catIndices.begin(), catIndices.end());
        order.insert(order.end(), dogIndices.begin(), dogIndices.end());

        unsigned int seed = 0;
        std::shuffle (order.begin(), order.end(), std::default_random_engine(seed));

        vector<int> labels;
        labels.reserve(totalCount);

        Mat training_mat(totalCount, targetSize.area(), CV_32FC1);

        for(int i = 0; i < totalCount; ++i)
        {
            int j = order[i];

            if(j >= catRects.size())
            {
                //remove offset for dog
                j -= catRects.size();
                dogTransformedRects.row(j).copyTo(training_mat.row(i));
                labels.emplace_back(-1);
            }
            else
            {
                catTransformedRects.row(j).copyTo(training_mat.row(i));
                labels.emplace_back(1);
            }
        }

        Ptr<ml::SVM> svm = ml::SVM::create();
        svm->setType(ml::SVM::C_SVC);
        svm->setKernel(ml::SVM::LINEAR);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
        svm->train(training_mat, ml::ROW_SAMPLE, labels);

        testSVM(svm, sampleSize, testSize);

        //early out
        if(count == maxIterations - 1)
        {
            break;
        }

        Mat catScore;
        svm->predict(catTransformedRects, catScore, ml::StatModel::RAW_OUTPUT);

        Mat dogScore;
        svm->predict(dogTransformedRects, dogScore, ml::StatModel::RAW_OUTPUT);

        svm->clear();

        catIndices.clear();
        catIndices = getNMSSelection(catRects, catScore, 0.5f + (count * confidenceStep), 0.4f);

        dogIndices.clear();
        dogIndices = getNMSSelection(dogRects, dogScore, 0.5f + (count * confidenceStep), 0.4f);
    }

    waitKey(0);

    return 0;
}