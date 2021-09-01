#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/ml.hpp>
#include <random>

using namespace std;
using namespace cv;
using namespace cv::ximgproc::segmentation;

vector<Mat> processedImages;
vector<int> labels;
const Size targetSize(250, 250);

vector<Mat> showSelectiveSearch(const Mat& im)
{
    Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();

    ss->setBaseImage(im);
    ss->switchToSelectiveSearchFast();

    // run selective search segmentation on input image
    std::vector<Rect> rects;
    ss->process(rects);
/*     std::cout << "Total Number of Region Proposals: " << rects.size() << std::endl;
 */
    auto size = im.size();

    Mat imOut = im.clone();

    vector<float> scores;
    Point imMid(size.width/2, size.height/2);
    Point imSize(size.width, size.height);

    //filter rects 
    for(int i = rects.size() - 1; i >= 0; --i)
    {
        Rect& r = rects[i];

        if((float)r.width / size.width  < 0.5 || (float)r.height / size.height < 0.5)
        {
            rects.erase(rects.begin() + i);
            continue;
        }
        else
        {
            //rectangle(imOut, r, Scalar(0, 255, 0));

            Point boxMid(r.x + r.width/2, r.y + r.height/2);

            //circle(imOut, boxMid, 2, Scalar(0, 0, 255));

            Point score(imMid - boxMid);
            float x = (float)score.x / size.width;
            float y = (float)score.y / size.height;

            //squared norm as the score
            //heavily discriminates against boxes far from the center
            scores.emplace_back(1.0f - (x*x + y*y));
        }
    }

    reverse(scores.begin(), scores.end());

    vector<int> nmsIndices;
    dnn::NMSBoxes(rects, scores, 0.95f, 0.4f, nmsIndices);
    vector<Mat> croppedImages;

    for(int i = 0; i < nmsIndices.size(); ++i)
    {
        rectangle(imOut, rects[nmsIndices[i]], Scalar(0, 255, 0));
        croppedImages.emplace_back(imOut(rects[nmsIndices[i]]));
    }

    // show output
    //imshow("Output", imOut);

    return croppedImages;
}

/* void showBlob(const Mat& image, string name)
{
	std::vector<KeyPoint> keypoints;
    Ptr<SimpleBlobDetector> detector = cv::SimpleBlobDetector::create();
	detector->detect( image, keypoints);

    Mat im_with_keypoints;
    drawKeypoints( image, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

    imshow(name, im_with_keypoints );
} */

Mat transformImage(const Mat& image, Size targetSize)
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

void getProcessedImages(std::string path, vector<Mat>& outVector)
{
    Mat image = imread(path);

     // Check for failure
    if (image.empty()) 
    {
        cout << "Could not open or find the image for " << path << endl;
        return;
    }


    /* Mat adap_mean_2, adap_mean_2_blurred;

    adaptiveThreshold(imGrey, adap_mean_2, 255, 
                                        ADAPTIVE_THRESH_MEAN_C, 
                                        THRESH_BINARY, 7, 2);

    adaptiveThreshold(imGrey, adap_mean_2_blurred, 255, 
                                        ADAPTIVE_THRESH_MEAN_C, 
                                        THRESH_BINARY, 7, 2);

    imshow("adap_mean_2", adap_mean_2);
    imshow("adap_mean_2_blurred", adap_mean_2_blurred); */

    /* Mat thresh;
    threshold(imErode, thresh, 127, 255, THRESH_BINARY_INV | THRESH_OTSU);
    imshow("thresh", thresh);

    RNG rng(12345);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( imErode, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
    Mat drawing = Mat::zeros( imErode.size(), CV_8UC3 );

    sort(contours.begin(), contours.end(), [](vector<Point> contour1, vector<Point> contour2 )
    {
        double i = fabs( contourArea(cv::Mat(contour1)) );
        double j = fabs( contourArea(cv::Mat(contour2)) );
        return ( i > j );
    });

    for( size_t i = 0; i < contours.size() && i < 10; i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
    }
    imshow( "Contours", drawing ); */

    vector<Mat> selectedImages = showSelectiveSearch(image);
    

    for(int i = 0; i < selectedImages.size(); ++i)
    {
        selectedImages[i] = transformImage(selectedImages[i], targetSize);

        /* char buf[16];
        sprintf(buf, "IM: %d", i);
        imshow(string(buf), selectedImages[i]); */
    }

    outVector.insert(outVector.begin(), selectedImages.begin(), selectedImages.end());

    //waitKey(0); // Wait for any keystroke in the window
}

int main() {
    std::cout << "Hello World!" << "\n";

    const int sampleSize = 1000;

    char path[128];
    for(int i = 0; i < sampleSize; ++i)
    {
        sprintf(path, "../PetImages/Cat/%d.jpg", i);
        getProcessedImages(string(path), processedImages);
    }

    labels.insert(labels.end(), processedImages.size(), 1);

    int failCount = 0;
    for(int i = 0; i < sampleSize + failCount; ++i)
    {
        sprintf(path, "../PetImages/Dog/%d.jpg", i);
        //getProcessedImages(string(buf), processedImages);
        Mat imDog = imread(path);

        if (imDog.empty()) 
        {
            cout << "Could not open or find the image for " << path << endl;
            failCount++;
            continue;
        }

        processedImages.emplace_back( transformImage(imDog, targetSize) );
    }

    labels.insert(labels.end(), sampleSize, 0);

    //same seed will shuffle the images with their labels together
    unsigned int seed = 0;

    shuffle (processedImages.begin(), processedImages.end(), std::default_random_engine(seed));
    shuffle (labels.begin(), labels.end(), std::default_random_engine(seed));

    Mat training_mat(processedImages.size(), targetSize.area(), CV_32FC1);

    for(int count = 0; count < processedImages.size(); ++count)
    {
        int ii = 0; // Current column in training_mat
        for (int i = 0; i<processedImages[count].rows; i++) {
            for (int j = 0; j < processedImages[count].cols; j++) {
                training_mat.at<float>(count,ii++) = processedImages[count].at<uchar>(i,j);
            }
        }
    }
    
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(training_mat, ml::ROW_SAMPLE, labels);

    vector<float> totalCatScores;
    vector<float> totalDogScores;

    for(int i = sampleSize; i < sampleSize + 10; ++i)
    {
        sprintf(path, "../PetImages/Cat/%d.jpg", i);
        Mat imCatTest = imread(path);

        sprintf(path, "../PetImages/Dog/%d.jpg", i);
        Mat imDogTest = imread(path);

        transformImage(imCatTest, targetSize);
        transformImage(imDogTest, targetSize);

        Mat test_mat(2, targetSize.area(), CV_32FC1);

        int ii = 0; // Current column in training_mat
        for (int i = 0; i<targetSize.width; i++) {
            for (int j = 0; j < targetSize.height; j++) {
                test_mat.at<float>(0,ii) = imCatTest.at<uchar>(i,j);
                test_mat.at<float>(1,ii) = imDogTest.at<uchar>(i,j);
                ii++;
            }
        }
        

        Mat scores;
        svm->predict(test_mat, scores, ml::StatModel::RAW_OUTPUT);

        cout << scores.at<float>(0,0) << " " << scores.at<float>(1,0) << endl;

        totalCatScores.emplace_back(scores.at<float>(0,0));
        totalDogScores.emplace_back(scores.at<float>(1,0));
    }

    float catAverage = accumulate( totalCatScores.begin(), totalCatScores.end(), 0.0)/totalCatScores.size();              
    cout << "The catAverage is" << catAverage << endl;

    float dogAverage = accumulate( totalDogScores.begin(), totalDogScores.end(), 0.0)/totalDogScores.size();              
    cout << "The dogAverage is" << dogAverage << endl;   

    /* showImages("../PetImages/Cat/0.jpg");
    showImages("../PetImages/Cat/1.jpg");
    showImages("../PetImages/Cat/2.jpg");
    showImages("../PetImages/Cat/3.jpg");
    showImages("../PetImages/Cat/4.jpg");
    showImages("../PetImages/Cat/5.jpg");
    showImages("../PetImages/Cat/6.jpg");
    showImages("../PetImages/Cat/7.jpg");
    showImages("../PetImages/Cat/8.jpg"); */

    waitKey(0);

    return 0;
}