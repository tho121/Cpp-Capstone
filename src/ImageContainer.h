#include <vector>
#include <string>
#include <mutex>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class ImageContainer
{
public:

    ImageContainer(const vector<string>& loadPaths, int sampleSize, int offset = 0);

    vector<Mat> getImages();
    vector<Mat> getImages(int categoryIndex) {return images_[categoryIndex];}
    int getCategorySize() {return categorySize_;}

private:

    void loadImages(const vector<string>& paths, int sampleSize, int offset = 0);
    vector<Mat> loadImagesAsync(const string& path, int sampleSize, int offset = 0);

    vector<vector<Mat>> images_;
    int categorySize_;

    std::mutex mutex_;

};