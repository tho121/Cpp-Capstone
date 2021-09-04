#include "ImageContainer.h"
#include <thread>
#include <future>
#include <mutex>

ImageContainer::ImageContainer(const vector<string>& loadPaths, int sampleSize, Size imSize, int offset)
:   categorySize_(sampleSize)
,   imSize_(imSize)
{
    loadImages(loadPaths, offset);
}

vector<Mat> ImageContainer::getImages()
{
    std::lock_guard<std::mutex> lck(mutex_);

    //reserve enough space, assume equal number of images per category
    vector<Mat> allImages;
    allImages.reserve(categorySize_ * images_.size());

    for(int i = 0; i < images_.size(); ++i)
    {
        allImages.insert(allImages.end(), images_[i].begin(), images_[i].end());
    }

    return std::move(allImages);
}

void ImageContainer::loadImages(const vector<string>& paths, int offset)
{
    vector<future<vector<Mat>>> futures;

    char path[128];
    for(int i = 0; i < paths.size(); ++i)
    {
        futures.emplace_back(async(launch::async, &ImageContainer::loadImagesAsync, this, paths[i], offset));
    }

    for(int i = 0; i < futures.size(); ++i)
    {
        vector<Mat> images = futures[i].get();

        std::lock_guard<std::mutex> lck(mutex_);
        images_.emplace_back(images);
    }
}

vector<Mat> ImageContainer::loadImagesAsync(const string& path, int offset)
{
    vector<Mat> imageCategory;
    imageCategory.reserve(categorySize_);

    bool doResize = imSize_.area() > 0;

    char pathBuf[128];
    int failureOffset = 0;
    for(int i = offset; i < offset + categorySize_ + failureOffset; ++i)
    {
        sprintf(pathBuf, path.c_str(), i);

        if(i % 100 == 0)
            cout << "Loading Sample " << i << " from " << pathBuf << endl;

        Mat image = imread(pathBuf, IMREAD_GRAYSCALE);

        // Check for failure
        if (image.empty()) 
        {
            cout << "Could not open or find the image for " << pathBuf << endl;
            failureOffset++;
            continue;
        }

        if(doResize)
            resize(image, image, imSize_);
            
        imageCategory.emplace_back(image);
    }

    return std::move(imageCategory);
}