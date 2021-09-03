#include "ImageContainer.h"
#include <thread>
#include <future>
#include <mutex>

ImageContainer::ImageContainer(const vector<string>& loadPaths, int sampleSize, int offset)
{
    loadImages(loadPaths, sampleSize, offset);
}

vector<Mat> ImageContainer::getImages()
{
    std::lock_guard<std::mutex> lck(mutex_);
    
    vector<Mat> allImages;
    allImages.reserve(categorySize_ * images_.size());

    for(int i = 0; i < images_.size(); ++i)
    {
        allImages.insert(allImages.end(), images_[i].begin(), images_[i].end());
    }

    return std::move(allImages);
}

void ImageContainer::loadImages(const vector<string>& paths, int sampleSize, int offset)
{
    categorySize_ = sampleSize;

    vector<future<vector<Mat>>> futures;

    char path[128];
    for(int i = 0; i < paths.size(); ++i)
    {
        futures.push_back(async(launch::async, &ImageContainer::loadImagesAsync, this, paths[i], sampleSize, offset));
    }

    for(int i = 0; i < futures.size(); ++i)
    {
        vector<Mat> images = futures[i].get();

        std::lock_guard<std::mutex> lck(mutex_);
        images_.emplace_back(images);
    }
}

vector<Mat> ImageContainer::loadImagesAsync(const string& path, int sampleSize, int offset)
{
    vector<Mat> imageCategory;
    imageCategory.reserve(sampleSize);

    char pathBuf[128];
    int failureOffset = 0;
    for(int i = offset; i < offset + sampleSize + failureOffset; ++i)
    {
        if(i % 100 == 0)
            cout << "Loading Sample " << i << " from " << path << endl;

        sprintf(pathBuf, path.c_str(), i);

        Mat image = imread(pathBuf, IMREAD_GRAYSCALE);

        // Check for failure
        if (image.empty()) 
        {
            cout << "Could not open or find the image for " << path << endl;
            failureOffset++;
            continue;
        }

        imageCategory.emplace_back(image);
    }

    return std::move(imageCategory);
}