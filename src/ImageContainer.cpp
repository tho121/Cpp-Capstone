#include "ImageContainer.h"

ImageContainer::ImageContainer(const vector<string>& loadPaths, int sampleSize, int offset)
{
    loadImages(loadPaths, sampleSize, offset);
}

vector<Mat> ImageContainer::getImages()
{
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
    images_.clear();

    categorySize_ = sampleSize;

    char path[128];
    for(int p = 0; p < paths.size(); ++p)
    {
        vector<Mat> imageCategory;
        imageCategory.reserve(sampleSize);

        int failureOffset = 0;
        for(int i = offset; i < offset + sampleSize + failureOffset; ++i)
        {
            if(i % 100 == 0)
                cout << "Loading Sample " << i << " of category " << p << std::endl;

            sprintf(path, paths[p].c_str(), i);

            Mat image = imread(path, IMREAD_GRAYSCALE);

            // Check for failure
            if (image.empty()) 
            {
                std::cout << "Could not open or find the image for " << path << std::endl;
                failureOffset++;
                continue;
            }

            imageCategory.emplace_back(image);
        }

        images_.emplace_back(imageCategory);
    }
}