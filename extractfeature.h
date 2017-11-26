#ifndef EXTRACT_FEATURE_H
#define EXTRACT_FEATURE_H
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

std::vector<float> extractfeature(Mat faceROI);
void caffe_predefine();
#endif
