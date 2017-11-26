#ifndef OPENCV_UTIL_H
#define OPENCV_UTIL_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <gtkmm.h>
#include <vector>

using namespace cv;
using namespace std;
//void basic_draw(Mat & img, std::vector<cv::Point> & p, int shape, int type, int thickness, Gdk::RGBA color);

int blendImages(cv::Mat & s, cv::Mat & t, cv::Mat & dest, double alpha, double gamma);
int rotateImage(cv::Mat & s, cv::Mat & d, double angle, double scale);
int fourierTrans(cv::Mat & s, cv::Mat & d);
int logTrans(cv::Mat & s, cv::Mat & d, double bias, double bend);
int remapTrans(cv::Mat & s, int method);


int adjustImage(cv::Mat & frame);

std::vector<cv::Mat> detectFaces(cv::Mat & frame);
int detectFaces(cv::Mat & frame, std::vector<pair<cv::Mat, cv::Rect> > & faces_all);
int detectEyes(cv::Mat & frame);
int detectNose(cv::Mat & frame);
int detectMouth(cv::Mat & frame);
//int detectLandmark(cv::Mat & frame);

int detectPedestrian(cv::Mat & frame);

void getMaxClass(dnn::Blob &probBlob, int *classId, double *classProb);
std::vector<String> readClassNames(const char *filename = "synset_words.txt");
std::string int_string(int a);
float mean(const std::vector<float>& v);
float cov(const std::vector<float>& v1, const std::vector<float>& v2);
float coefficient(const std::vector<float>& v1, const std::vector<float>& v2);

float cos_distance(const std::vector<float>& vecfeature1, vector<float>& vecfeature2);
void read_csv(const string & filename, vector<Mat> & images, vector<int>& labels, char sep=';');

std::vector<std::string> read_features(std::string & path);

#endif
