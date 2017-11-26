#include <iostream>
#include <fstream>
//#include <Flandmark.h>
//#include <CFeaturePool.h>
//#include <CSparseLBPFeatures.h>
//#include <helpers.h>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <sys/types.h>
#include <dirent.h>

#include "util.h"

using namespace cv;
using namespace std;
using namespace cv::face;
using namespace cv::dnn;
//void basic_draw(Mat & img, std::vector<cv::Point> & p, int shape, int type, int thickness, Gdk::RGBA color)
//{

//}

int blendImages(cv::Mat & s, cv::Mat & t, cv::Mat & dest, double alpha, double gamma)
{
	Mat temp;
	if(!s.data)
		return -1;
	if(!t.data)
		return -1;

  if(s.rows==t.rows && s.cols==t.cols){
    addWeighted(s, alpha/100.0, t, 1-alpha/100.0, gamma, dest);
  }
  else if(s.rows>=t.rows && s.cols >=t.cols){
    Mat ROI;

    ROI=s(Rect((s.cols-t.cols)/2,(s.rows-t.rows)/2, t.cols, t.rows));
    addWeighted(ROI, alpha/100.0, t, 1-alpha/100.0, gamma, temp);
    dest=temp.clone();
  }
  else if(s.rows<=t.rows && s.cols<=t.cols){
    Mat ROI;
    ROI=t(Rect((t.cols-s.cols)/2, (t.rows-s.rows)/2, s.cols, s.rows));
    addWeighted(s, alpha/100.0, ROI, 1-alpha/100.0, gamma, temp);
    dest=temp.clone();
  }
  else{
    std::cout<<"couldn't blend"<<std::endl;
    return false;
	}
	
}

int rotateImage(cv::Mat & s, cv::Mat & d, double angle, double scale)
{
  Point2f center=Point2f(s.cols/2, s.rows/2);

  double angle1=angle*CV_PI/180;
  double a=sin(angle1), b=cos(angle1);
  int rotate_width=int(s.rows*fabs(a)+s.cols*fabs(b));
  int rotate_height=int(s.cols*fabs(a)+s.rows*fabs(b));

  Mat rotated;
  rotated=getRotationMatrix2D(center, angle, scale);

  warpAffine(s, d, rotated, Size(rotate_width, rotate_height));

	return 0;
}

int fourierTrans(cv::Mat & s, cv::Mat & d)
{
	Mat padded;
	int m=getOptimalDFTSize(s.rows);
	int n=getOptimalDFTSize(s.cols);
	copyMakeBorder(s, padded, 0, m-s.rows, 0, n-s.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[]={Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};

	Mat complexI;
	merge(planes, 2, complexI);
	
	dft(complexI, complexI);
	split(complexI, planes);
	
	magnitude(planes[0],planes[1], planes[0]);
	Mat magI=planes[0];

	magI+=Scalar::all(1);

	log(magI, magI);

	magI=magI(Rect(0,0,magI.cols & -2, magI.rows & -2));

	int cx=magI.cols/2;
	int cy=magI.rows/2;

	Mat q0(magI, Rect(0,0,cx, cy));
	Mat q1(magI, Rect(cx, 0, cx, cy));
	Mat q2(magI, Rect(0,cy, cx, cy));
	Mat q3(magI, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, NORM_MINMAX);

	d=magI.clone();
	return 0;
}

int logTrans(cv::Mat & s, cv::Mat & d, double bias, double bend)
{
	double gray=0;

	const int channels=s.channels();

	switch(channels){
	case 1:
		for(int i=0; i<s.rows; i++)
			for(int j=0; j<s.cols; j++){
				gray=(double)s.at<uchar>(i,j);
				gray=bend*log((double)(1+gray))+bias;
				s.at<uchar>(i,j)=saturate_cast<uchar>(gray);
			}
		
			break;
	case 3:
		Mat planes[3];
		split(s, planes);
		for(int i=0;i<s.rows;i++){
			for(int j=0;j<s.cols;j++){
				planes[0].at<uchar>(i,j)=saturate_cast<uchar>(bend*log((double)(1+(double)planes[0].at<uchar>(i,j)))+bias);
				planes[1].at<uchar>(i,j)=saturate_cast<uchar>(bend*log((double)(1+(double)planes[1].at<uchar>(i,j)))+bias);
				planes[2].at<uchar>(i,j)=saturate_cast<uchar>(bend*log((double)(1+(double)planes[2].at<uchar>(i,j)))+bias);
			}
		}
		merge(planes, 3 , s);
		break;
	}
	
	d=s.clone();

}

int remapTrans(cv::Mat & s, int method)
{
	Mat temp;
	Mat map_x, map_y;

	temp.create(s.size(), s.type());
	map_x.create(s.size(), CV_32FC1);
	map_y.create(s.size(), CV_32FC1);

	for(int j=0;j<s.rows;j++){
		for(int i=0;i<s.cols;i++){	
			switch(method){
			case 0:
				if(i>s.cols*0.25 && i<s.cols*0.75 && j>s.rows*0.25 && j<s.rows*0.75){
					map_x.at<float>(j,i)=2*(i-s.cols*0.25)+0.5;
					map_y.at<float>(j,i)=2*(j-s.rows*0.25)+0.5;
				}
				else{
					map_x.at<float>(j,i)=0;
					map_y.at<float>(j,i)=0;
				}
				break;
			case 1:
				map_x.at<float>(j,i)=i;
				map_y.at<float>(j,i)=s.rows-j;
				break;
			case 2:
				map_x.at<float>(j,i)=s.cols-i;
				map_y.at<float>(j,i)=j;
				break;
			case 3:
				map_x.at<float>(j,i)=s.cols-i;
				map_y.at<float>(j,i)=s.rows-j;
				break;
			case 4:
				map_x.at<float>(j,i)=j;
				map_y.at<float>(j,i)=i;
				break;
			}
		}				
	}

	remap(s, s, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
	return 0;
}

int adjustImage(cv::Mat & frame)
{
	String face_cascade_name = "xml/haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "xml/haarcascade_eye_tree_eyeglasses.xml";
	CascadeClassifier face_cascade;
	CascadeClassifier eyes_cascade;

	double alpha;

	if(!face_cascade.load(face_cascade_name)){
		std::cout<<"error load face config file\n";
		return -1;
	}
	if(!eyes_cascade.load(eyes_cascade_name)){
		std::cout<<"error load eyes config file\n";
		return -1;
	}

	Mat frm_gray;

	cvtColor(frame, frm_gray, CV_BGR2GRAY);
	equalizeHist(frm_gray, frm_gray);
	
	std::vector<Rect> faces;

	face_cascade.detectMultiScale(frm_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30,30));
	double max_face_size=0;
	for(std::vector<cv::Rect>::iterator it=faces.begin(); it!=faces.end(); ++it){
		if((*it).width*(*it).height>max_face_size)
			max_face_size=(*it).width*(*it).height;	
	}
	
	for(size_t i=0; i<faces.size(); i++){
		if(faces[i].width*faces[i].height<max_face_size*0.5)
			continue;
		Mat faceROI=frm_gray(faces[i]);
		std::vector<Rect> eyes;

		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30,30));

		if(eyes.size()<2){
			std::cout<<"can't find eyes, please using other parameters and try again\n";
			return -1;
		}
		std::cout<<"eye vector size: "<<eyes.size()<<std::endl;
//		for(size_t j=0;j<eyes.size();j++){			
			Point p1(faces[i].x+eyes[0].x+eyes[0].width*0.5, faces[i].y+eyes[0].y+eyes[0].height*0.5);
			Point p2(faces[i].x+eyes[1].x+eyes[1].width*0.5, faces[i].y+eyes[1].y+eyes[1].height*0.5);
			alpha=(p2.y-p1.y)*1.0/(p2.x-p1.x);	
//		}
	}

	std::cout<<"alpha: "<<alpha<<std::endl;
	std::cout<<"angle: "<<atan(alpha)<<std::endl;	

	Mat temp;
	rotateImage(frame, temp, atan(alpha)*180/CV_PI,1);
	frame=temp.clone();
	return 0;
}

int detectEyes(cv::Mat & frame)
{
	String face_cascade_name = "xml/haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "xml/haarcascade_eye_tree_eyeglasses.xml";
	CascadeClassifier face_cascade;
	CascadeClassifier eyes_cascade;

	if(!face_cascade.load(face_cascade_name)){
		std::cout<<"error load face config file\n";
		return -1;
	}
	if(!eyes_cascade.load(eyes_cascade_name)){
		std::cout<<"error load eyes config file\n";
		return -1;
	}

	Mat frm_gray;

	cvtColor(frame, frm_gray, CV_BGR2GRAY);
	equalizeHist(frm_gray, frm_gray);
	
	std::vector<Rect> faces;

	face_cascade.detectMultiScale(frm_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30,30));
	double max_face_size=0;
	for(std::vector<cv::Rect>::iterator it=faces.begin(); it!=faces.end(); ++it){
		if((*it).width*(*it).height>max_face_size)
			max_face_size=(*it).width*(*it).height;	
	}

	for(size_t i=0; i<faces.size(); i++){
		if(faces[i].width*faces[i].height<max_face_size*0.5)
			continue;
		Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse(frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

		char buf[512];
		sprintf(buf, "width: %d, height:%d", faces[i].width, faces[i].height);
		cv::String text(buf);
		cv::putText(frame, text, cv::Point(faces[i].x, faces[i].y),cv::FONT_HERSHEY_SIMPLEX ,1, Scalar(255,0,255));

		Mat faceROI=frm_gray(faces[i]);
		std::vector<Rect> eyes;
	

		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30,30));
		
		for(size_t j=0; j<eyes.size(); j++){
			
			Point center(faces[i].x+eyes[j].x+eyes[j].width*0.5, faces[i].y+eyes[j].y+eyes[j].height*0.5);
			if(faces[i].contains(center)){
				int radius=cvRound((eyes[j].width+eyes[j].height)*0.25);		
				circle(frame, center, radius, Scalar(255,0,255),4,8,0);
			}
		}
	}
	return 0;
}
int detectFaces(cv::Mat & frame, std::vector<pair<cv::Mat, cv::Rect> > & faces_all)
{
	String face_cascade_name = "xml/haarcascade_frontalface_alt.xml";
	CascadeClassifier face_cascade;
	
	if(!face_cascade.load(face_cascade_name)){
		std::cout<<"error load config file\n";
		exit(-1);
	}

	std::vector<Rect> faces;
	Mat frm_gray;

	cvtColor(frame, frm_gray, CV_BGR2GRAY);
	equalizeHist(frm_gray, frm_gray);

	face_cascade.detectMultiScale(frm_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30,30));
	double max_face_size=0;
	for(std::vector<cv::Rect>::iterator it=faces.begin(); it!=faces.end(); ++it){
		if((*it).width*(*it).height>max_face_size)
			max_face_size=(*it).width*(*it).height;	
	}
	
	for(size_t i=0; i<faces.size(); i++){
		if(faces[i].width*faces[i].height<max_face_size*0.5)
			continue;
		Point center(faces[i].x+faces[i].width*0.5, faces[i].y+faces[i].height*0.5);
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255,0, 255), 4, 8, 0);
		rectangle(frame, faces[i], Scalar(0,255,255), 2, 8);
		std::pair<cv::Mat, cv::Rect> temp=make_pair(frame(faces[i]), faces[i]);			
		faces_all.push_back(temp);
	}
	
	return 0;
}

std::vector<cv::Mat>  detectFaces(cv::Mat & frame)
{
	std::vector<cv::Mat> f_roi;
	String face_cascade_name = "xml/haarcascade_frontalface_alt.xml";
	CascadeClassifier face_cascade;
	
	if(!face_cascade.load(face_cascade_name)){
		std::cout<<"error load config file\n";
		exit(-1);
	}

	std::vector<Rect> faces;
	Mat frm_gray;

	cvtColor(frame, frm_gray, CV_BGR2GRAY);
	equalizeHist(frm_gray, frm_gray);

	face_cascade.detectMultiScale(frm_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30,30));
	double max_face_size=0;
	for(std::vector<cv::Rect>::iterator it=faces.begin(); it!=faces.end(); ++it){
		if((*it).width*(*it).height>max_face_size)
			max_face_size=(*it).width*(*it).height;	
	}
	
	for(size_t i=0; i<faces.size(); i++){
		if(faces[i].width*faces[i].height<max_face_size*0.5)
			continue;
		Point center(faces[i].x+faces[i].width*0.5, faces[i].y+faces[i].height*0.5);
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255,0, 255), 4, 8, 0);
		rectangle(frame, faces[i], Scalar(0,255,255), 2, 8);
		
		f_roi.push_back(frame(faces[i]));	
	}
	
	return f_roi;
}

int detectNose(cv::Mat & frame)
{
	String face_cascade_name = "xml/haarcascade_frontalface_alt.xml";
//	String eyes_cascade_name = "xml/haarcascade_eye_tree_eyeglasses.xml";
	String nose_cascade_name="xml/haarcascade_mcs_nose.xml";
	
	CascadeClassifier face_cascade;
	CascadeClassifier nose_cascade;

	if(!face_cascade.load(face_cascade_name)){
		std::cout<<"error load face config file\n";
		return -1;
	}
	if(!nose_cascade.load(nose_cascade_name)){
		std::cout<<"error load nose config file\n";
		return -1;
	}

	std::vector<Rect> faces;
	Mat frm_gray;

	cvtColor(frame, frm_gray, CV_BGR2GRAY);
	equalizeHist(frm_gray, frm_gray);

	face_cascade.detectMultiScale(frm_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30,30));

	for(size_t i=0; i<faces.size(); i++){
		Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse(frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

		Mat faceROI=frm_gray(faces[i]);
		std::vector<Rect> nose;
		nose_cascade.detectMultiScale(faceROI, nose, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30,30));
		for(size_t j=0; j<nose.size(); j++){
			Point center(faces[i].x+nose[j].x+nose[j].width*0.5, faces[i].y+nose[j].y+nose[j].height*0.5);
			int radius=cvRound((nose[j].width+nose[j].height)*0.25);

			circle(frame, center, radius, Scalar(255,0,255),4,8,0);
		}
	}
	return 0;
}

int detectMouth(cv::Mat & frame)
{
	String face_cascade_name = "xml/haarcascade_frontalface_alt.xml";
	String mouth_cascade_name="xml/haarcascade_mcs_mouth.xml";
	
	CascadeClassifier face_cascade;
	CascadeClassifier mouth_cascade;

	if(!face_cascade.load(face_cascade_name)){
		std::cout<<"error load face config file\n";
		return -1;
	}
	if(!mouth_cascade.load(mouth_cascade_name)){
		std::cout<<"error load mouth config file\n";
		return -1;
	}


	std::vector<Rect> faces;
	Mat frm_gray;

	cvtColor(frame, frm_gray, CV_BGR2GRAY);
	equalizeHist(frm_gray, frm_gray);

	face_cascade.detectMultiScale(frm_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30,30));

	for(size_t i=0; i<faces.size(); i++){
		Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse(frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

		Mat faceROI=frm_gray(faces[i]);
		std::vector<Rect> mouth;
		mouth_cascade.detectMultiScale(faceROI, mouth, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30,30));
		for(size_t j=0; j<mouth.size(); j++){
			Point center(faces[i].x+mouth[j].x+mouth[j].width*0.5, faces[i].y+mouth[j].y+mouth[j].height*0.5);
			int radius=cvRound((mouth[j].width+mouth[j].height)*0.25);

			circle(frame, center, radius, Scalar(255,0,255),4,8,0);
		}
	}

	return 0;
}
/*
cimg_library::CImg<unsigned char> * cvImgToCImg(cv::Mat &cvImg)
{
  cimg_library::CImg<unsigned char> * result = new cimg_library::CImg<unsigned char>(cvImg.cols, cvImg.rows);

  for (int x = 0; x < cvImg.cols; ++x)
    for (int y = 0; y < cvImg.rows; ++y)
      (*result)(x, y) = cvImg.at<uchar>(y, x);

  return result;
}
cv::Mat & CImgtoCvImg(cv::Mat &result, cimg_library::CImg<unsigned char> *img)
{
  result = cv::Mat(img->height(), img->width(), CV_8U);

  for (int x=0; x < result.cols; ++x)
    for (int y=0; y < result.rows; ++y)
      result.at<uchar>(y, x) = (*img)(x, y);

  return result;
}



int detectLandmark(cv::Mat & frame)
{

	adjustImage(frame);

	cv::String face_cascade_name = "haarcascade_frontalface_alt.xml";
	cv::CascadeClassifier face_cascade;

//	std::string landmark_file="xml/INDIVIDUAL_FRONTAL_AFLW_SPLIT_1.xml";
	std::string landmark_file="xml/FDPM.xml";

  clandmark::Flandmark *flandmark = clandmark::Flandmark::getInstanceOf(landmark_file.c_str());
  if(!flandmark)
  {
     std::cerr << "Usage: static_input <flandmark_model.xml> <input_image> [<output_image>]" << std::endl;
    return -1;
  }

  clandmark::CFeaturePool *featurePool = new clandmark::CFeaturePool(flandmark->getBaseWindowSize()[0], flandmark->getBaseWindowSize()[1]);
  featurePool->addFeaturesToPool(new clandmark::CSparseLBPFeatures(
          featurePool->getWidth(),
          featurePool->getHeight(),
          featurePool->getPyramidLevels(),
          featurePool->getCumulativeWidths()
          )
        );
	flandmark->setNFfeaturesPool(featurePool);
	if( !face_cascade.load(face_cascade_name) )
  {
    printf("--(!)Error loading\n");
    return -1;
  };
	

  std::vector<cv::Rect> faces;
  cv::Mat frame_gray;
  int bbox[8];
  clandmark::fl_double_t *landmarks;

  cv::cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );
	double max_face_size=0;
	for(std::vector<cv::Rect>::iterator it=faces.begin(); it!=faces.end(); ++it){
		if((*it).width*(*it).height>max_face_size)
			max_face_size=(*it).width*(*it).height;	
	}

  for( uint32_t i = 0; i < faces.size(); i++ )
  {
    // Get detected face bounding box
  	  bbox[0] = faces[i].x;
  	  bbox[1] = faces[i].y;
  	  bbox[2] = faces[i].x+faces[i].width;
  	  bbox[3] = faces[i].y;
  	  bbox[4] = faces[i].x+faces[i].width;
  	  bbox[5] = faces[i].y+faces[i].height;
  	  bbox[6] = faces[i].x;
  	  bbox[7] = faces[i].y+faces[i].height;

//    	Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
//    	ellipse(frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );


			if((faces[i].width*faces[i].height)>max_face_size*0.6){
    	Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    	ellipse(frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
			cimg_library::CImg<unsigned char>* frm_gray = cvImgToCImg(frame_gray);
			flandmark->detect_optimized(frm_gray, bbox);
			delete frm_gray;
			landmarks = flandmark->getLandmarks();

      cv::Point start=cv::Point(int(landmarks[0]), int(landmarks[1]));
      cv::circle(frame,start,  2, cv::Scalar(255, 0, 0), -1);
	    for (int i=2; i < 2*flandmark->getLandmarksCount(); i+=2)
  	  {
    		  cv::Point end=cv::Point(int(landmarks[i]), int(landmarks[i+1]));
 //       	cv::line(frame, start, end, Scalar(255,0,55), 2);
          cv::circle(frame, end, 2, cv::Scalar(0, 0, 255), -1);
      		start=end;
    	}
			 	clandmark::printTimingStats(flandmark->timings);
        clandmark::printLandmarks(landmarks, flandmark->getLandmarksCount());
        clandmark::printLandmarks(flandmark->getLandmarksNF(), flandmark->getLandmarksCount());
		}
	}

  delete featurePool;
  delete flandmark;


	return 0;
}
*/
int detectPedestrian(cv::Mat & frame)
{
	try{
		vector<Rect> people;
	
		HOGDescriptor hog;
		hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

		hog.detectMultiScale(frame, people,0,Size(8,8),Size(0,0),1.03,2); 
	
		for(int i=0; i<people.size(); i++){
			Rect r=people[i];
			rectangle(frame, r.tl(), r.br(), Scalar(255,0,0), 2);
		}
	}
	catch(cv::Exception & ex)
	{
		std::cout<<"detectPedestrian: "<<ex.what()<<std::endl;
	}
	return 0;

}


void getMaxClass(dnn::Blob &probBlob, int *classId, double *classProb)
{
    Mat probMat = probBlob.matRefConst().reshape(1, 1); //reshape the blob to 1x1000 matrix
    Point classNumber;

    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}

std::vector<String> readClassNames(const char *filename )
{
    std::vector<String> classNames;

    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        std::cerr << "File with classes labels not found: " << filename << std::endl;
        exit(-1);
    }

    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back(name.substr(name.find(' ') + 1));
    }

    fp.close();
    return classNames;
}
string int_string(int a)
{
    std::stringstream ss;
    std::string str;
    ss << a;
    ss >> str;
    return str;
}

float mean(const std::vector<float>& v)
{
    assert(v.size() != 0);
    float ret = 0.0;
    for (std::vector<float>::size_type i = 0; i != v.size(); ++i)
    {
        ret += v[i];
    }
    return ret / v.size();
}

float cov(const std::vector<float>& v1, const std::vector<float>& v2)
{
    assert(v1.size() == v2.size() && v1.size() > 1);
    float ret = 0.0;
    float v1a = mean(v1), v2a = mean(v2);

    for (std::vector<float>::size_type i = 0; i != v1.size(); ++i)
    {
        ret += (v1[i] - v1a) * (v2[i] - v2a);
    }

    return ret / (v1.size() - 1);
}

// 相关系数
float coefficient(const std::vector<float>& v1, const std::vector<float>& v2)
{
    assert(v1.size() == v2.size());
    return cov(v1, v2) / sqrt(cov(v1, v1) * cov(v2, v2));
}
//cos 相似性度量
float cos_distance(const std::vector<float>& vecfeature1, vector<float>& vecfeature2)
{
    float cos_dis=0;
    float dotmal=0, norm1=0, norm2=0;
    for (int i = 0; i < vecfeature1.size(); i++)
    {
        dotmal += vecfeature1[i] * vecfeature2[i];
        norm1 += vecfeature1[i] * vecfeature1[i];
        norm2 += vecfeature2[i] * vecfeature2[i];

    }
    norm1 = sqrt(norm1);
    norm2 = sqrt(norm2);
    cos_dis = dotmal / (norm1*norm2);
    return cos_dis;
}

void read_csv(const string & filename, vector<Mat> & images, vector<int>& labels, char sep)
{
  std::ifstream file(filename.c_str(), ifstream::in);
  if(!file){
    std::cout<<"can't open file"<<std::endl;
    return ;
  }
  string line, path, classlabel;

  while(getline(file, line)){
    stringstream liness(line);
    getline(liness, path,sep);
    getline(liness, classlabel);
    if(!path.empty() && !classlabel.empty()){
			cv::Mat temp=imread(path,CV_LOAD_IMAGE_GRAYSCALE);
			cv::resize(temp, temp, Size(224,224));
      images.push_back(temp);
      labels.push_back(atoi(classlabel.c_str()));
    }
  }
}

std::vector<std::string> read_features(std::string & path)
{
	DIR *dp;
	struct dirent *entry;

	std::vector<std::string> temp;	

	std::cout<<"test........................\n";

	dp=opendir(path.c_str());
	if(dp){
		while(entry=readdir(dp)){
			if(strcmp(entry->d_name, ".")==0)
				continue;			
			if(strcmp(entry->d_name, "..")==0)
				continue;			
			temp.push_back(entry->d_name);
			std::cout<<entry->d_name<<std::endl;
		}
	}
		
	closedir(dp);
	return temp;
}

