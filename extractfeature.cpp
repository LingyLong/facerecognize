#include "extractfeature.h"
#include "caffe_net_memorylayer.h"
#include <boost/shared_ptr.hpp>

namespace caffe
{
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);
	extern INSTANTIATE_CLASS(ConvolutionLayer);
//	REGISTER_LAYER_CLASS(Convolution);
	extern INSTANTIATE_CLASS(ReLULayer);
//	REGISTER_LAYER_CLASS(ReLU);
	extern INSTANTIATE_CLASS(PoolingLayer);
//	REGISTER_LAYER_CLASS(Pooling);
	extern INSTANTIATE_CLASS(LRNLayer);
//	REGISTER_LAYER_CLASS(LRN);
	extern INSTANTIATE_CLASS(SoftmaxLayer);
//	REGISTER_LAYER_CLASS(Softmax);
	extern INSTANTIATE_CLASS(MemoryDataLayer);
}

template <typename Dtype>
caffe::Net<Dtype>* Net_Init_Load(std::string param_file, std::string pretrained_file, caffe::Phase phase)
{
	caffe::Net<Dtype>* net(new caffe::Net<Dtype>("xml/vgg_face_memorydata.prototxt", caffe::TEST));
	net->CopyTrainedLayersFrom("xml/VGG_FACE.caffemodel");
	return net;
}



std::vector<float> extractfeature(Mat faceROI)
{
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	std::vector<Mat> test;
	std::vector<int> testlabel;
	std::vector<float> test_vector;
	test.push_back(faceROI);
	testlabel.push_back(0);
	memory_layer->AddMatVector(test, testlabel);
	test.clear();
	testlabel.clear();
	std::vector<caffe::Blob<float>*> input_vec;
	net->Forward(input_vec);
	boost::shared_ptr<caffe::Blob<float> > fc8=net->blob_by_name("fc8");
	int test_num=0;
	while(test_num<2622)
		test_vector.push_back(fc8->data_at(0, test_num++, 1, 1));

	return test_vector;
}

void caffe_predefine()
{
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	net=Net_Init_Load<float>("xml/vgg_face_memorydata.prototxt", "xml/VGG_FACE.caffemodel", caffe::TEST);
	memory_layer=(caffe::MemoryDataLayer<float> *)net->layers()[0].get();
}

