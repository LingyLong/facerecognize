#include "extractfeature.h"

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

int main()
{
	caffe_predefine();
	Mat lena=imread("../pic/ml/s1/ly1.jpg");
	cv::resize(lena, lena, Size(224,244));
	vector<float> lena_vec=extractfeature(lena);
	Mat image=imread("../pic/ml/s1/ly0.jpg");
	cv::resize(image, image, Size(224,244));
	vector<float> img_vec=extractfeature(image);

	cout<< coefficient(lena_vec, img_vec)<<endl;
	
	return 0;
}

