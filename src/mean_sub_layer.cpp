#include "mean_sub_layer.hpp"

#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace caffe
{

template <typename Dtype>
void MeanSubLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const MeanSubParameter& param = this->layer_param_.mean_sub_param();

  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(param.mean_file(), &blob_proto);
  data_mean_.FromProto(blob_proto);
}

template <typename Dtype>
void MeanSubLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& src = *(bottom[0]);
  CHECK_EQ(src.channels(), data_mean_.channels());
  CHECK_EQ(src.height(), data_mean_.height());
  CHECK_EQ(src.width(), data_mean_.width());

  top[0]->ReshapeLike(src);
}

template <typename Dtype>
void MeanSubLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>& src = *(bottom[0]);
  Blob<Dtype>& dst = *(top[0]);

  dst.ShareData(src);

  const int LENGTH = dst.count(1);

  Dtype* mean_ptr = data_mean_.mutable_cpu_data();
  Dtype* data_iter = dst.mutable_cpu_data();

#ifndef NDEBUG
  std::vector<cv::Mat> mean_channels(3);
  mean_channels[0] = cv::Mat(cv::Size(data_mean_.width(),  data_mean_.height()),
                             CV_MAKETYPE((sizeof(Dtype) == sizeof(float)) ? CV_32F : CV_64F, 1),
                             mean_ptr);
  mean_channels[1] = cv::Mat(cv::Size(data_mean_.width(), data_mean_.height()),
                             CV_MAKETYPE((sizeof(Dtype) == sizeof(float)) ? CV_32F : CV_64F, 1),
                             mean_ptr + data_mean_.count(2));
  mean_channels[2] = cv::Mat(cv::Size(data_mean_.width(), data_mean_.height()),
                             CV_MAKETYPE((sizeof(Dtype) == sizeof(float)) ? CV_32F : CV_64F, 1),
                             mean_ptr +  + data_mean_.count(2) * 2);
  cv::Mat mean_mat;
  cv::merge(mean_channels, mean_mat);
#endif // !NDEBUG

  for (int n = dst.num(); n--;) {
#ifndef NDEBUG
    std::vector<cv::Mat> src_channels(3);
    src_channels[0] = cv::Mat(cv::Size(data_mean_.width(), data_mean_.height()),
                              CV_MAKETYPE(sizeof(Dtype) == sizeof(float) ? CV_32F : CV_64F, 1),
                              data_iter);
    src_channels[1] = cv::Mat(cv::Size(data_mean_.width(), data_mean_.height()),
                               CV_MAKETYPE(sizeof(Dtype) == sizeof(float) ? CV_32F : CV_64F, 1),
                               data_iter + data_mean_.count(2));
    src_channels[2] = cv::Mat(cv::Size(data_mean_.width(), data_mean_.height()),
                               CV_MAKETYPE(sizeof(Dtype) == sizeof(float) ? CV_32F : CV_64F, 1),
                               data_iter +  + data_mean_.count(2) * 2);
    cv::Mat src_mat;
    cv::merge(src_channels, src_mat);
#endif // !NDEBUG

    caffe_axpy(LENGTH, static_cast<Dtype>(-1), mean_ptr, data_iter);

#ifndef NDEBUG
    std::vector<cv::Mat> sub_channels(3);
    sub_channels[0] = cv::Mat(cv::Size(data_mean_.width(), data_mean_.height()),
                              CV_MAKETYPE(sizeof(Dtype) == sizeof(float) ? CV_32F : CV_64F, 1),
                              data_iter);
    sub_channels[1] = cv::Mat(cv::Size(data_mean_.width(), data_mean_.height()),
                               CV_MAKETYPE(sizeof(Dtype) == sizeof(float) ? CV_32F : CV_64F, 1),
                               data_iter + data_mean_.count(2));
    sub_channels[2] = cv::Mat(cv::Size(data_mean_.width(), data_mean_.height()),
                               CV_MAKETYPE(sizeof(Dtype) == sizeof(float) ? CV_32F : CV_64F, 1),
                               data_iter +  + data_mean_.count(2) * 2);
    cv::Mat sub_mat;
    cv::merge(sub_channels, sub_mat);
#endif // !NDEBUG

    data_iter += LENGTH;
  }
}
#ifdef CPU_ONLY
STUB_GPU(MeanSubLayer);
#endif

INSTANTIATE_CLASS(MeanSubLayer);
REGISTER_LAYER_CLASS(MeanSub);
} // namespace caffe