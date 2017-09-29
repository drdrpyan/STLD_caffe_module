#include "padding_layer.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace caffe
{

template <typename Dtype>
PaddingLayer<Dtype>::PaddingLayer(const LayerParameter& param) 
  : Layer<Dtype>(param), TYPE_(param.padding_param().type()),
    PAD_UP_(param.padding_param().pad_up()), 
    PAD_DOWN_(param.padding_param().pad_down()),
    PAD_LEFT_(param.padding_param().pad_left()),
    PAD_RIGHT_(param.padding_param().pad_right()) {
  CHECK(TYPE_ == PaddingParameter::PaddingType::PaddingParameter_PaddingType_MIRROR)
    << "Not implemented yet.";
}

template <typename Dtype>
void PaddingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); i++) {
    std::vector<int> top_shape = bottom[i]->shape();
    top_shape[2] += (PAD_UP_+ PAD_DOWN_);
    top_shape[3] += (PAD_LEFT_ + PAD_RIGHT_);
    top[i]->Reshape(top_shape);
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  switch (TYPE_) {
    case PaddingParameter::PaddingType::PaddingParameter_PaddingType_MIRROR:
      MirrorPadding(bottom, top);
      break;
    default:
      LOG(ERROR) << "Illegal padding type.";
  }
}

template <typename Dtype>
void PaddingLayer<Dtype>::MirrorPadding(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) const {
  for (int i = 0; i < bottom.size(); i++) {
    const Blob<Dtype>& src = *(bottom[i]);
    Blob<Dtype>& dst = *(top[i]);

    for (int n = 0; n < src.num(); n++) {
      for (int c = 0; c < src.channels(); c++) {

        // padding left, right
        for (int h = 0; h < src.height(); h++) {
          const Dtype* src_row = src.cpu_data() + src.offset(n, c, h);
          Dtype* dst_row = dst.mutable_cpu_data() + dst.offset(n, c, h + PAD_UP_);

          Dtype* left_dst = dst.mutable_cpu_data() + dst.offset(n, c, h + PAD_UP_);
          Dtype* right_dst = dst.mutable_cpu_data() +
              dst.offset(n, c, h + PAD_UP_, src.width() - PAD_RIGHT_);
          Dtype* mid_dst = dst.mutable_cpu_data() +
              dst.offset(n, c, h + PAD_UP_, PAD_LEFT_);

          std::reverse_copy(src_row, src_row + PAD_LEFT_, dst_row);
          std::reverse_copy(src_row + src.width() - PAD_RIGHT_,
                            src_row + src.width(),
                            dst_row + dst.width() - PAD_RIGHT_);
          caffe_copy(src.width(), src_row, dst_row + PAD_LEFT_);
        }

        // padding up
        for (int j = 0; j < PAD_UP_; j++) {
          const Dtype* copy_src = dst.cpu_data() + dst.offset(n, c, PAD_UP_ + j);
          Dtype* copy_dst = dst.mutable_cpu_data() + dst.offset(n, c, PAD_UP_ - 1 - j);
          caffe_copy(dst.width(), copy_src, copy_dst);
        }

        // padding down
        for (int j = 0; j < PAD_DOWN_; j++) {
          const Dtype* copy_src = dst.cpu_data() + 
              dst.offset(n, c, dst.height() - PAD_DOWN_ - 1 - j);
          Dtype* copy_dst = dst.mutable_cpu_data() + 
              dst.offset(n, c, dst.height() - PAD_DOWN_ + j);
          caffe_copy(dst.width(), copy_src, copy_dst);
        }
      }

      //std::vector<cv::Mat> channel;
      //channel.push_back(cv::Mat(dst.height(), dst.width(), CV_32FC1,
      //                          dst.mutable_cpu_data() + dst.offset(n, 0)));
      //channel.push_back(cv::Mat(dst.height(), dst.width(), CV_32FC1,
      //                     dst.mutable_cpu_data() + dst.offset(n, 1)));
      //channel.push_back(cv::Mat(dst.height(), dst.width(), CV_32FC1,
      //                     dst.mutable_cpu_data() + dst.offset(n, 2)));
      //cv::Mat bgr, bgr2;
      //cv::merge(channel, bgr);
      //bgr.convertTo(bgr2, CV_8UC3);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PaddingLayer);
#endif

INSTANTIATE_CLASS(PaddingLayer);
REGISTER_LAYER_CLASS(Padding);

} // namespace caffe