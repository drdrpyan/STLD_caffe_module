#ifndef BGM_IMG_DECODER_HPP_
#define BGM_IMG_DECODER_HPP_

#include <opencv2/core.hpp>

#include "caffe/blob.hpp"

#include <vector>

namespace bgm
{

template <typename Dtype>
class ImgDecoder
{
 public:
  void Decode(const caffe::Blob<Dtype>& img_blob,
              std::vector<cv::Mat>* img);
  void Decode(const caffe::Blob<Dtype>& img_blob,
              int img_elem_type,
              std::vector<cv::Mat>* img);

 private:
  cv::Mat ConvertElemType(const cv::Mat& src, int dst_elem_type) const;

}; // class ImgDecoder

// template functions
template <typename Dtype>
inline void ImgDecoder<Dtype>::Decode(const caffe::Blob<Dtype>& img_blob,
                                      std::vector<cv::Mat>* img) {
  Decode(img_blob, (sizeof(Dtype) == sizeof(float) ? CV_32F : CV_64F),
         img);
}

template <typename Dtype>
void ImgDecoder<Dtype>::Decode(const caffe::Blob<Dtype>& img_blob,
                               int img_elem_type,
                               std::vector<cv::Mat>* img) {
  CHECK(img);
  img->resize(img_blob.num());

  const int BATCH = img_blob.num();
  const int CHANNEL = img_blob.channels();
  const cv::Size IMG_SIZE(img_blob.width(), img_blob.height());
  const int CH_MAT_TYPE = 
      CV_MAKETYPE((sizeof(Dtype) == sizeof(float)) ? CV_32F : CV_64F, 1);

  Dtype* img_blob_data = const_cast<Dtype*>(img_blob.cpu_data());

  for (int n = 0; n < BATCH; ++n) {
    std::vector<cv::Mat> ch_mat(CHANNEL);
    for (int c = 0; c < CHANNEL; ++c) {
      ch_mat[c] = cv::Mat(IMG_SIZE, CH_MAT_TYPE,
                          img_blob_data + img_blob.offset(n, c));
    }

    cv::Mat decoded;
    cv::merge(ch_mat, decoded);

    (*img)[n] = ConvertElemType(decoded, img_elem_type);
  }
}

template <typename Dtype>
cv::Mat ImgDecoder<Dtype>::ConvertElemType(
    const cv::Mat& src, int dst_elem_type) const {
  CHECK(dst_elem_type == CV_8U
        || dst_elem_type == CV_32F
        || dst_elem_type == CV_64F) << "Illegal cv::Mat type";

  if (src.depth() == dst_elem_type)
    return src;
  else {
    cv::Mat dst;
    int mat_type = CV_MAKETYPE(dst_elem_type, src.channels());
    src.convertTo(dst, mat_type);
    return dst;
  }
}

} // namespace bgm

#endif // !BGM_IMG_DECODER_HPP_