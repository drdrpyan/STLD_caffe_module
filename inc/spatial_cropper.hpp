//#ifndef BGM_SPATIAL_CROPPER_HPP_
//#define BGM_SPATIAL_CROPPER_HPP_
//
//#include <opencv2/core.hpp>
//
//#include "caffe/blob.hpp"
//#include "caffe/util/math_functions.hpp"
//
//namespace bgm
//{
//
//template <typename Dtype>
//class SpatialCropper
//{
// public:
//  void Crop(const caffe::Blob<Dtype>& src, 
//            const cv::Rect& roi, Dtype* dst);
//
//  void Crop_cpu(const caffe::Blob<Dtype>& src,
//                const cv::Rect& roi, Dtype* dst);
//  void Crop_gpu(const caffe::Blob<Dtype>)
//
// private:
//  void CropOneChannel(const Dtype* src, int src_width, 
//                      int dst_width, int dst_height, Dtype* dst) const;
//};
//
//// template functions
//template <typename Dtype>
//void SpatialCropper<Dtype>::Crop(const caffe::Blob<Dtype>& src,
//                                 const cv::Rect& roi, Dtype* dst) {
//  CHECK_GE(roi.x, 0);
//  CHECK_GT(roi.width, 0);
//  CHECK_LT(roi.x + roi.width, src.width());
//  CHECK_GE(roi.y, 0);
//  CHECK_GT(roi.height, 0);
//  CHECK_LT(roi.y + roi.height, src.height());
//  CHECK(dst);
//
//
//  for(int n=0; n)
//}
//
//template <typename Dtype>
//void SpatialCropper<Dtype>::CropOneChannel(
//    const Dtype* src, int src_width, 
//    int dst_width, int dst_height, Dtype* dst) const {
//  CHECK(src);
//  CHECK_GT(src_width, 0);
//  CHECK_GT(dst_width, 0);
//  CHECK_GT(dst_height, 0);
//  CHECK(dst);
//
//  const Dtype* src_iter = src;
//  Dtype* dst_iter = dst;
//  for (int i = dst_height; i--; ) {
//    caffe::caffe_copy<Dtype>(dst_width, src_iter, dst_iter);
//    src_iter += src_width;
//    dst_iter += dst_width;
//  }
//}
//
//} // namespace bgm
//
//#endif // !BGM_BLOB_CROPPER_HPP_
