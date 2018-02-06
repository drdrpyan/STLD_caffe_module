//#ifndef BGM_BLOB_ROI_HPP_
//#define BGM_BLOB_ROI_HPP_
//
//#include <vector>
//
//#include <opencv2/core.hpp>
//
//#include "caffe/blob.hpp"
//
//namespace bgm
//{
//
//template <typename Dtype>
//class BlobROI
//{
// public:
//  BlobROI(caffe::Blob<Dtype>& blob, int n);
//  BlobROI(caffe::Blob<Dtype>& blob, const cv::Rect& roi);
//  BlobROI(caffe::Blob<Dtype>& blob, const cv::Rect& roi,
//          const cv::Range& ch_range);
//  BlobROI(caffe::Blob<Dtype>& blob,
//          const cv::Range& x_range, const cv::Range& y_range);
//  BlobROI(caffe::Blob<Dtype>& blob,
//          const cv::Range& x_range, const cv::Range& y_range,
//          const cv::Range& ch_range);
//
//  void CopyTo(BlobROI& dst) const;
//  int num() const;
//  int channels() const;
//  int height() const;
//  int width() const;
//
// private:
//  bool CheckRange(const BlobROI& roi1, const BlobROI& roi2) const;
//
//  mutable caffe::Blob<Dtype>* blob_;
//  bool whole_wh_;
//  bool whole_ch_;
//  cv::Rect range_wh_;
//  cv::Range range_ch_;
//};
//
//} // namespace bgm
//
//#endif // !BGM_BLOB_ROI_HPP_
