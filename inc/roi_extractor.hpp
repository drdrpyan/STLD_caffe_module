#ifndef BGM_ROI_EXTRACTOR_HPP_
#define BGM_ROI_EXTRACTOR_HPP_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "caffe/blob.hpp"

namespace bgm
{

template <typename Dtype>
class ROIExtractor
{
 public:
  virtual void Extract(
      const caffe::Blob<Dtype>& feature_map, const cv::Size& img_size,
      const std::vector<std::vector<cv::Rect_<Dtype> > >& img_roi,
      caffe::Blob<Dtype>* roi_feature_map) = 0;
};

template <typename Dtype>
class ROIAlignExtractor : public ROIExtractor<Dtype>
{
 public:
  virtual void Extract(
      const caffe::Blob<Dtype>& feature_map, const cv::Size& img_size,
      const std::vector<std::vector<cv::Rect_<Dtype> > >& img_roi,
      caffe::Blob<Dtype>* roi_feature_map) override;

 private:
  void Extract(const cv::Mat& feature_map,
               const cv::Size& img_size, const cv::Rect_<Dtype>& img_roi,
               cv::Mat& extracted);
};

// template functions
template <typename Dtype>
void ROIAlignExtractor<Dtype>::Extract(
    const caffe::Blob<Dtype>& feature_map, const cv::Size& img_size,
    const std::vector<std::vector<cv::Rect_<Dtype> > >& img_roi,
    caffe::Blob<Dtype>* roi_feature_map) {
  CHECK(roi_feature_map);
  //CHECK_EQ(feature_map.num(), roi_feature_map->num());
  CHECK_EQ(feature_map.channels(), roi_feature_map->channels());

  const int MAT_DEPTH = (sizeof(float) == sizeof(Dtype)) ? CV_32F : CV_64F;
  const int MAT_TYPE = CV_MAKETYPE(MAT_DEPTH, 1);

  const int BATCH_SIZE = feature_map.num();
  const int CHANNEL = feature_map.channels();
  const cv::Size SRC_SIZE(feature_map.width(), feature_map.height());
  const cv::Size DST_SIZE(roi_feature_map->width(), roi_feature_map->height());

  const int SRC_STEP = SRC_SIZE.area();
  const int DST_STEP = DST_SIZE.area();

  const Dtype* src_iter = feature_map.cpu_data();
  Dtype* dst_iter = roi_feature_map->mutable_cpu_data();

  for (int n = feature_map.num(); n--;) {
    const std::vector<cv::Rect_<Dtype> >& current_rois = img_roi[n];
    cv::Mat src_mat(SRC_SIZE, MAT_TYPE,
                      const_cast<Dtype*>(src_iter));

    for (int r = 0; r < current_rois.size(); ++r) {
      for (int c = feature_map.channels(); c--;) {
        cv::Mat dst_mat(DST_SIZE, MAT_TYPE, dst_iter);
        Extract(src_mat, img_size, current_rois[r], dst_mat);
        dst_iter += DST_STEP;
      }
    }
  }
}

template <typename Dtype>
void ROIAlignExtractor<Dtype>::Extract(
    const cv::Mat& feature_map,
    const cv::Size& img_size, const cv::Rect_<Dtype>& img_roi,
    cv::Mat& extracted) {
  CHECK(!extracted.empty());
  CHECK_EQ(feature_map.type(), extracted.type());

  const float IMG_TO_FM_W_SCALE = feature_map.cols / static_cast<float>(img_size.width);
  const float IMG_TO_FM_H_SCALE = feature_map.rows / static_cast<float>(img_size.height);
  const cv::Rect_<Dtype> FM_ROI(img_roi.x * IMG_TO_FM_W_SCALE,
                                img_roi.y * IMG_TO_FM_H_SCALE,
                                img_roi.width * IMG_TO_FM_W_SCALE,
                                img_roi.height * IMG_TO_FM_H_SCALE);

  const float DST_TO_FMROI_W_SCALE = FM_ROI.width / extracted.cols;
  const float DST_TO_FMROI_H_SCALE = FM_ROI.height / extracted.rows;

  for (int y = 0; y < extracted.cols; ++y) {
    for (int x = 0; x < extracted.cols; ++x) {
      float fm_x = x * DST_TO_FMROI_W_SCALE + FM_ROI.x;
      float fm_y = y * DST_TO_FMROI_H_SCALE + FM_ROI.y;

      if (fm_x < 0 || fm_y < 0 || fm_x > feature_map.cols - 1 || fm_y > feature_map.rows - 1)
        extracted(cv::Rect(x, y, 1, 1)) = 0;
      else
        cv::getRectSubPix(feature_map, cv::Size(1, 1), cv::Point2f(fm_x, fm_y),
                          extracted(cv::Rect(x, y, 1, 1)));
    }
  }

}

//template <typename Dtype>
//void ROIAlignExtractor<Dtype>::GetPixel(
//    const cv::Size& img_size, const cv::Rect_<Dtype>& img_roi,
//    const cv::Size& feature_map_size, const cv::Size& dst_size,
//    std::vector<cv::Point2f>* x, std::vector<cv::Point2f>* y) const {
//  CHECK(x);
//  CHECK(y);
//
//  const float IMG_TO_FM_W_SCALE = feature_map_size.width / static_cast<float>(img_size.width);
//  const float IMG_TO_FM_H_SCALE = feature_map_size.height / static_cast<float>(img_size.height);
//  const cv::Rect_<Dtype> fm_roi(img_roi.x * IMG_TO_FM_W_SCALE,
//                          img_roi.y * IMG_TO_FM_H_SCALE,
//                          img_roi.width * IMG_TO_FM_W_SCALE,
//                          img_roi.height * IMG_TO_FM_H_SCALE);
//  
//  const float ROI_TO_DST_W_SCALE = dst_size.width / img_roi.width;
//  const float ROI_TO_DST_H_SCALE = dst_size.height / img_roi.height;
//
//  x->resize(dst_size.width);
//  y->resize(dst_size.height);
//  for(int i)
//  fm_roi.w
//}

} // namespace bgm

#endif // !BGM_ROI_EXTRACTOR_HPP_
