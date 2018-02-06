#include "detection_check_layer.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace caffe
{
template <typename Dtype>
void DetectionCheckLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const DetectionCheckParameter& param = layer_param_.detection_check_param();

  eval_performance_ = param.eval_performance();
  draw_ = param.draw();

  conf_threshold_ = param.detection_conf_threshold();
  iou_threshold_ = param.iou_threshold();

  log_path_ = param.log_path();
  OpenOutStreams(log_path_);

  detection_only_ = param.detection_only();

  img_decoder_.reset(new bgm::ImgDecoder<Dtype>);
  detection_decoder_.reset(new bgm::DetectionDecoder<Dtype>);
  anno_decoder_.reset(new bgm::AnnoDecoder<Dtype>);

  img_cnt_ = 0;
  tp_ = 0;
  fp_ = 0;
  fn_ = 0;

  do_nms_ = param.do_nms();
  if (do_nms_) {
    float nms_overlap_threshold = param.nms_overlap_threshold();
    //bgm::DetectionNMS<Dtype>* detection_nms = new bgm::ConfMaxVOCNMS<Dtype>(nms_overlap_threshold);
    bgm::DetectionNMS<Dtype>* detection_nms = new bgm::MeanSizeNMS<Dtype>(nms_overlap_threshold);
    nms_.reset(detection_nms);
  }
}

template <typename Dtype>
void DetectionCheckLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void DetectionCheckLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  std::vector<std::vector<bgm::Detection<Dtype> > > detection;
  std::vector<cv::Mat> img;
  std::vector<std::vector<bgm::BoxAnnotation<Dtype> > > gt;

  DecodeDetection(bottom, &detection);
  DecodeImg(bottom, &img);
  CHECK_EQ(detection.size(), img.size());
  LogDetection(detection);
  SaveImg(log_path_ + "/img/", img);

  if (eval_performance_) {
    DecodeGT(bottom, &gt);
    CHECK_EQ(detection.size(), gt.size());
    LogGT(gt);
  }

  if (eval_performance_ || draw_) {
    for (int n = 0; n < detection.size(); ++n) {
      if (eval_performance_) {
        std::vector<int> tp_idx, fp_idx, fn_idx;
        bgm::DetectionMatching<Dtype>(detection[n], gt[n], 
                                      conf_threshold_, iou_threshold_,
                                      &tp_idx, &fp_idx, &fn_idx);
        LogEval(img_cnt_ + n, tp_idx, fp_idx, fn_idx);
        tp_ += tp_idx.size();
        fp_ += fp_idx.size();
        fn_ += fn_idx.size();

        if (draw_) {
          cv::Mat result_img = DrawResultGT(detection[n], gt[n],
                                            tp_idx, fp_idx, fn_idx, img[n]);
          cv::imwrite(log_path_ + "/result_img/" + GetImgName(img_cnt_ + n),
                      result_img);
        }
      }
      else if (draw_) {
        cv::Mat result_img = DrawResult(detection[n], img[n]);
        cv::imwrite(log_path_ + "/result_img/" + GetImgName(img_cnt_ + n),
                      result_img);
      }
    }
  }

  img_cnt_ += img.size();
}

template <typename Dtype>
inline void DetectionCheckLayer<Dtype>::DecodeDetection(
    const std::vector<Blob<Dtype>*>& bottom,
    std::vector<std::vector<bgm::Detection<Dtype> > >* detection) {
  std::vector<Blob<Dtype>*> detection_blobs(bottom.begin(),
                                            bottom.begin() + 3); // 0, 1, 2

  if(!do_nms_)
    detection_decoder_->Decode(detection_blobs, detection);
  else {
    std::vector<std::vector<bgm::Detection<Dtype> > > temp_detection;
    detection_decoder_->Decode(detection_blobs, &temp_detection);

    detection->resize(temp_detection.size());

    for (int i = 0; i < temp_detection.size(); ++i)
      (*nms_)(temp_detection[i], &((*detection)[i]));
  }
}

template <typename Dtype>
void DetectionCheckLayer<Dtype>::LogDetection(
    const std::vector<std::vector<bgm::Detection<Dtype> > >& detection) {
  for (int i = 0; i < detection.size(); ++i) {
    result_log_ << GetImgName(img_cnt_ + i) << ' ' << detection[i].size();

    for (int j = 0; j < detection[i].size(); ++j) {
      result_log_ << ' ' << detection[i][j].label;
      result_log_ << ' ' << detection[i][j].conf;
      result_log_ << ' ' << detection[i][j].bbox.x;
      result_log_ << ' ' << detection[i][j].bbox.y;
      result_log_ << ' ' << detection[i][j].bbox.width;
      result_log_ << ' ' << detection[i][j].bbox.height;
    }

    result_log_ << std::endl;
  }
}

template <typename Dtype>
void DetectionCheckLayer<Dtype>::SaveImg(
    const std::string& path, const std::vector<cv::Mat>& img) const {
  for (int i = 0; i < img.size(); ++i) {
    std::string img_name = path + "/" + GetImgName(img_cnt_ + i);
    cv::imwrite(img_name, img[i]);
  }
}

template <typename Dtype>
void DetectionCheckLayer<Dtype>::LogGT(
    const std::vector<std::vector<bgm::BoxAnnotation<Dtype> > >& gt) {
  for (int i = 0; i < gt.size(); ++i) {
    gt_log_ << GetImgName(img_cnt_ + i) << ' ' << gt[i].size();

    for (int j = 0; j < gt[i].size(); ++j) {
      gt_log_ << ' ' << gt[i][j].label;
      gt_log_ << ' ' << gt[i][j].bbox.x;
      gt_log_ << ' ' << gt[i][j].bbox.y;
      gt_log_ << ' ' << gt[i][j].bbox.width;
      gt_log_ << ' ' << gt[i][j].bbox.height;
    }

    gt_log_ << std::endl;
  }
}

template <typename Dtype>
void DetectionCheckLayer<Dtype>::LogEval(int img_cnt,
                                         const std::vector<int>& tp_idx,
                                         const std::vector<int>& fp_idx,
                                         const std::vector<int>& fn_idx) {
  eval_log_ << GetImgName(img_cnt) << std::endl;
  
  eval_log_ << '\t' << tp_idx.size();
  for (int i = 0; i < tp_idx.size(); ++i)
    eval_log_ << ' ' << tp_idx[i];
  eval_log_ << std::endl;

  eval_log_ << '\t' << fp_idx.size();
  for (int i = 0; i < fp_idx.size(); ++i)
    eval_log_ << ' ' << fp_idx[i];
  eval_log_ << std::endl;

  eval_log_ << '\t' << fn_idx.size();
  for (int i = 0; i < fn_idx.size(); ++i)
    eval_log_ << ' ' << fn_idx[i];
  eval_log_ << std::endl;
}

template <typename Dtype>
cv::Mat DetectionCheckLayer<Dtype>::DrawResult(
    const std::vector<bgm::Detection<Dtype> >& detection,
    const cv::Mat& img) const {
  cv::Mat result_img = img.clone();
  for (int i = 0; i < detection.size(); ++i)
    cv::rectangle(result_img, detection[i].bbox, cv::Scalar(0, 255, 0));
  return result_img;
}

template <typename Dtype>
cv::Mat DetectionCheckLayer<Dtype>::DrawResultGT(
    const std::vector<bgm::Detection<Dtype> >& detection,
    const std::vector<bgm::BoxAnnotation<Dtype> >& gt,
    const std::vector<int>& tp_idx,
    const std::vector<int>& fp_idx,
    const std::vector<int>& fn_idx,
    const cv::Mat& img) const {
  const cv::Scalar TP_COLOR(0, 255, 0);
  const cv::Scalar FP_COLOR(0, 0, 255);
  const cv::Scalar GT_COLOR(0x66, 0x66, 0x66);
  const cv::Scalar FN_COLOR(255, 0, 0);

  cv::Mat result_img = img.clone();
  
  for (int i = 0; i < gt.size(); ++i) {
    if (std::find(fn_idx.begin(), fn_idx.end(), i) != fn_idx.end())
      cv::rectangle(result_img, gt[i].bbox, GT_COLOR);
    else
      cv::rectangle(result_img, gt[i].bbox, FN_COLOR);
  }

  for (int i = 0; i < detection.size(); ++i) {
    if(std::find(fp_idx.begin(), fp_idx.end(), i) != fp_idx.end())
      cv::rectangle(result_img, detection[i].bbox, FP_COLOR);
    else if(std::find(tp_idx.begin(), tp_idx.end(), i) != tp_idx.end())
      cv::rectangle(result_img, detection[i].bbox, TP_COLOR);
  }

  return result_img;
}

template <typename Dtype>
std::string DetectionCheckLayer<Dtype>::GetImgName(int img_cnt) const {
  char img_name[256];
  std::sprintf(img_name, "%06d.jpg", img_cnt);
  return img_name;
}

#ifdef CPU_ONLY
STUB_GPU(DetectionCheckLayer);
#endif

INSTANTIATE_CLASS(DetectionCheckLayer);
REGISTER_LAYER_CLASS(DetectionCheck);

} // namespace caffe