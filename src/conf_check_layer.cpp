#include "conf_check_layer.hpp"

#include "detection_util.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>

namespace caffe
{
template <typename Dtype>
void ConfCheckLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const ConfCheckParameter& param = (this->layer_param_).conf_check_param();

  out_path_ = param.out_path();
  draw_ = param.draw();

  if ((this->layer_param_).has_subwin_offset_param()) {
    const SubwinOffsetParameter& offset_param = (this->layer_param_).subwin_offset_param();
    win_size_.width = offset_param.win_size().width();
    win_size_.height = offset_param.win_size().height();

    win_offset_.resize(offset_param.win_offset_size());
    for (int i = 0; i < win_offset_.size(); ++i) {
      win_offset_[i].x = offset_param.win_offset(i).x();
      win_offset_[i].y = offset_param.win_offset(i).y();
    }

    global_detection_ = offset_param.global_detection();
  }
  else
    global_detection_ = false;

  img_cnt_ = 0;

  log_.open(out_path_ + "/log.txt");
  anno_decoder_.reset(new bgm::AnnoDecoder<Dtype>);
  img_decoder_.reset(new bgm::ImgDecoder<Dtype>);
  obj_contained_.reset(new bgm::IntersectionOverObjContained<Dtype>(0.7f));
}

template <typename Dtype>
void ConfCheckLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  if (global_detection_) {
    CHECK_EQ(bottom[0]->num(), win_offset_.size());
    CHECK_EQ(bottom[1]->num(), 1);
  }
  else
    CHECK_EQ(bottom[0]->num(), bottom[1]->num());

  CHECK_EQ(bottom[0]->count(), bottom[0]->num());
}

template <typename Dtype>
void ConfCheckLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  std::vector<bool> gt;
  DecodeGT(*(bottom[1]), &gt);

  const Dtype* conf = bottom[0]->cpu_data();
  std::vector<Dtype> conf_vec(conf, conf + bottom[0]->count());

  if (global_detection_) {
    std::string img_name = GetImgName(img_cnt_++);

    for (int i = 0; i < gt.size(); ++i) {  
      log_ << std::to_string(gt[i] ? 1 : 0) << ' ' << std::to_string(conf_vec[i]) << std::endl;
    }

    if (draw_) {
      std::vector<cv::Mat> src_img;
      img_decoder_->Decode(*(bottom[2]), &src_img);
      cv::Mat result_img = DrawGlobalResult(src_img[0], gt, conf_vec);

      cv::imwrite(out_path_ + '/' + img_name, result_img);
    }
  }
  else {
    std::vector<cv::Mat> src_img;
    if (draw_)
      img_decoder_->Decode(*(bottom[2]), &src_img);

    for (int i = 0; i < gt.size(); ++i) {
      std::string img_name = GetImgName(img_cnt_++);

      log_ << img_name << ' ' << (gt[i] ? 1 : 0);
      log_ << ' ' << bgm::Sigmoid(conf_vec[i]) << std::endl;
      //log_ << ' ' << conf_vec[i] << std::endl;

      if (draw_) {
        cv::Mat result_img = DrawResult(src_img[i], gt[i], bgm::Sigmoid(conf_vec[i]));
        std::string out_path = out_path_ + '/' + img_name;
        cv::imwrite(out_path, result_img);
      }
    }
  }
    
}

template <typename Dtype>
void ConfCheckLayer<Dtype>::DecodeGT(
    const Blob<Dtype>& gt_blob, std::vector<bool>* pos_neg) const {
  CHECK(pos_neg);
  if (global_detection_) {
    LOG(FATAL) << "Not implemented yet";

    std::vector<std::vector<Dtype> > gt_label;
    std::vector<std::vector<cv::Rect_<Dtype> > > gt_bbox;
    //anno_decoder_->Decode(gt_)
  }
  else {
    const int NUM_BATCH = gt_blob.num();
    const int NUM_ELEM = gt_blob.count() / NUM_BATCH;

    const Dtype* gt_ptr = gt_blob.cpu_data();

    pos_neg->resize(NUM_BATCH);
    for (int i = 0; i < NUM_BATCH; ++i) {
      const Dtype* gt_iter = gt_ptr + gt_blob.offset(i);

      bool empty = true;
      for (int j = 0; j < NUM_ELEM && empty; ++j) {
        empty = (gt_iter[j] == LabelParameter::NONE);
        empty = empty || (gt_iter[j] == LabelParameter::DUMMY_LABEL);
      }

      (*pos_neg)[i] = !empty;
    }
  }
}

template <typename Dtype>
cv::Mat ConfCheckLayer<Dtype>::DrawGlobalResult(
    const cv::Mat& src, const std::vector<bool>& gt,
    const std::vector<Dtype>& conf) const {
  CHECK_EQ(gt.size(), conf.size());

  cv::Mat img = src.clone();

  cv::Mat color_table;
  cv::applyColorMap(cv::Mat(conf), color_table, cv::COLORMAP_JET);

  for (int i = 0; i < gt.size(); ++i) {
    cv::Rect roi(win_offset_[i], win_size_);
    cv::Scalar color = color_table.at<cv::Vec3b>(i, 0);

    cv::rectangle(img, roi, color, 2);
    cv::putText(img, std::to_string(gt[i]) + ' ' + std::to_string(conf[i]),
                cv::Point(roi.x + 10, roi.y + 10), 1, 3, color, 2);
  }
}

template <typename Dtype>
cv::Mat ConfCheckLayer<Dtype>::DrawResult(const cv::Mat& src, 
                                          bool gt, Dtype conf) const {
  cv::Mat img = src.clone();
  std::string text = std::to_string(gt) + ' ' + std::to_string(conf);
  cv::putText(img, text, cv::Point(30, 30), 1, 1, cv::Scalar(0, 0, 255), 2);

  return img;
}
#ifdef CPU_ONLY
STUB_GPU(ConfCheckLayer);
#endif

INSTANTIATE_CLASS(ConfCheckLayer);
REGISTER_LAYER_CLASS(ConfCheck);
} // namespace caffe