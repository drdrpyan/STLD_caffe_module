#include "check_layer.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//#include <fstream>

std::ofstream ofs;

namespace caffe
{

template <typename Dtype>
void CheckLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CheckParameter param = this->layer_param_.check_param();

  //dst_path_ = param.dst_path();
  bbox_norm_ = param.bbox_norm();
  if (bbox_norm_) {
    w_divider_ = param.w_divider();
    CHECK_GT(w_divider_, 0);
    h_divider_ = param.h_divider();
    CHECK_GT(h_divider_, 0);
  }

  cnt_ = 0;

  //ofs.open(dst_path_ + "/list.txt");
}

template <typename Dtype>
void CheckLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), 3);

  int num_batch = bottom[0]->num();

  //std::vector<int> gt_shape(4, 1);
  //gt_shape[0] = num_batch;

  //if (bottom.size() > 1) {
  //  CHECK(bottom[1]->shape() == gt_shape);
  //}
  //if (bottom.size() > 2) {
  //  gt_shape[1] = 4;
  //  CHECK(bottom[2]->shape() == gt_shape);
  //}

  if (bottom.size() > 1) {
    CHECK_EQ(bottom[1]->num(), bottom[0]->num());
    CHECK_EQ(bottom[1]->channels(), 1);
  }
  if (bottom.size() > 2) {
    CHECK_EQ(bottom[2]->num(), bottom[1]->num());
    CHECK_EQ(bottom[2]->channels(), 4);
    CHECK_EQ(bottom[2]->height(), bottom[1]->height());
  }
}

//template <typename Dtype>
//void CheckLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//                                    const vector<Blob<Dtype>*>& top) {
//  
//  std::vector<cv::Mat> imgs;
//  std::vector<int> labels;
//  std::vector<cv::Rect2f> bboxes;
//
//  DecodeImg(*(bottom[0]), &imgs);
  //if (bottom.size() > 1)
  //  DecodeLabel(*(bottom[1]), &labels);
  //if (bottom.size() > 2) {
  //  if (bbox_norm_)
  //    DecodeBBox(*(bottom[2]), imgs[0].cols, imgs[0].rows, &bboxes);
  //  else
  //    DecodeBBox(*(bottom[2]), &bboxes);
  //}

  //for (int i = 0; i < imgs.size(); ++i) {
  //  cv::Mat img = imgs[i];
  //  std::string out_name;

  //  //// draw activation region
  //  //cv::rectangle(img, cv::Rect(48, 48, 32, 32), cv::Scalar(255, 255, 255));

  //  if (bottom.size() == 1) {
  //    out_name = OutName(dst_path_);
  //  }
  //  if (bottom.size() == 2) {
  //    int label = labels[i];
  //    out_name = OutName(dst_path_, label);
  //    DrawLabel(label, img);
  //  }
  //  else if (bottom.size() == 3) {
  //    int label = labels[i];
  //    const cv::Rect2f& bbox = bboxes[i];
  //    out_name = OutName(dst_path_, label, bbox);
  //    DrawGT(label, bbox, img);
  //  }
  //  else
  //    LOG(FATAL) << "Illegal bottoms";

  //  //LOG(INFO) << "Write img : " << out_name;

  //  //int label = labels[i];
  //  //out_name = OutName(dst_path_, label);

  //  //ofs << out_name << ' ' << label << std::endl;

  //  //cv::imwrite(out_name, img);
  //}
//}

template <typename Dtype>
void CheckLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {

  std::vector<cv::Mat> imgs;
  std::vector<std::vector<int> > labels;
    std::vector<std::vector<cv::Rect_<Dtype> > > bboxes;

  DecodeImg(*(bottom[0]), &imgs);

  if (bottom.size() == 3) {
    DecodeLabelBBox(*(bottom[1]), *(bottom[2]),
                    &labels, &bboxes);
  }

  for (int i = 0; i < imgs.size(); ++i) {
    cv::Mat img = imgs[i];

    if (bottom.size() > 2) {
      for (int j = 0; j < bboxes[i].size(); ++j)
        cv::rectangle(img, bboxes[i][j], cv::Scalar(255, 255, 0));
    }
  }

}

template <typename Dtype>
void CheckLayer<Dtype>::DecodeImg(Blob<Dtype>& img_blob,
                                  std::vector<cv::Mat>* img_mat) const {
  CHECK(img_mat);
  
  const int BATCH_SIZE = img_blob.num();
  const int CHANNELS = img_blob.channels();
  const int WIDTH = img_blob.width();
  const int HEIGHT = img_blob.height();
  const int MAT_TYPE = CV_MAKETYPE((sizeof(Dtype) == sizeof(float)) ? CV_32F : CV_64F,
                                   1);

  img_mat->resize(BATCH_SIZE);
  Dtype* img_data = img_blob.mutable_cpu_data();
  for (int n = 0; n < BATCH_SIZE; ++n) {
    std::vector<cv::Mat> channels(CHANNELS);
    for (int c = 0; c < CHANNELS; ++c) {
      channels[c] = cv::Mat(cv::Size(WIDTH, HEIGHT), MAT_TYPE,
                            img_data + img_blob.offset(n, c));
    }

    cv::merge(channels, (*img_mat)[n]);
    if (CHANNELS == 1)
      (*img_mat)[n] = (*img_mat)[n].clone();
  }
}

//template <typename Dtype>
//void CheckLayer<Dtype>::DecodeLabel(const Blob<Dtype>& label_blob,
//                                    std::vector<int>* label_int) const {
//  CHECK(label_int);
//
//  const Dtype* label_data = label_blob.cpu_data();
//  label_int->assign(label_data, label_data + label_blob.count());
//}
//
//template <typename Dtype>
//void CheckLayer<Dtype>::DecodeBBox(
//    const Blob<Dtype>& bbox_blob,
//    std::vector<cv::Rect2f>* bbox_rect) const {
//  CHECK(bbox_rect);
//
//  const int BATCH_SIZE = bbox_blob.num();
//  bbox_rect->resize(BATCH_SIZE);
//  const Dtype* bbox_iter = bbox_blob.cpu_data();
//  
//  for (int n = 0; n < BATCH_SIZE; ++n) {
//    Dtype x = *bbox_iter++;
//    Dtype y = *bbox_iter++;
//    Dtype w = *bbox_iter++;
//    Dtype h = *bbox_iter++;
//
//    (*bbox_rect)[n] = cv::Rect2f(x, y, w, h);
//  }
//
//  //for (int n = 0; n < BATCH_SIZE; ++n) {
//  //  const Dtype* x_iter = bbox_blob.cpu_data() + bbox_blob.offset(n, 0);
//  //  const Dtype* y_iter = bbox_blob.cpu_data() + bbox_blob.offset(n, 1);
//  //  const Dtype* w_iter = bbox_blob.cpu_data() + bbox_blob.offset(n, 2);
//  //  const Dtype* h_iter = bbox_blob.cpu_data() + bbox_blob.offset(n, 3);
//
//  //  (*bbox_rect)[n] = cv::Rect2f(*x_iter++, *y_iter++,
//  //                               *w_iter++, *h_iter++);
//  //}
//}
//
//template <typename Dtype>
//void CheckLayer<Dtype>::DecodeBBox(
//    const Blob<Dtype>& bbox_blob,
//    int img_width, int img_height,
//    std::vector<cv::Rect2f>* bbox_rect) const {
//  CHECK_GT(img_width, 0);
//  CHECK_GT(img_height, 0);
//
//  DecodeBBox(bbox_blob, bbox_rect);
//
//  for (auto iter = bbox_rect->begin(); iter != bbox_rect->end(); ++iter) {
//    iter->width = std::exp(iter->width) * img_width;
//    iter->height = std::exp(iter->height) * img_height;
//
//    iter->x = (iter->x * img_width) - (iter->width / 2);
//    iter->y = (iter->y * img_height) - (iter->height / 2);
//  }
//}

template <typename Dtype>
void CheckLayer<Dtype>::DecodeLabelBBox(
    const Blob<Dtype>& label_blob,
    const Blob<Dtype>& bbox_blob,
    std::vector<std::vector<int> >* label,
    std::vector<std::vector<cv::Rect_<Dtype> > >* bbox) const {
  CHECK_EQ(label_blob.num(), bbox_blob.num());
  CHECK_EQ(label_blob.channels(), 1);
  CHECK_EQ(bbox_blob.channels(), 4);
  CHECK_EQ(label_blob.height(), bbox_blob.height());
  CHECK(label);
  CHECK(bbox);

  label->resize(label_blob.num());
  bbox->resize(bbox_blob.num());

  const Dtype* label_ptr = label_blob.cpu_data();
  const Dtype* bbox_ptr = bbox_blob.cpu_data();

  for (int n = 0; n < label_blob.num(); ++n) {
    const Dtype* label_iter = label_ptr + label_blob.offset(n);
    const Dtype* x_iter = bbox_ptr + bbox_blob.offset(n, 0);
    const Dtype* y_iter = bbox_ptr + bbox_blob.offset(n, 1);
    const Dtype* w_iter = bbox_ptr + bbox_blob.offset(n, 2);
    const Dtype* h_iter = bbox_ptr + bbox_blob.offset(n, 3);

    (*label)[n].clear();
    (*bbox)[n].clear();

    for (int i = 0; i < label_blob.height(); ++i) {
      int l = *label_iter;
      if ((l != LabelParameter::NONE) && (l != LabelParameter::DUMMY_LABEL)) {
        Dtype x = *x_iter;
        Dtype y = *y_iter;
        Dtype w = *w_iter;
        Dtype h = *h_iter;

        if (bbox_norm_) {
          w = std::exp(w) * w_divider_;
          h = std::exp(h) * h_divider_;
          x = (x * w_divider_) - (w / 2.);
          y = (y * h_divider_) - (h / 2.);
        }
        cv::Rect_<Dtype> b(x, y, w, h);

        (*label)[n].push_back(l);
        (*bbox)[n].push_back(b);
      }

      ++label_iter;
      ++x_iter;
      ++y_iter;
      ++w_iter;
      ++h_iter;
    }
  }
}


template <typename Dtype>
void CheckLayer<Dtype>::DrawLabel(int label, cv::Mat& dst) const {
  cv::putText(dst, std::to_string(label), cv::Point(10, 10), 1, 2,
              cv::Scalar(255, 0, 0));
}

template <typename Dtype>
void CheckLayer<Dtype>::DrawGT(int label, const cv::Rect2f& rect, 
                               cv::Mat& dst) const {
  if(label != LabelParameter::NONE && label != LabelParameter::DUMMY_LABEL)
    cv::rectangle(dst, rect, cv::Scalar(255, 0, 0));
  cv::putText(dst, std::to_string(label), cv::Point(10, 20), 1, 1,
              cv::Scalar(255, 0, 0));
}

template <typename Dtype>
std::string CheckLayer<Dtype>::OutName(const std::string& folder,
                                       int label, cv::Rect2f bbox) {
  //char buffer[256];
  //sprintf(buffer, "%s/%d_%d_%f_%f_%f_%f.png", folder, cnt_++, label, 
  //        bbox.x, bbox.y, bbox.width, bbox.height);
  std::string str = "" + folder + '/' + std::to_string(cnt_++) + '_' + std::to_string(label);

  //if (label != LabelParameter::NONE && label != LabelParameter::DUMMY_LABEL) {
  //  str += "_" + std::to_string(bbox.x) + '_' + std::to_string(bbox.y);
  //  str += "_" + std::to_string(bbox.width) + '_' + std::to_string(bbox.height);
  //}
  str += ".png";
  return str;
}

#ifdef CPU_ONLY
STUB_GPU(CheckLayer);
#endif

INSTANTIATE_CLASS(CheckLayer);
REGISTER_LAYER_CLASS(Check);
} // namespace caffe