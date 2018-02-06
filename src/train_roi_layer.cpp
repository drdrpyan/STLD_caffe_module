#include "train_roi_layer.hpp"

namespace caffe
{

template <typename Dtype>
void TrainROILayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  TrainROIParameter param = this->layer_param_.train_roi_param();

  img_size_.width = param.img_width();
  img_size_.height = param.img_height();
  CHECK_GT(img_size_.width, 0);
  CHECK_GT(img_size_.height, 0);

  roi_size_.width = param.roi_width();
  roi_size_.height = param.roi_height();
  CHECK_GT(roi_size_.width, 0);
  CHECK_GT(roi_size_.height, 0);

  anno_decoder_.reset(new bgm::AnnoDecoder<Dtype>);
  roi_generator_.reset(new bgm::TrainROINearGridGenerator<Dtype>(
                               img_size_.width, img_size_.height, 7, 7));
  roi_extractor_.reset(new bgm::ROIAlignExtractor<Dtype>);
}

template <typename Dtype>
void TrainROILayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[1]->num(), bottom[2]->num());
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[2]->channels(), 4);
  CHECK_EQ(bottom[1]->count(2), bottom[2]->count(2));

  std::vector<int> top_shape(4);
  top_shape[0] = bottom[0]->num() * 10;
  top_shape[1] = bottom[0]->channels();
  top_shape[2] = roi_size_.height;
  top_shape[3] = roi_size_.width;
  top[0]->Reshape(top_shape);

  top_shape[1] = 1;
  top_shape[2] = 1;
  top_shape[3] = 1;
  top[1]->Reshape(top_shape);

  top_shape[1] = 4;
  top[2]->Reshape(top_shape);
}

template <typename Dtype>
void TrainROILayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::vector<std::vector<int> > gt_label;
  std::vector <std::vector<cv::Rect_<Dtype> > > gt_bbox;
  std::vector<Blob<Dtype>*> anno_blobs(2);
  anno_blobs[0] = bottom[1];
  anno_blobs[1] = bottom[2];
  anno_decoder_->Decode(anno_blobs, &gt_label, &gt_bbox);

  std::vector<std::vector<cv::Rect_<Dtype> > > roi;
  std::vector<std::vector<int> > roi_label;
  std::vector<std::vector<cv::Rect_<Dtype> > > roi_bbox;
  roi_generator_->Generate(gt_label, gt_bbox, &roi, &roi_label, &roi_bbox);
  
  int num_roi = NumROI(roi);

  std::vector<int> top_shape = top[0]->shape();
  top_shape[0] = num_roi;
  top[0]->Reshape(top_shape);

  roi_extractor_->Extract(*(bottom[0]), img_size_, roi, top[0]);

  MakeLabelTop(num_roi, roi_label, top[1]);
  MakeBBoxTop(num_roi, roi_bbox, top[2]);
}

template <typename Dtype>
void TrainROILayer<Dtype>::MakeLabelTop(
    int num_roi,
    const std::vector<std::vector<int> >& roi_label,
    Blob<Dtype>* label_top) const {
  CHECK_GT(num_roi, 0);
  CHECK(label_top);

  std::vector<int> top_shape = label_top->shape();
  top_shape[0] = num_roi;
  label_top->Reshape(top_shape);

  Dtype* top_iter = label_top->mutable_cpu_data();
  for (auto i = roi_label.cbegin(); i != roi_label.cend(); ++i)
    for (auto j = i->cbegin(); j != i->cend(); ++j)
      *top_iter++ = *j;
}

template <typename Dtype>
void TrainROILayer<Dtype>::MakeBBoxTop(
    int num_roi,
    const std::vector<std::vector<cv::Rect_<Dtype> > >& roi_bbox,
    Blob<Dtype>* bbox_top) const {
  CHECK_GT(num_roi, 0);
  CHECK(bbox_top);

  std::vector<int> top_shape = bbox_top->shape();
  top_shape[0] = num_roi;
  bbox_top->Reshape(top_shape);

  Dtype* top_iter = bbox_top->mutable_cpu_data();
  for (auto i = roi_bbox.cbegin(); i != roi_bbox.cend(); ++i) {
    for (auto j = i->cbegin(); j != i->cend(); ++j) {
      *top_iter++ = j->x;
      *top_iter++ = j->y;
      *top_iter++ = j->width;
      *top_iter++ = j->height;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(TrainROILayer);
#endif

INSTANTIATE_CLASS(TrainROILayer);
REGISTER_LAYER_CLASS(TrainROI);
} // namespace caffe