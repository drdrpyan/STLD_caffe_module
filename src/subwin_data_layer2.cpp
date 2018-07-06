#include "subwin_data_layer2.hpp"

namespace caffe
{
template <typename Dtype>
void SubwinData2Layer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  DataParameter* data_param = this->layer_param_.mutable_data_param();
  batch_size_ = data_param->batch_size();
  CHECK_GT(batch_size_, 0);
  data_param->set_batch_size(1);

  subwin_data_out_blobs_.resize(top.size());
  subwin_data_top_.resize(top.size());
  for (int i = 0; i < top.size(); ++i) {
    Blob<Dtype>* blob = new Blob<Dtype>;
    subwin_data_out_blobs_[i].reset(blob);
    subwin_data_top_[i] = blob;
  }

  SubwinDataLayer<Dtype>::LayerSetUp(bottom, subwin_data_top_);

  Fetch_cpu(bottom);

  for (int i = 0; i < top.size(); ++i) {
    std::vector<int> top_shape = subwin_data_top_[i]->shape();
    top_shape[0] = batch_size_;
    top[i]->Reshape(top_shape);
  }

  anno_encoder_.reset(new bgm::AnnoEncoder<Dtype>);
  anno_decoder_.reset(new bgm::AnnoDecoder<Dtype>);
}
template <typename Dtype>
void SubwinData2Layer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //DataParameter* data_param = this->layer_param_.mutable_data_param();
  //batch_size_ = data_param->batch_size();
  //CHECK_GT(batch_size_, 0);
  //data_param->set_batch_size(1);

  //subwin_data_out_blobs_.resize(top.size());
  //subwin_data_top_.resize(top.size());
  //for (int i = 0; i < top.size(); ++i) {
  //  Blob<Dtype>* blob = new Blob<Dtype>;
  //  subwin_data_out_blobs_[i].reset(blob);
  //  subwin_data_top_[i] = blob;
  //}

  SubwinDataLayer<Dtype>::DataLayerSetUp(bottom, subwin_data_top_);

  //Fetch_cpu(bottom);
}

//template <typename Dtype>
//void SubwinData2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
//                                      const vector<Blob<Dtype>*>& top) {
//  CHECK_EQ(subwin_data_top_.size(), top.size());
//
//  for (int i = 0; i < top.size(); ++i) {
//    std::vector<int> top_shape = subwin_data_top_[i]->shape();
//    top_shape[0] = batch_size_;
//    top[i]->Reshape(top_shape);
//  }
//}

template <typename Dtype>
void SubwinData2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  //std::vector<std::vector<int> > label, temp_label;
  //std::vector<std::vector<cv::Rect_<Dtype> > > bbox, temp_bbox;
  //
  //int subwin_batch_size = subwin_data_top_[0]->num();

  //std::vector<Blob<Dtype>*> label_bbox(2);
  //label_bbox[0] = subwin_data_top_[1];
  //label_bbox[1] = subwin_data_top_[2];
  //temp_label.clear();
  //temp_bbox.clear();
  //anno_decoder_->Decode(label_bbox, &temp_label, &temp_bbox);

  //int top_cursor = 0;
  //while (top_cursor < batch_size_) {
  //  if (subwin_data_cursor_ == subwin_batch_size) {
  //    Fetch_cpu(bottom);
  //    subwin_batch_size = subwin_data_top_[0]->num();

  //    std::vector<Blob<Dtype>*> label_bbox(2);
  //    label_bbox[0] = subwin_data_top_[1];
  //    label_bbox[1] = subwin_data_top_[2];
  //    temp_label.clear();
  //    temp_bbox.clear();
  //    anno_decoder_->Decode(label_bbox, &temp_label, &temp_bbox);
  //  }

  //  int num_copy = std::min(batch_size_ - top_cursor,
  //                          subwin_batch_size - subwin_data_cursor_);

  //  // forward data
  //  const Dtype* src = subwin_data_top_[0]->cpu_data();
  //  src += subwin_data_top_[0]->offset(subwin_data_cursor_);

  //  Dtype* dst = top[0]->mutable_cpu_data();
  //  dst += top[0]->offset(top_cursor);

  //  int length = num_copy * subwin_data_top_[0]->count(1);

  //  caffe_copy<Dtype>(length, src, dst);

  //  // forward label, bbox
  //  label.insert(label.end(),
  //               temp_label.begin() + subwin_data_cursor_,
  //               temp_label.begin() + subwin_data_cursor_ + num_copy);
  //  bbox.insert(bbox.end(),
  //              temp_bbox.begin() + subwin_data_cursor_,
  //              temp_bbox.begin() + subwin_data_cursor_ + num_copy);

  //  //for (int i = 0; i < top.size(); ++i) {
  //  //  const Dtype* src = subwin_data_top_[i]->cpu_data();
  //  //  src += subwin_data_top_[i]->offset(subwin_data_cursor_);

  //  //  Dtype* dst = top[i]->mutable_cpu_data();
  //  //  dst += top[i]->offset(top_cursor);

  //  //  int length = num_copy * subwin_data_top_[i]->count(1);

  //  //  caffe_copy<Dtype>(length, src, dst);
  //  //}

  //  top_cursor += num_copy;
  //  subwin_data_cursor_ += num_copy;
  //}

  //ForwardAnno(label, bbox, top[1], top[2]);

  if (subwin_data_cursor_ == 4) {
    subwin_data_cursor_ = 0;
    Fetch_cpu(bottom);
  }

  for (int i = 0; i < top.size(); ++i) {
    std::vector<int> top_shape = subwin_data_top_[i]->shape();
    top_shape[0] = 3;

    const Dtype* src = subwin_data_top_[i]->cpu_data();
    src += subwin_data_top_[i]->offset(subwin_data_cursor_*3);
    Dtype* dst = top[i]->mutable_cpu_data();
    int length = 3 * subwin_data_top_[i]->count(1);
    caffe_copy<Dtype>(length, src, dst);
  }

  subwin_data_cursor_++;
}

template <typename Dtype>
void SubwinData2Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  if (subwin_data_cursor_ == 4) {
    subwin_data_cursor_ = 0;
    Fetch_gpu(bottom);
  }

  for (int i = 0; i < top.size(); ++i) {
    std::vector<int> top_shape = subwin_data_top_[i]->shape();
    top_shape[0] = 3; // 여기
    //top_shape[0] = 2;
    top[i]->Reshape(top_shape);

    const Dtype* src = subwin_data_top_[i]->gpu_data();
    src += subwin_data_top_[i]->offset(subwin_data_cursor_*3);
    Dtype* dst = top[i]->mutable_gpu_data();
    int length = 3 * subwin_data_top_[i]->count(1);
    //int length = 2 * subwin_data_top_[i]->count(1); // 여기
    caffe_copy<Dtype>(length, src, dst);
  }

  subwin_data_cursor_++;

  //int subwin_batch_size = subwin_data_top_[0]->num();

  //int top_cursor = 0;
  //while (top_cursor < batch_size_) {
  //  if (subwin_data_cursor_ == subwin_batch_size) {
  //    Fetch_gpu(bottom);
  //    subwin_batch_size = subwin_data_top_[0]->num();
  //  }

  //  int num_copy = std::min(batch_size_ - top_cursor,
  //                          subwin_batch_size - subwin_data_cursor_);

  //  for (int i = 0; i < top.size(); ++i) {
  //    const Dtype* src = subwin_data_top_[i]->gpu_data();
  //    src += subwin_data_top_[i]->offset(subwin_data_cursor_);

  //    Dtype* dst = top[i]->mutable_gpu_data();
  //    dst += top[i]->offset(top_cursor);

  //    int length = num_copy * subwin_data_top_[i]->count(1);

  //    caffe_copy(length, src, dst);
  //  }

  //  top_cursor += num_copy;
  //  subwin_data_cursor_ += num_copy;
  //}
}

template <typename Dtype>
void SubwinData2Layer<Dtype>::Fetch_cpu(const vector<Blob<Dtype>*>& bottom) {
  subwin_data_cursor_ = 0;

  SubwinDataLayer<Dtype>::Forward_cpu(bottom, subwin_data_top_);

  CHECK_GT(subwin_data_top_.size(), 0);
  int subwin_batch_size = subwin_data_top_[0]->num();
  for (int i = 1; i < subwin_data_top_.size(); ++i) {
    CHECK_EQ(subwin_data_top_[i]->num(), subwin_batch_size);
  }
}

template <typename Dtype>
void SubwinData2Layer<Dtype>::Fetch_gpu(const vector<Blob<Dtype>*>& bottom) {
  subwin_data_cursor_ = 0;

  SubwinDataLayer<Dtype>::Forward_gpu(bottom, subwin_data_top_);

  CHECK_GT(subwin_data_top_.size(), 0);
  int subwin_batch_size = subwin_data_top_[0]->num();
  for (int i = 1; i < subwin_data_top_.size(); ++i) {
    CHECK_EQ(subwin_data_top_[i]->num(), subwin_batch_size);
  }
}

template <typename Dtype>
void SubwinData2Layer<Dtype>::ForwardAnno(
    const std::vector<std::vector<int> >& label,
    const std::vector<std::vector<cv::Rect_<Dtype> > >& bbox,
    Blob<Dtype>* label_blob, Blob<Dtype>* bbox_blob) {
  CHECK_EQ(label.size(), bbox.size());
  CHECK(label_blob);
  CHECK(bbox_blob);

  int num_anno = 0;
  for (int i = 0; i < label.size(); ++i) {
    CHECK_EQ(label[i].size(), bbox[i].size());
    if (num_anno < label[i].size())
      num_anno = label[i].size();
    //num_anno = std::max(num_anno, label[i].size());
  }

  std::vector<int> top_shape(3);
  top_shape[0] = batch_size_;
  top_shape[1] = 1;
  top_shape[2] = num_anno;
  label_blob->Reshape(top_shape);

  top_shape[1] = 4;
  bbox_blob->Reshape(top_shape);

  std::vector<Blob<Dtype>*> encode_dst(2);
  encode_dst[0] = label_blob;
  encode_dst[1] = bbox_blob;

  anno_encoder_->Encode(label, bbox, encode_dst);
}

#ifdef CPU_ONLY
STUB_GPU(SubwinData2Layer);
#endif

INSTANTIATE_CLASS(SubwinData2Layer);
REGISTER_LAYER_CLASS(SubwinData2);
} // namespace caffe