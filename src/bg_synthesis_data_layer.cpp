#include "bg_synthesis_data_layer.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>

//#include <fstream>
//#include <opencv2/highgui.hpp>
//std::string img_path = "f:/bosch_db/bosch_synth";
//std::string list_name = "f:/bosch_db/bosch_synth/list.txt";
//int img_cnt = 0;
//std::ofstream ofs;

namespace caffe
{

template <typename Dtype>
void BGSynthesisDataLayer<Dtype>::DataLayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const BGSynthesisDataParameter& param = layer_param_.bg_synthesis_data_param();

  width_ = param.width();
  CHECK_GT(width_, 0);
  height_ = param.height();
  CHECK_GT(height_, 0);

  //LayerParameter obj_data_param;
  //obj_data_param.set_phase(layer_param_.phase());
  //obj_data_param.set_allocated_bg_synthesis_data_param(&(param.obj_data_param()));
  //obj_data_layer_->LayerSetUp(param.obj_data_param());
  //obj_data_layer_->LayerSetUp(obj_data_param);
  obj_data_layer_.reset(new ImageDataLayer<Dtype>(param.obj_data_param()));
  obj_data_out_.resize(2);
  obj_data_out_[0] = &obj_data_;
  obj_data_out_[1] = &obj_label_;
  obj_data_layer_->LayerSetUp(bottom, obj_data_out_);
  //obj_data_layer_->DataLayerSetUp(bottom, obj_data_out_);

  bg_data_layer_.reset(new ImageDataLayer<Dtype>(param.bg_data_param()));
  //bg_data_layer_->LayerSetUp(param.bg_data_param());
  bg_data_out_.resize(2);
  bg_data_out_[0] = &bg_data_;
  bg_data_out_[1] = &bg_label_;
  bg_data_layer_->LayerSetUp(bottom, bg_data_out_);
  //bg_data_layer_->DataLayerSetUp(bottom, bg_data_out_);

  max_obj_ = param.max_obj();
  if (param.has_min_obj())
    min_obj_ = param.min_obj();
  else
    min_obj_ = max_obj_;

  if (param.has_active_region_param()) {
    auto region = param.active_region_param().region();
    activation_region_.x = region.top_left().x();
    activation_region_.y = region.top_left().y();
    activation_region_.width = region.size().width();
    activation_region_.height = region.size().height();

    activation_method_ = param.active_region_param().method();
  }
  else {
    activation_region_ = cv::Rect(0, 0, width_, height_);
    activation_method_ = ActivationRegionParameter::WHOLE;
  }

  batch_size_ = param.batch_size();

  num_neg_ = param.num_neg();
  CHECK_LE(num_neg_, batch_size_);

  //uniform_dist_.reset(new std::uniform_int_distribution<int>(0, 4096));
  //random_generator_ = std::bind(*uniform_dist_, random_engine_);

  resize_prob_ = 0.7;
  resize_min_ = 0.8;
  resize_max_ = 1.2;
  gaussian_noise_prob_ = 0.4;

  //ofs.open(list_name);
}

template <typename Dtype>
void BGSynthesisDataLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  std::vector<int> data_shape(4);
  data_shape[0] = batch_size_;
  data_shape[1] = 3;
  data_shape[2] = height_;
  data_shape[3] = width_;
  top[0]->Reshape(data_shape);

  if (top.size() > 1) {
    std::vector<int> gt_shape(4);
    gt_shape[0] = batch_size_;
    gt_shape[1] = 5;
    gt_shape[2] = 1;
    gt_shape[3] = 1;
    top[1]->Reshape(gt_shape);
  }
}

template <typename Dtype>
void BGSynthesisDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  cv::Mat bg_src = PopBG();
  std::vector<cv::Mat> bg;
  if (TryBernoulli(resize_prob_))
    bg_src = RandomResize(bg_src, resize_min_, resize_max_);
  CropBG(bg_src, batch_size_, &bg);

  std::vector<int> num_obj;
  GetUniformRandomInteger(batch_size_ - num_neg_,
                   min_obj_, max_obj_, &num_obj);
  
  int num_nonneg = batch_size_ - num_neg_;
  
  std::vector<std::vector<int> > labels(num_nonneg);
  std::vector<std::vector<cv::Rect> > bboxes(num_nonneg);
  
  for (int i = 0; i < num_nonneg; ++i) {
    std::vector<cv::Mat> temp_obj_data(num_obj[i]);
    std::vector<int> temp_label(num_obj[i]);
    std::vector<cv::Rect> temp_bbox(num_obj[i]);
    for (int j = 0; j < num_obj[i]; ++j) {
      auto obj = PopObj();
      temp_obj_data[j] = obj.first;

      if (TryBernoulli(resize_prob_)) {
        temp_obj_data[j] = RandomResize(temp_obj_data[j],
                                        resize_min_,
                                        resize_max_);
      }

      temp_label[j] = obj.second;
    }
    Synthesis(temp_obj_data, &(bg[i]), &temp_bbox);

    labels[i] = temp_label;
    bboxes[i] = temp_bbox;
  }

  for (int i = 0; i < bg.size(); ++i) {
    if (TryBernoulli(gaussian_noise_prob_)) {
      bg[i] = ApplyGaussianNoise(bg[i]);
    }

    //// debug
    //cv::Mat debug = bg[i];
    //std::string img_name = img_path + '/' + std::to_string(img_cnt++) + ".png";
    //cv::imwrite(img_name, bg[i]);
    //if (i < num_nonneg) {
    //  ofs << std::to_string(img_cnt) << ".png " << labels[i][0] << ' ';
    //  ofs << bboxes[i][0].x << ' ';
    //  ofs << bboxes[i][0].y << ' ';
    //  ofs << bboxes[i][0].width << ' ';
    //  ofs << bboxes[i][0].height << ' ';
    //  ofs << std::endl;
    //}
    //else
    //  ofs << std::to_string(img_cnt) << ".png " << LabelParameter::NONE << std::endl;
  }

  MakeTop(bg, num_nonneg, labels, bboxes, top);
}

template <typename Dtype>
void BGSynthesisDataLayer<Dtype>::LoadObj(int num) {
  std::vector<Blob<Dtype>*> dummy_bottom;

  for (int n = num; n--; ) {
    obj_data_layer_->Forward(dummy_bottom, obj_data_out_);

    //if (obj_data_out_[0]->width() < 10)
    //  continue;

    std::vector<cv::Mat> obj_data;
    BlobToMat(*(obj_data_out_[0]), &obj_data);

    const Dtype* obj_label_ptr = obj_data_out_[1]->cpu_data();
    std::vector<int> obj_label(obj_label_ptr,
                               obj_label_ptr + obj_data_out_[1]->count());

    //CHECK_EQ(obj_data.size(), obj_label.size());
    for (int i = 0; i < obj_data.size(); ++i)
      obj_.push_back(std::make_pair(obj_data[i], obj_label[i]));
  }
}

template <typename Dtype>
void BGSynthesisDataLayer<Dtype>::LoadBG(int num) {
  std::vector<Blob<Dtype>*> dummy_bottom;

  for (int n = num; n--;) {
    bg_data_layer_->Forward(dummy_bottom, bg_data_out_);

    std::vector<cv::Mat> bg_data;
    BlobToMat(*(bg_data_out_[0]), &bg_data);

    bg_.insert(bg_.end(), bg_data.begin(), bg_data.end());
  }
}

template <typename Dtype>
void BGSynthesisDataLayer<Dtype>::CropBG(
    const cv::Mat& src, int num, 
    std::vector<cv::Mat>* bg) {
  CHECK(!src.empty());
  CHECK_GT(num, 0);
  CHECK(bg);

  int x_range = src.cols - width_ + 1;
  int y_range = src.rows - height_ + 1;
  int num_candidate = x_range * y_range;

  if (x_range < 0 || y_range < 0) {
    cv::Mat next_bg = PopBG();
    CropBG(next_bg, num, bg);
  }
  else if (num_candidate < num) {
    CropBG(src, num_candidate, bg);

    cv::Mat next_bg = PopBG();
    std::vector<cv::Mat> remain_result;
    CropBG(next_bg, num - num_candidate, &remain_result);
    
    bg->insert(bg->end(),
               remain_result.begin(),
               remain_result.end());
  }
  else {
    std::vector<int> x, y;
    GetUniformRandomInteger(num, 0, x_range - 1, &x);
    GetUniformRandomInteger(num, 0, y_range - 1, &y);

    bg->resize(num);
    for (int i = 0; i < num; ++i)
      (*bg)[i] = src(cv::Rect(x[i], y[i], 
                              width_, height_)).clone();
  }
}

template <typename Dtype>
void BGSynthesisDataLayer<Dtype>::Synthesis(
    const std::vector<cv::Mat>& obj, 
    cv::Mat* bg, std::vector<cv::Rect>* location) const {
  CHECK(!obj.empty());
  CHECK(bg);
  CHECK(location);

  location->clear();

  if (activation_method_ == ActivationRegionParameter::CENTER) {
    for (int i = 0; i < obj.size(); ++i) {
      int x_min = activation_region_.x - obj[i].cols / 2;
      int y_min = activation_region_.y - obj[i].rows / 2;
      int x_max = activation_region_.br().x - obj[i].cols / 2;
      int y_max = activation_region_.br().y - obj[i].rows / 2;

      std::vector<int> x, y;
      GetUniformRandomInteger(NUM_SYNTH_TRY, x_min, x_max, &x);
      GetUniformRandomInteger(NUM_SYNTH_TRY, y_min, y_max, &y);

      for (int j = 0; j < NUM_SYNTH_TRY; ++j) {
        cv::Rect temp_rect(x[j], y[j], 
                           obj[i].cols, obj[i].rows);
        cv::Rect obj_roi, bg_roi;
        GetSynthROI(temp_rect, &obj_roi, &bg_roi);

        if (!Overlap(*location, bg_roi)) {
          obj[i](obj_roi).copyTo((*bg)(bg_roi));
          location->push_back(bg_roi);
          break;
        }
      }
    }
  }
  else
    LOG(FATAL) << "Not implemented yet";
}

template <typename Dtype>
void BGSynthesisDataLayer<Dtype>::GetSynthROI(
    const cv::Rect& roi, cv::Rect* obj_roi,
    cv::Rect* bg_roi) const {
  CHECK(obj_roi);
  CHECK(bg_roi);

  *obj_roi = cv::Rect(0, 0, roi.width, roi.height);
  *bg_roi = roi;

  if (roi.x < 0) {
    obj_roi->x += (-roi.x);
    obj_roi->width -= (-roi.x);

    bg_roi->x = 0;
    bg_roi->width -= (-roi.x);    
  }
  if (roi.br().x >= width_) {
    obj_roi->width -= (roi.br().x - width_ + 1);
    bg_roi->width -= (roi.br().x - width_ + 1);
  }
  if (roi.y < 0) {
    obj_roi->y += (-roi.y);
    obj_roi->height -= (-roi.y);

    bg_roi->y = 0;
    bg_roi->height -= (-roi.y);
  }
  if (roi.br().y >= height_) {
    obj_roi->height -= (roi.br().y - height_ + 1);
    bg_roi->height -= (roi.br().y - height_ + 1);
  }
}

template <typename Dtype>
void BGSynthesisDataLayer<Dtype>::MakeTop(
    const std::vector<cv::Mat>& data,
    int num_nonneg,
    const std::vector<std::vector<int> >& label,
    const std::vector<std::vector<cv::Rect> >& bbox,
    const std::vector<Blob<Dtype>*>& top) {
  MatToBlob(data, top[0]->mutable_cpu_data());

  if (top.size() > 1) {
    Dtype* gt_iter = top[1]->mutable_cpu_data();
    for (int n = 0; n < num_nonneg; ++n) {
      for (int i = 0; i < max_obj_; ++i) {
        if (i < label[n].size()) {
          *gt_iter++ = label[n][i];
          *gt_iter++ = (bbox[n][i].x + (bbox[n][i].width / 2.0f)) / (float)width_;
          *gt_iter++ = (bbox[n][i].y + (bbox[n][i].height / 2.0f)) / (float)height_;
          *gt_iter++ = std::log(bbox[n][i].width / (float)width_);
          *gt_iter++ = std::log(bbox[n][i].height / (float)height_);
          //*gt_iter++ = (bbox[n][i].x + (bbox[n][i].width / 2.0f));
          //*gt_iter++ = (bbox[n][i].y + (bbox[n][i].height / 2.0f));
          //*gt_iter++ = bbox[n][i].width;
          //*gt_iter++ = bbox[n][i].height;
        }
        else {
          *gt_iter++ = LabelParameter::NONE;
          *gt_iter++ = BBoxParameter::DUMMY_VALUE;
          *gt_iter++ = BBoxParameter::DUMMY_VALUE;
          *gt_iter++ = BBoxParameter::DUMMY_VALUE;
          *gt_iter++ = BBoxParameter::DUMMY_VALUE;
        }
      }
    }
    for (int n = num_nonneg; n < batch_size_; ++n) {
      *gt_iter++ = LabelParameter::NONE;
      *gt_iter++ = BBoxParameter::DUMMY_VALUE;
      *gt_iter++ = BBoxParameter::DUMMY_VALUE;
      *gt_iter++ = BBoxParameter::DUMMY_VALUE;
      *gt_iter++ = BBoxParameter::DUMMY_VALUE;
    }
  }
}

template <typename Dtype>
bool BGSynthesisDataLayer<Dtype>::Overlap(
    const std::vector<cv::Rect>& prev_roi,
    const cv::Rect& new_roi) const {
  for (auto roi = prev_roi.begin();
       roi != prev_roi.end(); ++roi) {
    if (new_roi.x < roi->br().x &&
        new_roi.br().x > roi->x &&
        new_roi.y < roi->br().y &&
        new_roi.br().y > roi->y)
      return true;
  }
  return false;
}

template <typename Dtype>
void BGSynthesisDataLayer<Dtype>::MatToBlob(
    const std::vector<cv::Mat>& mat,
    Dtype* blob_data) const {
  CHECK(!mat.empty());
  CHECK(blob_data);

  int width = mat[0].cols;
  int height = mat[0].rows;
  int mat_type = mat[0].type();

  Dtype* data_iter = blob_data;
  for (auto m = mat.begin(); m != mat.end(); ++m) {
    CHECK_EQ(width, m->cols);
    CHECK_EQ(height, m->rows);
    CHECK_EQ(mat_type, m->type());
    //CHECK_EQ(sizeof(Dtype), m->depth());

    std::vector<cv::Mat> split_vec(m->channels());

    for (int c = 0; c < m->channels(); ++c) {
      if (sizeof(Dtype) == sizeof(float))
        split_vec[c] = cv::Mat(m->size(), CV_32FC1,
                               data_iter);
      else
        LOG(FATAL) << "Not implemented yet";

      data_iter += (m->rows * m->cols);
    }

    cv::split(*m, split_vec);
  }
}

template <typename Dtype>
void BGSynthesisDataLayer<Dtype>::BlobToMat(
    const Blob<Dtype>& blob,
    std::vector<cv::Mat>* mat) const {
  CHECK(mat);
  CHECK(blob.channels() == 1 || blob.channels() == 3);

  mat->resize(blob.num());

  int step = blob.width() * blob.height();
  Dtype* data_iter = (const_cast<Blob<Dtype>&>(blob)).mutable_cpu_data();
  
  for (int n = 0; n < blob.num(); ++n) {
    std::vector<cv::Mat> channels;
    channels.reserve(blob.channels());

    for (int c = blob.channels(); c--;) {
      channels.push_back(
        cv::Mat(cv::Size(blob.width(), blob.height()),
                (sizeof(Dtype) == sizeof(float)) ? CV_32FC1 : CV_64FC1,
                data_iter));
      data_iter += step;
    }

    cv::merge(channels, (*mat)[n]);
  }
}

template <typename Dtype>
void BGSynthesisDataLayer<Dtype>::GetUniformRandomInteger(
    int num, int min, int max, 
    std::vector<int>* random) const {
  CHECK_GT(num, 0);
  CHECK_GE(max, min);
  CHECK(random);

  std::uniform_int_distribution<int> dist(min, max);
  ////auto random_generator = std::bind(dist, random_engine_);
  auto random_generator = std::bind(
      dist, 
      std::default_random_engine(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count()));

  random->resize(num);
  for (auto iter = random->begin(); iter != random->end(); ++iter) {
    //*iter = dist(random_engine_);
    *iter = random_generator();
    //*iter = (random_generator_() % (max - min)) + min;
  }
}

template <typename Dtype>
void BGSynthesisDataLayer<Dtype>::GetUniformRandomReal(
    int num, float min, float max, std::vector<float>* random) const {
  CHECK_GT(num, 0);
  CHECK_GE(max, min);
  CHECK(random);

  std::uniform_real_distribution<float> dist(min, max);
  ////auto random_generator = std::bind(dist, random_engine_);
  auto random_generator = std::bind(
      dist, 
      std::default_random_engine(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count()));

  random->resize(num);
  for (auto iter = random->begin(); iter != random->end(); ++iter) {
    //*iter = dist(random_engine_);
    *iter = random_generator();
    //*iter = (random_generator_() % (max - min)) + min;
  }
}

template <typename Dtype>
bool BGSynthesisDataLayer<Dtype>::TryBernoulli(float p) {
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);

  if (random_prob_.empty()) {
    std::vector<float> random_prob;
    GetUniformRandomReal(128, 0.0f, 1.0f, &random_prob);
    random_prob_.insert(random_prob_.end(),
                        random_prob.begin(), random_prob.end());
  }

  float prob = random_prob_.front();
  random_prob_.pop_front();

  return prob <= p;
}

template <typename Dtype>
cv::Mat BGSynthesisDataLayer<Dtype>::RandomResize(
    const cv::Mat& src,
    float scale_min, float scale_max,
    int scale_precision) const {
  CHECK(!src.empty());
  CHECK_LE(scale_min, scale_max);
  CHECK_GE(scale_precision, 0);

  float e = std::pow(10.0f, scale_precision);

  int s_min = std::round(scale_min * e);
  int s_max = std::round(scale_max * e);
  std::vector<int> scale;
  GetUniformRandomInteger(2, s_min, s_max, &scale);

  float w_scale = scale[0] / e;
  float h_scale = scale[1] / e;
  cv::Size size(std::round(src.cols * w_scale),
                std::round(src.rows * h_scale));

  cv::Mat resized;
  cv::resize(src, resized, size);

  return resized;
}

template <typename Dtype>
cv::Mat BGSynthesisDataLayer<Dtype>::ApplyGaussianNoise(
    const cv::Mat src) const {
  cv::Mat noise(src.size(), src.type());

  cv::randn(noise, 0, 10);

  cv::Mat noised = src.clone();
  noised += noise;

  //cv::normalize(noised, noised, 0.0, 1.0);
  if(sizeof(Dtype) == sizeof(float))
    normalize(noised, noised, 0.0, 1.0, CV_MINMAX, CV_32F);
  else
    normalize(noised, noised, 0.0, 1.0, CV_MINMAX, CV_64F);

  noised *= 255;

  return noised;
}

//template <typename Dtype>
//cv::Mat BGSynthesisDataLayer<Dtype>::ApplyPepperSaltNoise(
//    const cv::Mat src) const {
// 
//}

template <typename Dtype>
std::pair<cv::Mat, int> BGSynthesisDataLayer<Dtype>::PopObj() {
  while (obj_.empty())
    LoadObj();

  std::pair<cv::Mat, int> front = obj_.front();
  obj_.pop_front();
  return front;
}

template <typename Dtype>
cv::Mat BGSynthesisDataLayer<Dtype>::PopBG() {
  while (bg_.empty())
    LoadBG();

  cv::Mat front = bg_.front();
  bg_.pop_front();
  return front;
}

#ifdef CPU_ONLY
STUB_GPU(BGSynthesisDataLayer);
#endif

INSTANTIATE_CLASS(BGSynthesisDataLayer);
REGISTER_LAYER_CLASS(BGSynthesisData);

} // namespace caffe