#include "context_and_small_layer.hpp"

#include <opencv2/imgproc.hpp>

namespace caffe
{

template <typename Dtype>
void ContextAndSmallLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    CHECK_EQ(bottom[i]->num_axes(), 4);
    CHECK(bottom[i]->channels() == 1 ||
          bottom[i]->channels() == 3);
    CHECK_GT(bottom[i]->height(), small_area_.br().y);
    CHECK_GT(bottom[i]->width(), small_area_.br().x);

    std::vector<int> shape = bottom[i]->shape();
    shape[0] *= 2;
    top[i]->Reshape(shape);
  }
}

template <typename Dtype>
void ContextAndSmallLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    caffe::caffe_copy(bottom[i]->count(),
                      bottom[i]->cpu_data(),
                      top[i]->mutable_cpu_data());


    std::vector<cv::Mat> bottom_mat;
    BlobToCvMat(*(bottom[i]), &bottom_mat);

    std::vector<cv::Mat> small;
    ExtractSmall(bottom_mat, &small);

    if (bottom_mat[0].channels() == 3) {
      Dtype* top_data = top[i]->mutable_cpu_data();

      for (int j = 0; j < small.size(); ++j) {
        std::vector<cv::Mat> split_vec(3);
        for (int c = 0; c < 3; ++c) {
          Dtype* ch_dst = top_data + top[i]->offset(small.size() + j, c);
          split_vec[c] = cv::Mat(small[j].size(), CV_32FC1, ch_dst);
        }
        cv::split(small[j], split_vec);
      }
    }
    else
      LOG(FATAL) << "Not implemented yet.";
  }

  //// debug
  //Dtype* input_data = top[0]->mutable_cpu_data();
  //Dtype* small_data = input_data + top[0]->offset(30);
  //for (int i = 0; i < 30; ++i) {
  //  cv::Mat input_blue(cv::Size(127, 127), CV_32FC1, input_data);
  //  cv::Mat input_green(cv::Size(127, 127), CV_32FC1, input_data + (127*127));
  //  cv::Mat input_red(cv::Size(127, 127), CV_32FC1, input_data + 2*(127*127));
  //  input_data += 3 * 127 * 127;
  //  cv::Mat small_blue(cv::Size(127, 127), CV_32FC1, small_data);
  //  cv::Mat small_green(cv::Size(127, 127), CV_32FC1, small_data + (127*127));
  //  cv::Mat small_red(cv::Size(127, 127), CV_32FC1, small_data + 2*(127*127));
  //  small_data += 3 * 127 * 127;

  //  std::vector<cv::Mat> input_ch(3);
  //  input_ch[0] = input_blue;
  //  input_ch[1] = input_green;
  //  input_ch[2] = input_red;
  //  cv::Mat input_img;
  //  cv::merge(input_ch, input_img);
  //  input_img.convertTo(input_img, CV_8UC3);
  //  
  //  std::vector<cv::Mat> small_ch(3);
  //  small_ch[0] = small_blue;
  //  small_ch[1] = small_green;
  //  small_ch[2] = small_red;
  //  cv::Mat small_img;
  //  cv::merge(small_ch, small_img);
  //  small_img.convertTo(small_img, CV_8UC3);
  //}

}

template <typename Dtype>
void ContextAndSmallLayer<Dtype>::BlobToCvMat(
    Blob<Dtype>& bottom_blob,
    std::vector<cv::Mat>* mat) {
  CHECK(mat);
  
  mat->resize(bottom_blob.num());
  if (bottom_blob.channels() == 3) {
    Dtype* bottom_data = bottom_blob.mutable_cpu_data();
    cv::Size mat_size(bottom_blob.width(),
                      bottom_blob.height());

    for (int i = 0; i < mat->size(); ++i) {
      std::vector<cv::Mat> channels(3);
      for (int c = 0; c < 3; ++c) {
        Dtype* ch_ptr = bottom_data + bottom_blob.offset(i, c);
        channels[c] = cv::Mat(mat_size, CV_32FC1,
                              ch_ptr);
      }

      cv::merge(channels, (*mat)[i]);
    }
  }
  else
    LOG(FATAL) << "Not implemented yet.";
}

template <typename Dtype>
void ContextAndSmallLayer<Dtype>::ExtractSmall(
    const std::vector<cv::Mat>& bottom_mat,
    std::vector<cv::Mat>* small) {
  CHECK_GT(bottom_mat.size(), 0);
  CHECK(small);

  small->resize(bottom_mat.size());

  if (bottom_mat[0].channels() == 3) {
    for (int i = 0; i < bottom_mat.size(); ++i) {
      cv::Mat src_img;
      bottom_mat[i].convertTo(src_img, CV_8UC3);

      cv::Mat src_small = src_img(small_area_);
      cv::resize(src_small, src_small, 
                 bottom_mat[i].size());
      src_small.convertTo((*small)[i], CV_32FC3);
    }
  }
  else
    LOG(FATAL) << "Not implemented yet.";
}

#ifdef CPU_ONLY
STUB_GPU(ContextAndSmallLayer);
#endif

INSTANTIATE_CLASS(ContextAndSmallLayer);
REGISTER_LAYER_CLASS(ContextAndSmall);

} // namespace caffe