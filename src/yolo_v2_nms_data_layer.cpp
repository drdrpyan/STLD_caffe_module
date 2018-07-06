#include "yolo_v2_nms_data_layer.hpp"

#include "detection_util.hpp"

#include <opencv2/imgproc.hpp>

namespace caffe
{
template <typename Dtype>
void YOLOV2NMSDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  InitYOLOV2Handler(layer_param_.yolo_v2_nms_data_param());

  anno_decoder_.reset(new bgm::AnnoDecoder<Dtype>);

}

template <typename Dtype>
inline void YOLOV2NMSDataLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  std::vector<int> top_shape = bottom[0]->shape();
  top_shape[1] = yolo_v2_handler_->anchor().size();

  top[0]->Reshape(top_shape);
  
  if (bbox_shape_ != top_shape) {
    int mat_type;
    if (sizeof(float) == sizeof(Dtype))
      mat_type = CV_32FC1;
    else if (sizeof(double) == sizeof(Dtype))
      mat_type = CV_64FC1;
    else
      LOG(FATAL) << "Dtype must be float or double.";

    bbox_shape_ = top_shape;
    
    const int size_2d = top_shape[2] * top_shape[3];
    const int size_3d = top_shape[1] * size_2d;
    bbox_.resize(bbox_shape_[0],
                 std::vector<cv::Rect_<Dtype> >(size_3d));

    iou_.resize(bbox_shape_[0], std::vector<cv::Mat>(bbox_shape_[1]));
    for (int i = 0; i < iou_.size(); ++i)
      for (int j = 0; j < iou_[i].size(); ++j)
        iou_[i][j] = cv::Mat(bbox_shape_[2], bbox_shape_[3], mat_type);

    max_iou_.resize(bbox_shape_[0]);
    //for(int i=0; i<max_iou_.size(); ++i)
    //  max_iou_[i] = cv::Mat(bbox_shape_[2], bbox_shape_[3], mat_type);
  }
}


template <typename Dtype>
void YOLOV2NMSDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  std::vector<std::vector<int> > gt_label;
  std::vector<std::vector<cv::Rect_<Dtype> > > gt_bbox;
  DecodeGT(bottom, &gt_label, &gt_bbox);

  DecodeBBox(*(bottom[0]));
  CalcBestIOU(gt_bbox);
  CalcLocalMaxima(*(top[0]));
}

template <typename Dtype>
void YOLOV2NMSDataLayer<Dtype>::InitYOLOV2Handler(
    const YOLOV2NMSDataParameter& param) {
  cv::Size cell_size(param.cell_size().width(), param.cell_size().height());
  CHECK_GT(cell_size.width, 0);
  CHECK_GT(cell_size.height, 0);

  int num_class = param.num_class();
  CHECK_GT(num_class, 0);

  std::vector<cv::Rect_<Dtype> > anchor(param.anchor_size());
  for (int i = 0; i < param.anchor_size(); ++i) {
    const caffe::Rect2f& a = param.anchor(i);
    anchor[i] = cv::Rect_<Dtype>(a.top_left().x(), a.top_left().y(),
                                 a.size().width(), a.size().height());
  }

  yolo_v2_handler_.reset(new bgm::YOLOV2Handler<Dtype>(cell_size, num_class, 
                                                       anchor));
}

template <typename Dtype>
void YOLOV2NMSDataLayer<Dtype>::DecodeBBox(const Blob<Dtype>& yolo_v2_out) {
  const Dtype* yolo_v2_out_ptr = yolo_v2_out.cpu_data();
  //cv::Rect_<Dtype>* bbox_ptr = bbox_.mutable_cpu_data();

  for (int n = 0; n < bbox_shape_[0]; ++n) {
    auto bbox_iter = bbox_[n].begin();

    for (int a = 0; a < bbox_shape_[1]; ++a) {
      int x_ch =
          yolo_v2_handler_->GetAnchorChannel(a, bgm::YOLOV2Handler<Dtype>::AnchorChannel::X);
      int y_ch =
          yolo_v2_handler_->GetAnchorChannel(a, bgm::YOLOV2Handler<Dtype>::AnchorChannel::Y);
      int w_ch =
          yolo_v2_handler_->GetAnchorChannel(a, bgm::YOLOV2Handler<Dtype>::AnchorChannel::W);
      int h_ch =
          yolo_v2_handler_->GetAnchorChannel(a, bgm::YOLOV2Handler<Dtype>::AnchorChannel::H);

      const Dtype* x_iter = yolo_v2_out_ptr + yolo_v2_out.offset(n, x_ch);
      const Dtype* y_iter = yolo_v2_out_ptr + yolo_v2_out.offset(n, y_ch);
      const Dtype* w_iter = yolo_v2_out_ptr + yolo_v2_out.offset(n, w_ch);
      const Dtype* h_iter = yolo_v2_out_ptr + yolo_v2_out.offset(n, h_ch);

      cv::Rect_<Dtype> bbox_yolo_form;
      for (int h = 0; h < yolo_v2_out.height(); ++h) {
        for (int w = 0; w < yolo_v2_out.width(); ++w) {
          bbox_yolo_form.x = *x_iter++;
          bbox_yolo_form.y = *y_iter++;
          bbox_yolo_form.width = *w_iter++;
          bbox_yolo_form.height = *h_iter++;
  //        bbox_yolo_form.x = yolo_v2_out_ptr[yolo_v2_out.offset(n, x_ch, h, w)];
  //        bbox_yolo_form.y = yolo_v2_out_ptr[yolo_v2_out.offset(n, y_ch, h, w)];
  //        bbox_yolo_form.width = yolo_v2_out_ptr[yolo_v2_out.offset(n, w_ch, h, w)];
  //        bbox_yolo_form.height = yolo_v2_out_ptr[yolo_v2_out.offset(n, h_ch, h, w)];

  //        bbox_ptr[bbox_.offset(n, a, h, w)] = 
  //            yolo_v2_handler_->YOLOBoxToRawBox(bbox_yolo_form, h, w, a);
          *bbox_iter++ = yolo_v2_handler_->YOLOBoxToRawBox(bbox_yolo_form,
                                                           h, w, a);
        }
      }
    }
  }
}

template <typename Dtype>
void YOLOV2NMSDataLayer<Dtype>::CalcBestIOU(
    const std::vector<std::vector<cv::Rect_<Dtype> > >& gt_bbox) {
  CHECK_EQ(bbox_shape_[0], gt_bbox.size());

  for (int n = 0; n < bbox_shape_[0]; ++n) {
    auto bbox_iter = bbox_[n].cbegin();

    for (int a = 0; a < bbox_shape_[1]; ++a) {
#ifndef NDEBUG
      const cv::Mat& debug = iou_[n][a];
#endif // !NDEBUG

      Dtype* iou_iter = reinterpret_cast<Dtype*>(iou_[n][a].data);
      for (int i = bbox_shape_[2] * bbox_shape_[3]; i--; ) {
        float max_iou = 0;
        for (auto gt = gt_bbox[n].cbegin(); gt != gt_bbox[n].cend(); ++gt) {
          float iou = bgm::CalcIoU(*bbox_iter, *gt);
          if (max_iou < iou)
            max_iou = iou;
        }

        *iou_iter = max_iou;

        ++bbox_iter;
        ++iou_iter;
      }
    }
  }

  //CHECK(bbox_.shape() == iou_.shape());
  //CHECK_EQ(iou_.num(), gt_bbox.size());

  //const cv::Rect_<Dtype>* bbox_ptr = bbox_.cpu_data();
  //Dtype* iou_ptr = iou_.mutable_cpu_data();

  //for (int n = 0; n < iou_.num(); ++n) {
  //  const std::vector<cv::Rect_<Dtype> >& current_gt_bbox = gt_bbox[n];

  //  const cv::Rect_<Dtype>* bbox_iter = bbox_ptr + bbox_.offset(n);
  //  Dtype* iou_iter = iou_ptr + iou_.offset(n);

  //  for (int i = iou_.count(1); i--;) {
  //    const cv::Rect_<Dtype>& prediction = *bbox_iter;

  //    float max_iou = 0;
  //    for (int j = 0; j < current_gt_bbox.size(); ++j) {
  //      float iou = bgm::CalcIoU(prediction, current_gt_bbox[j]);

  //      if (iou > max_iou)
  //        max_iou = iou;
  //    }
  //    *iou_iter = max_iou;

  //    ++bbox_iter;
  //    ++iou_iter;
  //  }
  //}
}

template <typename Dtype>
void YOLOV2NMSDataLayer<Dtype>::CalcLocalMaxima(
    Blob<Dtype>& local_maxima_out) {
  CalcChannelwiseMaxIOU();

  Dtype* out_ptr = local_maxima_out.mutable_cpu_data();

  for (int n = 0; n < local_maxima_out.num(); ++n) {
    for (int a = 0; a < local_maxima_out.channels(); ++a) {
      Dtype* out_iter = out_ptr + local_maxima_out.offset(n, a);
      const Dtype* iou_iter = reinterpret_cast<Dtype*>(iou_[n][a].data);
      const Dtype* max_iou_iter = reinterpret_cast<Dtype*>(max_iou_[n].data);
      for (int i = local_maxima_out.count(2); i--; ) {
        *out_iter = ((*iou_iter > 0) && (*iou_iter >= *max_iou_iter)) ? 1 : 0;

        ++out_iter;
        ++iou_iter;
        ++max_iou_iter;
      }
    }
  }
  //const Dtype* iou_ptr = iou_.cpu_data();
  //const Dtype* max_iou_ptr = max_iou_.cpu_data();
  //Dtype* local_maxima_out_ptr = local_maxima_out.mutable_cpu_data();

  //for (int n = 0; n < iou_.num(); ++n) {
  //  for (int h = 0; h < iou_.height(); ++h) {
  //    int t = std::max(0, h - 1);
  //    int b = std::min(h + 1, static_cast<int>(iou_.height() - 1));
  //    for (int w = 0; w < iou_.width(); ++w) {
  //      int l = std::max(0, w - 1);
  //      int r = std::min(w + 1, static_cast<int>(iou_.width() - 1));

  //      //Dtype tl_val = max_iou_ptr[max_iou_.offset(n, 0, t, l)];
  //      //Dtype t_val = max_iou_ptr[max_iou_.offset(n, 0, t, w)];
  //      //Dtype tr_val = max_iou_ptr[max_iou_.offset(n, 0, t, r)];
  //      //Dtype l_val = max_iou_ptr[max_iou_.offset(n, 0, h, l)];
  //      //Dtype c_val = max_iou_ptr[max_iou_.offset(n, 0, h, w)];
  //      //Dtype r_val = max_iou_ptr[max_iou_.offset(n, 0, h, r)];
  //      //Dtype bl_val = max_iou_ptr[max_iou_.offset(n, 0, b, l)];
  //      //Dtype b_val = max_iou_ptr[max_iou_.offset(n, 0, b, w)];
  //      //Dtype br_val = max_iou_ptr[max_iou_.offset(n, 0, b, r)];

  //      for (int a = 0; a < iou_.channels(); ++a) {
  //        Dtype c_val = iou_ptr[iou_.offset(n, a, h, w)];
  //        bool local_max = (c_val > 0)
  //          && (c_val >= max_iou_ptr[max_iou_.offset(n, 0, t, l)])
  //          && (c_val >= max_iou_ptr[max_iou_.offset(n, 0, t, w)])
  //          && (c_val >= max_iou_ptr[max_iou_.offset(n, 0, t, r)])
  //          && (c_val >= max_iou_ptr[max_iou_.offset(n, 0, h, l)])
  //          && (c_val >= max_iou_ptr[max_iou_.offset(n, 0, h, w)])
  //          && (c_val >= max_iou_ptr[max_iou_.offset(n, 0, h, r)])
  //          && (c_val >= max_iou_ptr[max_iou_.offset(n, 0, b, l)])
  //          && (c_val >= max_iou_ptr[max_iou_.offset(n, 0, b, w)])
  //          && (c_val >= max_iou_ptr[max_iou_.offset(n, 0, b, r)]);
  //        local_maxima_out_ptr[local_maxima_out.offset(n, a, h, w)] 
  //            = local_max ? 1 : 0;
  //      }
  //    }
  //  }
  //}
}

template <typename Dtype>
void YOLOV2NMSDataLayer<Dtype>::CalcChannelwiseMaxIOU() {
  for (int n = 0; n < iou_.size(); ++n) {
    cv::Mat max_iou = iou_[n][0];
    
    for (int a = 1; a < iou_[n].size(); ++a)
      max_iou = cv::max(max_iou, iou_[n][a]);

    cv::dilate(max_iou, max_iou_[n], cv::Mat());
  }
  //const Dtype* iou_ptr = iou_.cpu_data();
  //Dtype* max_iou_ptr = max_iou_.mutable_cpu_data();

  //for (int n = 0; n < iou_.num(); ++n) {
  //  Dtype* max_iou_iter = max_iou_ptr + max_iou_.offset(n);

  //  caffe::caffe_copy(iou_.count(2),
  //                    iou_ptr + iou_.offset(n, 0), max_iou_iter);

  //  for (int c = 1; c < iou_.channels(); ++c) {
  //    const Dtype* iou_iter = iou_ptr + iou_.offset(n, c);
  //    for (int i = iou_.count(2); i--; ) {
  //      if (*max_iou_iter < *iou_iter)
  //        *max_iou_iter = *iou_iter;

  //      ++max_iou_iter;
  //      ++iou_iter;
  //    }
  //  }
  //}
}

#ifdef CPU_ONLY
STUB_GPU(YOLOV2NMSDataLayer);
#endif

INSTANTIATE_CLASS(YOLOV2NMSDataLayer);
REGISTER_LAYER_CLASS(YOLOV2NMSData);
} // namespace caffe