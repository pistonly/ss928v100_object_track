#include "ot_common_svp.h"
#include "ot_common_vdec.h"
#include "ot_common_video.h"
#include "ot_defines.h"
#include "ot_type.h"
#include "sample_comm.h"
// #include "sample_common_ive.h"
#include "ss_mpi_vpss.h"
#include <atomic>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
}


class HardwareDecoder {
public:
  HardwareDecoder(const std::string &rtsp_url);
  ~HardwareDecoder();
  void start_decode();
  bool get_frame_without_release();
  bool release_frames();
  bool get_frames(void *img_H, ot_svp_dst_img *dst_L);
  ot_video_frame_info frame_H, frame_L;
  bool is_ffmpeg_exit(){
    return ffmpeg_exit;
  }

private:
  void decode_thread();
  bool initialize_ffmpeg();
  bool initialize_vdec();

  std::string rtsp_url_;
  AVFormatContext *fmt_ctx_;
  int video_stream_index_;
  sample_vdec_attr sample_vdec_[OT_VDEC_MAX_CHN_NUM];
  std::thread decode_thread_;
  std::atomic<bool> decoding_;
  ot_vdec_chn_status vdec_status_;
  ot_vpss_grp vpss_grp;
  int frame_id = 0;
  bool ffmpeg_exit = false;
};
