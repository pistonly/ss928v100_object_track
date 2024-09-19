/**
 * @file      ffmpeg_vdec_vpss.hpp<7-one_camera_track>
 * @brief     Header of
 * @date      Tue Sep  3 12:02:02 2024
 * @author    liuyang
 * @copyright BSD-3-Clause
 * 
 * This module
 */

#include "ot_common_svp.h"
#include "ot_common_vdec.h"
#include "ot_common_video.h"
#include "ot_defines.h"
#include "ot_type.h"
#include "sample_comm.h"
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
  /**
   * @brief Summary
   * @details Description
   * @param[in] rtsp_url: url
   * @param[in] step_mode: if true, decoder is blocked unless frame is used;
   */
  HardwareDecoder(const std::string &rtsp_url, bool step_mode = false);
  ~HardwareDecoder();

  /**
   * @brief Start hardware decoding
   * @details Description
   */
  void start_decode();
  bool get_frame_without_release();
  bool release_frames();

  /**
   * @brief Summary
   * @details Description
   * @param[inout] img_H Description
   * @param[out] dst_L Description
   * @return Description
   */
  bool get_frames(void *img_H, ot_svp_dst_img *dst_L);
  ot_video_frame_info frame_H, frame_L;
  bool is_ffmpeg_exit() { return ffmpeg_exit; }

private:
  void decode_thread();
  void decode_thread_step();
  bool initialize_ffmpeg();
  bool initialize_vdec();

  bool mb_step_mode;
  bool mb_decode_step_on;
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
