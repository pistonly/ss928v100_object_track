#include "ffmpeg_vdec_vpss.hpp"
#include "ot_common_vdec.h"
#include "ot_common_video.h"
#include "ot_defines.h"
#include "ot_type.h"
#include "sample_comm.h"
#include "sample_common_ive.h"
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

#define UHD_STREAM_WIDTH 3840
#define UHD_STREAM_HEIGHT 2160
#define FHD_STREAM_WIDTH 1920
#define FHD_STREAM_HEIGHT 1080
#define REF_NUM 2
#define DISPLAY_NUM 2
#define SAMPLE_VDEC_COMM_VB_CNT 4
#define SAMPLE_VDEC_VPSS_LOW_DELAY_LINE_CNT 16

static ot_payload_type g_cur_type = OT_PT_H265;
// static ot_payload_type g_cur_type = OT_PT_H264;

static vdec_display_cfg g_vdec_display_cfg = {
    .pic_size = PIC_3840X2160,
    .intf_sync = OT_VO_OUT_3840x2160_30,
    .intf_type = OT_VO_INTF_HDMI,
};

static ot_size g_disp_size;
static td_s32 g_sample_exit = 0;

static td_void copy_save_frame(ot_video_frame_info *frame, td_u32 frame_id) {
  td_u32 height = frame->video_frame.height;
  td_u32 width = frame->video_frame.width;
  td_u32 size = height * width * 3 / 2; // 对于YUV420格式，大小为宽*高*1.5
  /* td_void *tmp = malloc(size); */
  /* if (tmp == NULL) { */
  /*     sample_print("malloc failed!\n"); */
  /*     return; */
  /* } */

  td_void *yuv = ss_mpi_sys_mmap_cached(frame->video_frame.phys_addr[0], size);
  if (yuv == NULL) {
    sample_print("mmap failed!\n");
    /* free(tmp); */
    return;
  }

  /* memcpy(tmp, yuv, size); */

  // 生成文件名
  char file_name[128];
  snprintf(file_name, sizeof(file_name), "./frame%ux%u_%u.yuv", width, height,
           frame_id);

  // 打开文件
  FILE *file = fopen(file_name, "wb");
  if (file == NULL) {
    sample_print("fopen failed!\n");
    ss_mpi_sys_munmap(yuv, size);
    /* free(tmp); */
    return;
  }

  // 写入文件
  /* fwrite(tmp, 1, size, file); */
  fwrite(yuv, 1, size, file);
  fclose(file);

  // 释放资源
  ss_mpi_sys_munmap(yuv, size);
  /* free(tmp); */

  sample_print("Frame %u saved as %s\n", frame_id, file_name);
}

static td_u32 sample_vdec_get_dimension(bool is_width) {
  if (g_cur_type == OT_PT_H264 || g_cur_type == OT_PT_H265 ||
      g_cur_type == OT_PT_JPEG || g_cur_type == OT_PT_MJPEG) {
    return is_width ? UHD_STREAM_WIDTH : UHD_STREAM_HEIGHT;
  }
  sample_print("Invalid type %d!\n", g_cur_type);
  return is_width ? UHD_STREAM_WIDTH : UHD_STREAM_HEIGHT;
}

static td_s32 sample_start_vdec(sample_vdec_attr *sample_vdec,
                                td_u32 vdec_chn_num, td_u32 len) {
  td_s32 ret = sample_comm_vdec_start(vdec_chn_num, sample_vdec, len);
  if (ret != TD_SUCCESS) {
    sample_print("Start VDEC failed for %#x!\n", ret);
    sample_comm_vdec_stop(vdec_chn_num);
  }
  return ret;
}

static td_s32 sample_init_module_vb(sample_vdec_attr *sample_vdec,
                                    td_u32 vdec_chn_num, ot_payload_type type,
                                    td_u32 len) {
  for (td_u32 i = 0; i < vdec_chn_num && i < len; i++) {
    sample_vdec[i].type = type;
    sample_vdec[i].width = sample_vdec_get_dimension(true);
    sample_vdec[i].height = sample_vdec_get_dimension(false);
    sample_vdec[i].mode = sample_comm_vdec_get_lowdelay_en()
                              ? OT_VDEC_SEND_MODE_COMPAT
                              : OT_VDEC_SEND_MODE_FRAME;
    sample_vdec[i].sample_vdec_video.dec_mode = OT_VIDEO_DEC_MODE_IP;
    sample_vdec[i].sample_vdec_video.bit_width = OT_DATA_BIT_WIDTH_8;
    sample_vdec[i].sample_vdec_video.ref_frame_num = REF_NUM;
    sample_vdec[i].display_frame_num = DISPLAY_NUM;
    sample_vdec[i].frame_buf_cnt =
        (type == OT_PT_JPEG) ? (sample_vdec[i].display_frame_num + 1)
                             : (sample_vdec[i].sample_vdec_video.ref_frame_num +
                                sample_vdec[i].display_frame_num + 1);
    if (type == OT_PT_JPEG) {
      sample_vdec[i].sample_vdec_picture.pixel_format =
          OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420;
      sample_vdec[i].sample_vdec_picture.alpha = 255; // Alpha value
    }
  }
  td_s32 ret = sample_comm_vdec_init_vb_pool(vdec_chn_num, sample_vdec, len);
  if (ret != TD_SUCCESS) {
    sample_print("Init module VB failed for %#x!\n", ret);
    return ret;
  }
  return TD_SUCCESS;
}

static td_s32 sample_init_sys_and_vb(sample_vdec_attr *sample_vdec,
                                     td_u32 vdec_chn_num, ot_payload_type type,
                                     td_u32 len) {
  ot_vb_cfg vb_cfg;
  ot_pic_buf_attr buf_attr = {0};
  td_s32 ret;

  sample_comm_vdec_set_lowdelay_en(TD_TRUE);

  ret = sample_comm_sys_get_pic_size(g_vdec_display_cfg.pic_size, &g_disp_size);
  if (ret != TD_SUCCESS) {
    sample_print("System get picture size failed for %#x!\n", ret);
    return ret;
  }
  buf_attr.align = OT_DEFAULT_ALIGN;
  buf_attr.bit_width = OT_DATA_BIT_WIDTH_8;
  buf_attr.compress_mode = OT_COMPRESS_MODE_SEG;
  buf_attr.height = g_disp_size.height;
  buf_attr.width = g_disp_size.width;
  buf_attr.pixel_format = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420;

  memset_s(&vb_cfg, sizeof(vb_cfg), 0, sizeof(vb_cfg));
  vb_cfg.max_pool_cnt = 1;
  vb_cfg.common_pool[0].blk_cnt = SAMPLE_VDEC_COMM_VB_CNT * vdec_chn_num;
  vb_cfg.common_pool[0].blk_size = ot_common_get_pic_buf_size(&buf_attr);
  ret = sample_comm_sys_init(&vb_cfg);
  if (ret != TD_SUCCESS) {
    sample_print("System init failed for %#x!\n", ret);
    sample_comm_sys_exit();
    return ret;
  }
  ret = sample_init_module_vb(sample_vdec, vdec_chn_num, type, len);
  if (ret != TD_SUCCESS) {
    sample_print("Module VB init failed for %#x!\n", ret);
    sample_comm_vdec_exit_vb_pool();
    sample_comm_sys_exit();
    return ret;
  }
  return TD_SUCCESS;
}

static td_void sample_stop_vpss(ot_vpss_grp vpss_grp, td_bool *vpss_chn_enable,
                                td_u32 chn_array_size) {
  td_s32 i;
  for (i = vpss_grp; i >= 0; i--) {
    vpss_grp = i;
    sample_common_vpss_stop(vpss_grp, &vpss_chn_enable[0], chn_array_size);
  }
}

static td_s32 sample_vdec_bind_vpss(td_u32 vpss_grp_num) {
  td_u32 i;
  td_s32 ret = TD_SUCCESS;
  for (i = 0; i < vpss_grp_num; i++) {
    ret = sample_comm_vdec_bind_vpss(i, i);
    if (ret != TD_SUCCESS) {
      sample_print("vdec bind vpss fail for %#x!\n", ret);
      return ret;
    }
  }
  return ret;
}

static td_s32 sample_vdec_unbind_vpss(td_u32 vpss_grp_num) {
  td_u32 i;
  td_s32 ret = TD_SUCCESS;
  for (i = 0; i < vpss_grp_num; i++) {
    ret = sample_comm_vdec_un_bind_vpss(i, i);
    if (ret != TD_SUCCESS) {
      sample_print("vdec unbind vpss fail for %#x!\n", ret);
    }
  }
  return ret;
}

static td_void sample_config_vpss_grp_attr(ot_vpss_grp_attr *vpss_grp_attr) {
  vpss_grp_attr->max_width = sample_vdec_get_dimension(true);
  vpss_grp_attr->max_height = sample_vdec_get_dimension(false);
  vpss_grp_attr->frame_rate.src_frame_rate = -1;
  vpss_grp_attr->frame_rate.dst_frame_rate = -1;
  vpss_grp_attr->pixel_format = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420;
  vpss_grp_attr->nr_en = TD_FALSE;
  vpss_grp_attr->ie_en = TD_FALSE;
  vpss_grp_attr->dci_en = TD_FALSE;
  vpss_grp_attr->dei_mode = OT_VPSS_DEI_MODE_OFF;
  vpss_grp_attr->buf_share_en = TD_FALSE;
}

static td_s32 sample_config_vpss_ldy_attr(td_u32 vpss_grp_num) {
  td_u32 i;
  td_s32 ret;
  ot_low_delay_info vpss_ldy_info;
  if (!sample_comm_vdec_get_lowdelay_en()) {
    return TD_SUCCESS;
  }
  for (i = 0; i < vpss_grp_num; i++) {
    ret = ss_mpi_vpss_get_low_delay_attr(i, 0, &vpss_ldy_info);
    if (ret != TD_SUCCESS) {
      sample_print("vpss get low delay attr fail for %#x!\n", ret);
      return ret;
    }
    vpss_ldy_info.enable = TD_TRUE;
    vpss_ldy_info.line_cnt = SAMPLE_VDEC_VPSS_LOW_DELAY_LINE_CNT;
    ret = ss_mpi_vpss_set_low_delay_attr(i, 0, &vpss_ldy_info);
    if (ret != TD_SUCCESS) {
      sample_print("vpss set low delay attr fail for %#x!\n", ret);
      return ret;
    }
  }
  return TD_SUCCESS;
}

static td_s32 sample_start_vpss(ot_vpss_grp *vpss_grp, td_u32 vpss_grp_num,
                                td_bool *vpss_chn_enable, td_u32 arr_len) {
  td_u32 i;
  td_s32 ret;
  ot_vpss_chn_attr vpss_chn_attr[OT_VPSS_MAX_CHN_NUM];
  ot_vpss_grp_attr vpss_grp_attr;
  (td_void) memset_s(&vpss_grp_attr, sizeof(ot_vpss_grp_attr), 0,
                     sizeof(ot_vpss_grp_attr));
  sample_comm_vpss_get_default_grp_attr(&vpss_grp_attr);
  vpss_grp_attr.max_width = g_disp_size.width;
  vpss_grp_attr.max_height = g_disp_size.height;

  (td_void) memset_s(vpss_chn_enable, arr_len * sizeof(td_bool), 0,
                     arr_len * sizeof(td_bool));

  vpss_chn_enable[0] = TD_TRUE;
  vpss_chn_attr[0].width = g_disp_size.width;   /* 4:crop */
  vpss_chn_attr[0].height = g_disp_size.height; /* 4:crop */
  vpss_chn_attr[0].depth = 1;
  vpss_chn_attr[1].compress_mode = OT_COMPRESS_MODE_NONE;
  vpss_chn_attr[0].chn_mode = OT_VPSS_CHN_MODE_USER;
  // vpss_chn_attr[0].pixel_format = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420;
  vpss_chn_attr[0].pixel_format = OT_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
  vpss_chn_attr[0].frame_rate.src_frame_rate = -1;
  vpss_chn_attr[0].frame_rate.dst_frame_rate = -1;

  vpss_chn_enable[1] = TD_TRUE;
  (td_void) memset_s(&vpss_chn_attr[1], sizeof(ot_vpss_chn_attr), 0,
                     sizeof(ot_vpss_chn_attr));
  sample_comm_vpss_get_default_chn_attr(&vpss_chn_attr[1]);
  vpss_chn_attr[1].width = g_disp_size.width / 2;   /* 4:crop */
  vpss_chn_attr[1].height = g_disp_size.height / 2; /* 4:crop */
  vpss_chn_attr[1].compress_mode = OT_COMPRESS_MODE_NONE;
  vpss_chn_attr[1].chn_mode = OT_VPSS_CHN_MODE_USER;
  vpss_chn_attr[1].pixel_format = OT_PIXEL_FORMAT_YUV_400;
  // vpss_chn_attr[1].pixel_format = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420;
  vpss_chn_attr[1].depth = 1;

  for (i = 0; i < vpss_grp_num; i++) {
    *vpss_grp = i;
    ret =
        sample_common_vpss_start(*vpss_grp, &vpss_chn_enable[0], &vpss_grp_attr,
                                 vpss_chn_attr, OT_VPSS_MAX_CHN_NUM);
    if (ret != TD_SUCCESS) {
      sample_print("start VPSS fail for %#x!\n", ret);
      sample_stop_vpss(*vpss_grp, &vpss_chn_enable[0], OT_VPSS_MAX_CHN_NUM);
      return ret;
    }
  }

  ret = sample_config_vpss_ldy_attr(vpss_grp_num);
  if (ret != TD_SUCCESS) {
    sample_stop_vpss(*vpss_grp, &vpss_chn_enable[0], OT_VPSS_MAX_CHN_NUM);
    return ret;
  }

  ret = sample_vdec_bind_vpss(vpss_grp_num);
  if (ret != TD_SUCCESS) {
    sample_vdec_unbind_vpss(vpss_grp_num);
    sample_stop_vpss(*vpss_grp, &vpss_chn_enable[0], OT_VPSS_MAX_CHN_NUM);
  }
  return ret;
}

HardwareDecoder::HardwareDecoder(const std::string &rtsp_url, bool step_mode)
    : mb_step_mode(step_mode), mb_decode_step_on(true), rtsp_url_(rtsp_url),
      fmt_ctx_(nullptr), video_stream_index_(-1), decoding_(false) {
  avformat_network_init();
  if (!initialize_ffmpeg() || !initialize_vdec()) {
    throw std::runtime_error("Initialization failed");
  }
}

HardwareDecoder::~HardwareDecoder() {
  decoding_ = false;
  if (decode_thread_.joinable()) {
    decode_thread_.join();
  }
  sample_comm_vdec_stop(1);
  sample_comm_vdec_exit_vb_pool();
  sample_comm_sys_exit();
  avformat_close_input(&fmt_ctx_);
  avformat_network_deinit();
}

bool HardwareDecoder::initialize_ffmpeg() {
  AVDictionary *options = nullptr;
  av_dict_set(&options, "rtsp_transport", "tcp", 0);
  av_dict_set(&options, "stimeout", "5000000", 0);
  av_dict_set(&options, "buffer_size", "1024000", 0);

  if (avformat_open_input(&fmt_ctx_, rtsp_url_.c_str(), nullptr, &options) !=
      0) {
    std::cerr << "Could not open source" << std::endl;
    av_dict_free(&options);
    return false;
  }
  av_dict_free(&options);

  if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
    std::cerr << "Could not find stream information" << std::endl;
    return false;
  }

  for (int i = 0; i < fmt_ctx_->nb_streams; i++) {
    if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream_index_ = i;
      break;
    }
  }

  if (video_stream_index_ == -1) {
    std::cerr << "Could not find video stream" << std::endl;
    return false;
  }

  return true;
}

bool HardwareDecoder::initialize_vdec() {
  td_s32 ret;
  td_u32 vdec_chn_num = 1;
  td_u32 vpss_grp_num = vdec_chn_num;
  td_bool vpss_chn_enable[OT_VPSS_MAX_CHN_NUM];
  sample_vo_cfg vo_config;

  ret = sample_init_sys_and_vb(sample_vdec_, vdec_chn_num, g_cur_type,
                               OT_VDEC_MAX_CHN_NUM);
  if (ret != TD_SUCCESS) {
    return false;
  }

  ret = sample_start_vdec(sample_vdec_, vdec_chn_num, OT_VDEC_MAX_CHN_NUM);
  if (ret != TD_SUCCESS) {
    return false;
  }

  ret = sample_start_vpss(&vpss_grp, vpss_grp_num, &vpss_chn_enable[0],
                          OT_VPSS_MAX_CHN_NUM);
  if (ret != TD_SUCCESS) {
    return false;
  }

  return true;
}

void HardwareDecoder::start_decode() {
  decoding_ = true;
  if (mb_step_mode)
    decode_thread_ = std::thread(&HardwareDecoder::decode_thread_step, this);
  else
    decode_thread_ = std::thread(&HardwareDecoder::decode_thread, this);
}

void HardwareDecoder::decode_thread_step() {
  AVPacket packet;
  ot_vdec_stream stream;
  int packet_num = 0;

  while (decoding_) {
    if (mb_step_mode) {
      if (!mb_decode_step_on) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // waiting
        continue;
      } else {
        if (packet_num > 0) {
          // TODO: lock for thread safe
          mb_decode_step_on = false;
        }
      }
    } else {
      std::cerr << "mb_step_mode should be ture" << std::endl;
    }
    if (av_read_frame(fmt_ctx_, &packet) < 0)
      continue;

    if (packet.stream_index == video_stream_index_) {
      stream.addr = packet.data;
      stream.len = packet.size;
      stream.pts = packet.pts;
      stream.need_display = TD_TRUE;
      stream.end_of_frame = TD_TRUE;
      stream.end_of_stream = TD_FALSE;

      td_s32 ret = ss_mpi_vdec_send_stream(0, &stream, -1);
      if (ret != TD_SUCCESS) {
        std::cerr << "Error sending stream to decoder for " << std::hex << ret
                  << std::endl;
        break;
      }
      packet_num++;
      // if (packet_num > 2) {
      //   mb_decode_step_on = false;
      // }
      std::cout << "red frame number: " << packet_num << std::endl;
    }
    av_packet_unref(&packet);

    td_s32 ret = ss_mpi_vdec_query_status(0, &vdec_status_);
    if (ret != TD_SUCCESS) {
      std::cerr << "Error querying VDEC status!" << std::endl;
    }
  }
  g_sample_exit = 1;
}

void HardwareDecoder::decode_thread() {
  AVPacket packet;
  ot_vdec_stream stream;
  int packet_num = 0;

  while (decoding_ && av_read_frame(fmt_ctx_, &packet) >= 0) {
    if (mb_step_mode) {
      std::cerr << "mb_step_mode should be false" << std::endl;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(33)); // waiting
    }

    if (packet.stream_index == video_stream_index_) {
      stream.addr = packet.data;
      stream.len = packet.size;
      stream.pts = packet.pts;
      stream.need_display = TD_TRUE;
      stream.end_of_frame = TD_TRUE;
      stream.end_of_stream = TD_FALSE;

      td_s32 ret = ss_mpi_vdec_send_stream(0, &stream, -1);
      if (ret != TD_SUCCESS) {
        std::cerr << "Error sending stream to decoder for " << std::hex << ret
                  << std::endl;
        break;
      }
      packet_num++;
      std::cout << "red frame number: " << packet_num << std::endl;
    }
    av_packet_unref(&packet);

    td_s32 ret = ss_mpi_vdec_query_status(0, &vdec_status_);
    if (ret != TD_SUCCESS) {
      std::cerr << "Error querying VDEC status!" << std::endl;
    }
  }
  g_sample_exit = 1;
}

bool HardwareDecoder::get_frames(void *img_H, ot_svp_dst_img *dst_L) {
  td_s32 ret = ss_mpi_vpss_get_chn_frame(vpss_grp, 0, &frame_H, 100);
  if (ret != TD_SUCCESS) {
    sample_print("get chn frame-0 failed for Err(%#x)\n", ret);
    return false;
  }
  // copy to frame
  td_u32 height = frame_H.video_frame.height;
  td_u32 width = frame_H.video_frame.width;
  td_u32 size = height * width * 3 / 2; // 对于YUV420格式，大小为宽*高*1.5

  td_void *yuv = ss_mpi_sys_mmap_cached(frame_H.video_frame.phys_addr[0], size);
  if (yuv == NULL) {
    sample_print("mmap failed!\n");
    return false;
  }

  memcpy(img_H, yuv, size);

  // -------------------- frame_L --------------------
  ret = ss_mpi_vpss_get_chn_frame(vpss_grp, 1, &frame_L, 100);
  if (ret != TD_SUCCESS) {
    sample_print("get chn frame-1 failed for Err(%#x)\n", ret);
    ret = ss_mpi_vdec_query_status(0, &vdec_status_);
    if (ret != TD_SUCCESS) {
      std::cerr << "Error querying VDEC status!" << std::endl;
    } else {
      std::cout << "INFO: \n"
                << " type: " << vdec_status_.type
                << ", left bytes: " << vdec_status_.left_stream_bytes
                << ", left frames: " << vdec_status_.left_stream_frames
                << ", left decoded_frames: " << vdec_status_.left_decoded_frames
                << ", is_started: " << vdec_status_.is_started
                << ", recv_stream_frames: " << vdec_status_.recv_stream_frames
                << ", dec_stream_frames: " << vdec_status_.dec_stream_frames
                << ", dec_w: " << vdec_status_.width
                << ", dec_h: " << vdec_status_.height << std::endl;

      std::cout
          << "VDEC status error: \n"
          << " set_pic_size_err: " << vdec_status_.dec_err.set_pic_size_err
          << ", set_protocol_num_err: "
          << vdec_status_.dec_err.set_protocol_num_err
          << ", set_ref_num_err: " << vdec_status_.dec_err.set_ref_num_err
          << ", set_pic_buf_size_err: "
          << vdec_status_.dec_err.set_pic_buf_size_err
          << ", format_err: " << vdec_status_.dec_err.format_err
          << ", stream_unsupport: " << vdec_status_.dec_err.stream_unsupport
          << ", pack_err: " << vdec_status_.dec_err.pack_err
          << ", stream_size_over: " << vdec_status_.dec_err.stream_size_over
          << ", stream not release: " << vdec_status_.dec_err.stream_not_release
          << std::endl;
    }
    return false;
  }
  // dma frame to ive image
  td_bool is_instant = TD_TRUE;
  ret = sample_common_ive_dma_image(&frame_L, dst_L, is_instant);
  if (ret != TD_SUCCESS) {
    sample_print("sample_ive_dma_image failed, Err(%#x)\n", ret);
    return false;
  }

  ret = release_frames();
  if (ret != TD_SUCCESS)
    return false;
  else {
    if (mb_step_mode) {
      // TODO: LOCK for thread safe
      mb_decode_step_on = true;
    }
    return true;
  }
}

bool HardwareDecoder::get_frame_without_release() {
  // NOTE: need release frames
  td_s32 ret = ss_mpi_vdec_query_status(0, &vdec_status_);
  if (ret != TD_SUCCESS) {
    std::cerr << "Error querying VDEC status!" << std::endl;
    return false;
  }

  ret = ss_mpi_vpss_get_chn_frame(vpss_grp, 0, &frame_H, 20000);
  if (ret != TD_SUCCESS) {
    sample_print("get chn frame failed for Err(%#x)\n", ret);
    return false;
  }

  ret = ss_mpi_vpss_get_chn_frame(vpss_grp, 1, &frame_L, 100);

  if (ret != TD_SUCCESS) {
    sample_print("get chn-1 frame failed for Err(%#x)\n", ret);
    std::cout << "INFO: \n"
              << " type: " << vdec_status_.type
              << ", left bytes: " << vdec_status_.left_stream_bytes
              << ", left frames: " << vdec_status_.left_stream_frames
              << ", left decoded_frames: " << vdec_status_.left_decoded_frames
              << ", is_started: " << vdec_status_.is_started
              << ", recv_stream_frames: " << vdec_status_.recv_stream_frames
              << ", dec_stream_frames: " << vdec_status_.dec_stream_frames
              << ", dec_w: " << vdec_status_.width
              << ", dec_h: " << vdec_status_.height << std::endl;

    std::cout << "VDEC status error: \n"
              << " set_pic_size_err: " << vdec_status_.dec_err.set_pic_size_err
              << ", set_protocol_num_err: "
              << vdec_status_.dec_err.set_protocol_num_err
              << ", set_ref_num_err: " << vdec_status_.dec_err.set_ref_num_err
              << ", set_pic_buf_size_err: "
              << vdec_status_.dec_err.set_pic_buf_size_err
              << ", format_err: " << vdec_status_.dec_err.format_err
              << ", stream_unsupport: " << vdec_status_.dec_err.stream_unsupport
              << ", pack_err: " << vdec_status_.dec_err.pack_err
              << ", stream_size_over: " << vdec_status_.dec_err.stream_size_over
              << ", stream not release: "
              << vdec_status_.dec_err.stream_not_release << std::endl;
    return false;
  } else {
    std::cout << "Received frame-1 with width: " << frame_L.video_frame.width
              << std::endl;
  }
  if (mb_step_mode) {
    std::cout << "set mb_decode_step_on" << std::endl;
    mb_decode_step_on = true;
  }

  return true;
}

// std::atomic<bool> running(true);

// void signal_handler(int signum) { running = false; }

// int main(int argc, char *argv[]) {
//   if (argc < 2) {
//     std::cerr << "Usage: " << argv[0] << " <RTSP URL>" << std::endl;
//     return -1;
//   }

//   std::string rtsp_url = argv[1];
//   signal(SIGINT, signal_handler); // capture Ctrl+C

//   try {
//     HardwareDecoder decoder(rtsp_url);
//     decoder.start_decode();

//     ot_video_frame_info frame;
//     while (running && g_sample_exit == 0) {
//       if (decoder.get_frame(frame)) {
//         // procesing
//         std::cout << "Received frame of width: " << frame.video_frame.width
//                   << std::endl;

//       } else {
//         std::this_thread::sleep_for(std::chrono::milliseconds(100)); //
//         waiting
//       }
//     }

//     std::cout << "Exiting gracefully..." << std::endl;
//   } catch (const std::exception &e) {
//     std::cerr << "Error: " << e.what() << std::endl;
//     return -1;
//   }

//   return 0;
// }
bool HardwareDecoder::release_frames() {
  td_s32 ret = ss_mpi_vpss_release_chn_frame(vpss_grp, 0, &frame_H);
  if (ret != TD_SUCCESS) {
    sample_print("vpss release chn frame-0 Err(%#x)\n", ret);
    return false;
  }

  ret = ss_mpi_vpss_release_chn_frame(vpss_grp, 1, &frame_L);
  if (ret != TD_SUCCESS) {
    sample_print("vpss release chn frame-1 Err(%#x)\n", ret);
    return false;
  }
}
