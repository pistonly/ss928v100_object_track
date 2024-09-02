#include "utils.hpp"
#include "sample_comm.h"
#include "ss_mpi_sys.h"

void copy_yuv420_from_frame(char *yuv420, ot_video_frame_info *frame) {
  td_u32 height = frame->video_frame.height;
  td_u32 width = frame->video_frame.width;
  td_u32 size = height * width * 3 / 2; // 对于YUV420格式，大小为宽*高*1.5

  td_void *frame_data =
      ss_mpi_sys_mmap_cached(frame->video_frame.phys_addr[0], size);
  if (frame_data == NULL) {
    sample_print("mmap failed!\n");
    /* free(tmp); */
    return;
  }

  memcpy(yuv420, frame_data, size);
}
