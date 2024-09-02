/*
  Copyright (c), 2001-2022, Shenshu Tech. Co., Ltd.
 */
#ifndef SAMPLE_COMMON_SVP_H
#define SAMPLE_COMMON_SVP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include "ss_mpi_sys.h"
#include "ot_common.h"
#include "ot_common_svp.h"
#include "sample_comm.h"
#include "ss_mpi_dsp.h"

#ifdef __cplusplus
#if __cplusplus
extern "C"{
#endif
#endif /* __cplusplus */

#define OT_SVP_RECT_NUM                         64
#define OT_POINT_NUM                            4

typedef enum {
    SAMPLE_SVP_ERR_LEVEL_DEBUG   = 0x0,    /* debug-level                                  */
    SAMPLE_SVP_ERR_LEVEL_INFO    = 0x1,    /* informational                                */
    SAMPLE_SVP_ERR_LEVEL_NOTICE  = 0x2,    /* normal but significant condition             */
    SAMPLE_SVP_ERR_LEVEL_WARNING = 0x3,    /* warning conditions                           */
    SAMPLE_SVP_ERR_LEVEL_ERROR   = 0x4,    /* error conditions                             */
    SAMPLE_SVP_ERR_LEVEL_CRIT    = 0x5,    /* critical conditions                          */
    SAMPLE_SVP_ERR_LEVEL_ALERT   = 0x6,    /* action must be taken immediately             */
    SAMPLE_SVP_ERR_LEVEL_FATAL   = 0x7,    /* just for compatibility with previous version */

    SAMPLE_SVP_ERR_LEVEL_BUTT
} sample_svp_err_level;

typedef struct {
    ot_point point[OT_POINT_NUM];
} ot_sample_svp_rect;

typedef struct {
    td_u16 num;
    ot_sample_svp_rect rect[OT_SVP_RECT_NUM];
} ot_sample_svp_rect_info;

typedef struct {
    td_bool is_venc_open;
    td_bool is_vo_open;
} ot_sample_svp_switch;

#define sample_svp_printf(level_str, msg, ...) \
do { \
    fprintf(stderr, "[level]:%s,[func]:%s [line]:%d [info]:" msg, level_str, __FUNCTION__, __LINE__, ##__VA_ARGS__); \
} while (0)

#define sample_svp_printf_red(level_str, msg, ...) \
do { \
    fprintf(stderr, "\033[0;31m [level]:%s,[func]:%s [line]:%d [info]:" msg"\033[0;39m\n", level_str, __FUNCTION__, \
        __LINE__, ##__VA_ARGS__); \
} while (0)
/* system is unusable   */
#define sample_svp_trace_fatal(msg, ...)   sample_svp_printf_red("Fatal", msg, ##__VA_ARGS__)
/* action must be taken immediately */
#define sample_svp_trace_alert(msg, ...)   sample_svp_printf_red("Alert", msg, ##__VA_ARGS__)
/* critical conditions */
#define sample_svp_trace_critical(msg, ...)    sample_svp_printf_red("Critical", msg, ##__VA_ARGS__)
/* error conditions */
#define sample_svp_trace_err(msg, ...)     sample_svp_printf_red("Error", msg, ##__VA_ARGS__)
/* warning conditions */
#define sample_svp_trace_warning(msg, ...)    sample_svp_printf("Warning", msg, ##__VA_ARGS__)
/* normal but significant condition  */
#define sample_svp_trace_notic(msg, ...)  sample_svp_printf("Notice", msg, ##__VA_ARGS__)
/* informational */
#define sample_svp_trace_info(msg, ...)    sample_svp_printf("Info", msg, ##__VA_ARGS__)
/* debug-level messages  */
#define sample_svp_trace_debug(msg, ...)  sample_svp_printf("Debug", msg, ##__VA_ARGS__)

/* exps is true, goto */
#define sample_svp_check_exps_goto(exps, label, level, msg, ...)                  \
do {                                                                              \
    if ((exps)) {                                                                 \
        sample_svp_trace_err(msg, ## __VA_ARGS__);                                \
        goto label;                                                               \
    }                                                                             \
} while (0)
/* exps is true, return td_void */
#define sample_svp_check_exps_return_void(exps, level, msg, ...)                 \
do {                                                                             \
    if ((exps)) {                                                                \
        sample_svp_trace_err(msg, ##__VA_ARGS__);                                \
        return;                                                                  \
    }                                                                            \
} while (0)
/* exps is true, return ret */
#define sample_svp_check_exps_return(exps, ret, level, msg, ...)                 \
do {                                                                             \
    if ((exps)) {                                                                \
        sample_svp_trace_err(msg, ##__VA_ARGS__);                                \
        return (ret);                                                            \
    }                                                                            \
} while (0)                                                                      \
/* exps is true, trace */
#define sample_svp_check_exps_trace(exps, level, msg, ...)                       \
do {                                                                             \
    if ((exps)) {                                                                \
        sample_svp_trace_err(msg, ##__VA_ARGS__);                                \
    }                                                                            \
} while (0)

#define sample_svp_check_exps_continue(exps, level, msg, ...)                    \
do {                                                                             \
    if ((exps)) {                                                                \
        sample_svp_trace_err(msg, ##__VA_ARGS__);                                \
        continue;                                                                \
    }                                                                            \
} while (0)                                                                      \

/* exps is not success, trace */
#define sample_svp_check_failed_trace(exps, level, msg, ...)                     \
do {                                                                             \
    if ((exps) != TD_SUCCESS) {                                                  \
        sample_svp_trace_err(msg, ##__VA_ARGS__);                                \
    }                                                                            \
} while (0)

/* exps is not success, goto */
#define sample_svp_check_failed_goto(exps, label, level, msg, ...)                \
do {                                                                              \
    if ((exps) != TD_SUCCESS) {                                                   \
        sample_svp_trace_err(msg, ## __VA_ARGS__);                                \
        goto label;                                                               \
    }                                                                             \
} while (0)                                                                       \

/* exps is not sucecss, return */
#define sample_svp_check_failed_return(exps, ret, level, msg, ...)               \
do {                                                                             \
    if ((exps) != TD_SUCCESS) {                                                  \
        sample_svp_trace_err(msg, ##__VA_ARGS__);                                \
        return (ret);                                                            \
    }                                                                            \
} while (0)

/* exps is not equal to success, goto with level SAMPLE_SVP_ERR_LEVEL_ERROR */
#define sample_svp_check_failed_err_level_goto(exps, label, msg, ...)             \
do {                                                                              \
    if ((exps) != TD_SUCCESS) {                                                   \
        sample_svp_trace_err(msg, ## __VA_ARGS__);                                \
        goto label;                                                               \
    }                                                                             \
} while (0)

/* exps is not equal to success, return with level SAMPLE_SVP_ERR_LEVEL_ERROR */
#define sample_svp_check_failed_err_level_return(exps, ret, msg, ...)             \
do {                                                                              \
    if ((exps) != TD_SUCCESS) {                                                   \
        sample_svp_trace_err(msg, ## __VA_ARGS__);                                \
        return (ret);                                                             \
    }                                                                             \
} while (0)

/* exps is not equal success, trace with level SAMPLE_SVP_ERR_LEVEL_ERROR */
#define sample_svp_check_failed_err_level_trace(exps, msg, ...)                  \
do {                                                                             \
    if ((exps) != TD_SUCCESS) {                                                  \
        sample_svp_trace_err(msg, ##__VA_ARGS__);                                \
    }                                                                            \
} while (0)

#define sample_svp_pause()                                                      \
do {                                                                            \
    printf("---------------press Enter key to exit!---------------\n");         \
    (void)getchar();                                                            \
} while (0)

#define SAMPLE_SVP_VB_POOL_NUM     2
#define SAMPLE_SVP_ALIGN_16        16
#define SAMPLE_SVP_ALIGN_32        32
#define SAMPLE_SVP_D1_PAL_HEIGHT   576
#define SAMPLE_SVP_D1_PAL_WIDTH    704
#define sample_svp_convert_addr_to_ptr(type, addr) ((type *)(td_uintptr_t)(addr))
#define sample_svp_convert_ptr_to_addr(type, addr) ((type)(td_uintptr_t)(addr))

/* free mmz */
#define sample_svp_mmz_free(phys, virt)                                                 \
do {                                                                                    \
    if (((phys) != 0) && ((virt) != 0)) {                                               \
        ss_mpi_sys_mmz_free((td_phys_addr_t)(phys), (td_void*)(td_uintptr_t)(virt));    \
        (phys) = 0;                                                                     \
        (virt) = 0;                                                                     \
    }                                                                                   \
} while (0)

#define sample_svp_close_file(fp)   \
do {                                \
    if ((fp) != TD_NULL) {          \
        fclose((fp));               \
        (fp) = TD_NULL;             \
    }                               \
} while (0)

/* System init */
td_s32 sample_common_svp_check_sys_init(td_void);

/* System exit */
td_void sample_common_svp_check_sys_exit(td_void);

/* Align */
td_u32 sample_common_svp_align(td_u32 size, td_u16 align);

/* Create mem info */
td_s32 sample_common_svp_create_mem_info(ot_svp_mem_info *mem_info, td_u32 size, td_u32 addr_offsset);

/* Destory mem info */
td_void sample_common_svp_destroy_mem_info(ot_svp_mem_info *mem_info, td_u32 addr_offsset);

/* Malloc memory */
td_s32 sample_common_svp_malloc_mem(td_char *mmb, td_char *zone, td_phys_addr_t *phys_addr,
    td_void **virt_addr, td_u32 size);

/* Malloc memory with cached */
td_s32 sample_common_svp_malloc_cached(td_char *mmb, td_char *zone, td_phys_addr_t *phys_addr,
    td_void **virt_addr, td_u32 size);

/* Fulsh cached */
td_s32 sample_common_svp_flush_cache(td_phys_addr_t phys_addr, td_void *virt_addr, td_u32 size);

/* function : Call vgs to fill rect */
td_s32 sample_common_svp_vgs_fill_rect(ot_video_frame_info *frame_info,
    ot_sample_svp_rect_info *rect, td_u32 color);

/* function : Start Vo */
td_s32 sample_common_svp_start_vo(sample_vo_cfg *vo_cfg);

/* function : Stop Vo */
td_void sample_common_svp_stop_vo(sample_vo_cfg *vo_cfg);

/* function : Start Vi/Vpss/Venc/Vo */
td_s32 sample_common_svp_start_vi_vpss_venc_vo(sample_vi_cfg *vi_config,
    ot_sample_svp_switch *switch_ptr, ot_pic_size *ext_pic_size_type);

/* function : Stop Vi/Vpss/Venc/Vo */
td_void sample_common_svp_stop_vi_vpss_venc_vo(sample_vi_cfg *vi_config,
    ot_sample_svp_switch *switch_ptr);

/*
 * Load bin
 */
td_s32 sample_comm_svp_load_core_binary(ot_svp_dsp_id core_id);

/*
 * UnLoad bin
 */
void sample_comm_svp_unload_core_binary(ot_svp_dsp_id core_id);


#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */
#endif /* SAMPLE_COMMON_SVP_H */
