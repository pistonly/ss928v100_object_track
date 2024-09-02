/*
  Copyright (c), 2001-2022, Shenshu Tech. Co., Ltd.
 */
#include "sample_common_svp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "ot_common.h"
#include "ot_common_video.h"
#include "ot_common_sys.h"
#include "ot_common_svp.h"
#include "ss_mpi_vb.h"
#include "sample_comm.h"
#include "sample_common_ive.h"

#define SAMPLE_SVP_BLK_CNT           16
#define SMAPLE_SVP_DISPLAY_BUF_LEN   3
#define SAMPLE_SVP_VI_CHN_INTERVAL   4
#define SAMPLE_SVP_VPSS_BORDER_WIDTH 2
#define SAMPLE_SVP_VO_DIS_BUF_LEN    3
#define SAMPLE_SVP_MAX_WIDTH         32768
#define SAMPLE_SVP_NUM_TWO          2
#define SAMPLE_SVP_DSP_BIN_NUM_PER  4
#define SAMPLE_SVP_DSP_MEM_TYPE_SYS_DDR 0
#define SAMPLE_SVP_DSP_MEM_TYPE_IRAM    1
#define SAMPLE_SVP_DSP_MEM_TYPE_DRAM_0  2
#define SAMPLE_SVP_DSP_MEM_TYPE_DRAM_1  3

static td_bool g_sample_svp_init_flag = TD_FALSE;

/* System init */
static td_s32 sample_comm_svp_sys_init(td_void)
{
    td_s32 ret;
    ot_vb_cfg vb_cfg;

    ret = ss_mpi_sys_exit();
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x):ss_mpi_sys_exit failed!\n", ret);
    ret = ss_mpi_vb_exit();
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x):ss_mpi_vb_exit failed!\n", ret);

    (td_void)memset_s(&vb_cfg, sizeof(ot_vb_cfg), 0, sizeof(ot_vb_cfg));

    vb_cfg.max_pool_cnt = SAMPLE_SVP_VB_POOL_NUM;
    vb_cfg.common_pool[1].blk_size = SAMPLE_SVP_D1_PAL_WIDTH * SAMPLE_SVP_D1_PAL_HEIGHT * SAMPLE_SVP_VB_POOL_NUM;
    vb_cfg.common_pool[1].blk_cnt  = 1;

    ret = ss_mpi_vb_set_cfg((const ot_vb_cfg *)&vb_cfg);
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x):ss_mpi_vb_set_config failed!\n", ret);

    ret = ss_mpi_vb_init();
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x):ss_mpi_vb_init failed!\n", ret);

    ret = ss_mpi_sys_init();
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x):ss_mpi_sys_init failed!\n", ret);

    return ret;
}

/* System exit */
static td_s32 sample_comm_svp_sys_exit(td_void)
{
    td_s32 ret;

    ret = ss_mpi_sys_exit();
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x):ss_mpi_sys_exit failed!\n", ret);

    ret = ss_mpi_vb_exit();
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x):ss_mpi_vb_exit failed!\n", ret);

    return TD_SUCCESS;
}

/* System init */
td_s32 sample_common_svp_check_sys_init(td_void)
{
    if (g_sample_svp_init_flag == TD_FALSE) {
        if (sample_comm_svp_sys_init() != TD_SUCCESS) {
            sample_svp_trace_err("Svp mpi init failed!\n");
            return TD_FALSE;
        }
        g_sample_svp_init_flag = TD_TRUE;
    }

    sample_svp_trace_info("Svp mpi init ok!\n");
    return TD_TRUE;
}

/* System exit */
td_void sample_common_svp_check_sys_exit(td_void)
{
    td_s32 ret;

    if (g_sample_svp_init_flag == TD_TRUE) {
        ret = sample_comm_svp_sys_exit();
        if (ret != TD_SUCCESS) {
            sample_svp_trace_err("svp mpi exit failed!\n");
        }
    }
    g_sample_svp_init_flag = TD_FALSE;
    sample_svp_trace_info("Svp mpi exit ok!\n");
}

/* Align */
td_u32 sample_common_svp_align(td_u32 size, td_u16 align)
{
    td_u32 stride;

    sample_svp_check_exps_return(align == 0, 0, SAMPLE_SVP_ERR_LEVEL_ERROR, "align can't be zero!\n");
    sample_svp_check_exps_return((size < 1) || (size > SAMPLE_SVP_MAX_WIDTH), 0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "size(%u) must be [1, %u]\n", size, SAMPLE_SVP_MAX_WIDTH);
    stride = size + (align - size % align) % align;
    return stride;
}

/* Create mem info */
td_s32 sample_common_svp_create_mem_info(ot_svp_mem_info *mem_info, td_u32 size, td_u32 addr_offset)
{
    td_s32 ret = TD_FAILURE;
    td_u32 size_tmp;

    sample_svp_check_exps_return(mem_info == TD_NULL, ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "mem_info can't be zero\n");

    size_tmp = size + addr_offset;
    mem_info->size = size;
    ret = ss_mpi_sys_mmz_alloc((td_phys_addr_t *)(&mem_info->phys_addr),
        (void **)&mem_info->virt_addr, TD_NULL, TD_NULL, size_tmp);
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x):ss_mpi_sys_alloc failed!\n", ret);

    mem_info->phys_addr += addr_offset;
    mem_info->virt_addr += addr_offset;

    return ret;
}

/* Destory mem info */
td_void sample_common_svp_destroy_mem_info(ot_svp_mem_info *mem_info, td_u32 addr_offset)
{
    sample_svp_check_exps_return_void(mem_info == TD_NULL, SAMPLE_SVP_ERR_LEVEL_ERROR, "mem_info can't be zero\n");

    if ((mem_info->virt_addr != 0) && (mem_info->phys_addr != 0)) {
        (td_void)ss_mpi_sys_mmz_free(mem_info->phys_addr - addr_offset,
            sample_svp_convert_addr_to_ptr(void, (mem_info->virt_addr - addr_offset)));
    }
    (td_void)memset_s(mem_info, sizeof(*mem_info), 0, sizeof(*mem_info));
}
/* Malloc memory */
td_s32 sample_common_svp_malloc_mem(td_char *mmb, td_char *zone, td_phys_addr_t *phys_addr,
    td_void **virt_addr, td_u32 size)
{
    td_s32 ret = TD_FAILURE;

    sample_svp_check_exps_return(phys_addr == TD_NULL, ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "phys_addr can't be null\n");
    sample_svp_check_exps_return(virt_addr == TD_NULL, ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "virt_addr can't be null\n");
    ret = ss_mpi_sys_mmz_alloc((td_phys_addr_t *)phys_addr, virt_addr, mmb, zone, size);

    return ret;
}

/* Malloc memory with cached */
td_s32 sample_common_svp_malloc_cached(td_char *mmb, td_char *zone, td_phys_addr_t *phys_addr,
    td_void **virt_addr, td_u32 size)
{
    td_s32 ret = TD_FAILURE;

    sample_svp_check_exps_return(phys_addr == TD_NULL, ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "phys_addr can't be null\n");
    sample_svp_check_exps_return(virt_addr == TD_NULL, ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "virt_addr can't be null\n");
    ret = ss_mpi_sys_mmz_alloc_cached((td_phys_addr_t *)phys_addr, virt_addr, mmb, zone, size);

    return ret;
}

/* Fulsh cached */
td_s32 sample_common_svp_flush_cache(td_phys_addr_t phys_addr, td_void *virt_addr, td_u32 size)
{
    td_s32 ret = TD_FAILURE;

    sample_svp_check_exps_return(virt_addr == TD_NULL, ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "virt_addr can't be null\n");
    ret = ss_mpi_sys_flush_cache((td_phys_addr_t)phys_addr, virt_addr, size);
    return ret;
}

/*
 * function : Init Vb
 */
static td_s32 sample_common_svp_vb_init(ot_pic_size *pic_type, ot_size *pic_size,
    td_u32 vpss_chn_num)
{
    td_s32 ret;
    td_u32 i;
    ot_vb_cfg vb_cfg = {0};
    ot_pic_buf_attr pic_buf_attr;
    ot_vb_calc_cfg calc_cfg;
    ot_vi_vpss_mode_type mode_type = OT_VI_ONLINE_VPSS_OFFLINE;
    ot_vi_video_mode video_mode = OT_VI_VIDEO_MODE_NORM;

    vb_cfg.max_pool_cnt = OT_SAMPLE_IVE_MAX_POOL_CNT;

    ret = sample_comm_sys_get_pic_size(pic_type[0], &pic_size[0]);
    sample_svp_trace_debug("vb init pic_size, width: %d, height: %d", pic_size[0].width, pic_size[0].height);
    sample_svp_check_exps_goto(ret != TD_SUCCESS, vb_fail_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "sample_comm_sys_get_pic_size failed,Error(%#x)!\n", ret);
    pic_buf_attr.width = pic_size[0].width;
    pic_buf_attr.height = pic_size[0].height;
    pic_buf_attr.align = OT_DEFAULT_ALIGN;
    pic_buf_attr.bit_width = OT_DATA_BIT_WIDTH_8;
    pic_buf_attr.pixel_format = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_422;
    pic_buf_attr.compress_mode = OT_COMPRESS_MODE_NONE;

    ot_common_get_pic_buf_cfg(&pic_buf_attr, &calc_cfg);

    vb_cfg.common_pool[0].blk_size = calc_cfg.vb_size;
    vb_cfg.common_pool[0].blk_cnt = SAMPLE_SVP_BLK_CNT;

    for (i = 1; (i < vpss_chn_num) && (i < OT_VB_MAX_COMMON_POOLS); i++) {
        ret = sample_comm_sys_get_pic_size(pic_type[i], &pic_size[i]);
        sample_svp_check_exps_goto(ret != TD_SUCCESS, vb_fail_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "sample_comm_sys_get_pic_size failed,Error(%#x)!\n", ret);
        pic_buf_attr.width = pic_size[i].width;
        pic_buf_attr.height = pic_size[i].height;
        pic_buf_attr.compress_mode = OT_COMPRESS_MODE_NONE;
        pic_buf_attr.align = OT_DEFAULT_ALIGN;

        ot_common_get_pic_buf_cfg(&pic_buf_attr, &calc_cfg);

        /* comm video buffer */
        vb_cfg.common_pool[i].blk_size = calc_cfg.vb_size;
        vb_cfg.common_pool[i].blk_cnt = SAMPLE_SVP_BLK_CNT;
    }

    ret = sample_comm_sys_init_with_vb_supplement(&vb_cfg, OT_VB_SUPPLEMENT_BNR_MOT_MASK);
    sample_svp_check_exps_goto(ret != TD_SUCCESS, vb_fail_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "sample_comm_sys_init failed,Error(%#x)!\n", ret);

    ret = sample_comm_vi_set_vi_vpss_mode(mode_type, video_mode);
    sample_svp_check_exps_goto(ret != TD_SUCCESS, vb_fail_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "sample_comm_vi_set_vi_vpss_mode failed!\n");
    return ret;
vb_fail_1:
    sample_comm_sys_exit();
vb_fail_0:
    return ret;
}

td_s32 sample_common_svp_vgs_fill_rect(ot_video_frame_info *frame_info,
    ot_sample_svp_rect_info *rect, td_u32 color)
{
    ot_vgs_handle vgs_handle = -1;
    td_s32 ret = TD_FAILURE;
    td_u16 i;
    ot_vgs_task_attr vgs_task;
    ot_cover vgs_add_cover;

    sample_svp_check_exps_return(frame_info == TD_NULL, ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "frame_info can't be null\n");
    sample_svp_check_exps_return(rect == TD_NULL, ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "rect can't be null\n");
    sample_svp_check_exps_return(rect->num > OT_SVP_RECT_NUM, ret,
        SAMPLE_SVP_ERR_LEVEL_ERROR, "rect->num can't lager than %u\n", OT_SVP_RECT_NUM);
    if (rect->num == 0) {
        return TD_SUCCESS;
    }

    ret = ss_mpi_vgs_begin_job(&vgs_handle);
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Vgs begin job fail,Error(%#x)\n", ret);

    ret = memcpy_s(&vgs_task.img_in, sizeof(ot_video_frame_info), frame_info, sizeof(ot_video_frame_info));
    sample_svp_check_exps_goto(ret != EOK, fail, SAMPLE_SVP_ERR_LEVEL_ERROR, "get img_in failed\n");
    ret = memcpy_s(&vgs_task.img_out, sizeof(ot_video_frame_info), frame_info, sizeof(ot_video_frame_info));
    sample_svp_check_exps_goto(ret != EOK, fail, SAMPLE_SVP_ERR_LEVEL_ERROR, "get img_out failed\n");

    vgs_add_cover.type = OT_COVER_QUAD;
    vgs_add_cover.color = color;
    for (i = 0; i < rect->num; i++) {
        vgs_add_cover.quad.is_solid = TD_FALSE;
        vgs_add_cover.quad.thick = OT_SAMPLE_IVE_DRAW_THICK;
        ret = memcpy_s(vgs_add_cover.quad.point, sizeof(rect->rect[i].point),
            rect->rect[i].point, sizeof(rect->rect[i].point));
        sample_svp_check_exps_goto(ret != EOK, fail, SAMPLE_SVP_ERR_LEVEL_ERROR, "get point failed\n");
        ret = ss_mpi_vgs_add_cover_task(vgs_handle, &vgs_task, &vgs_add_cover, 1);
        sample_svp_check_exps_goto(ret != TD_SUCCESS, fail, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "ss_mpi_vgs_add_cover_task fail,Error(%#x)\n", ret);
    }

    ret = ss_mpi_vgs_end_job(vgs_handle);
    sample_svp_check_exps_goto(ret != TD_SUCCESS, fail, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "ss_mpi_vgs_end_job fail,Error(%#x)\n", ret);

    return ret;
fail:
    ss_mpi_vgs_cancel_job(vgs_handle);
    return ret;
}

/* function : Start Vpss */
static td_s32 sample_common_svp_start_vpss(td_s32 vpss_grp_cnt, ot_size *pic_size, td_u32 vpss_chn_num)
{
    td_u32 i;
    ot_vpss_chn_attr vpss_chn_attr[OT_VPSS_MAX_CHN_NUM];
    ot_vpss_grp_attr vpss_grp_attr;
    td_bool chn_enable[OT_VPSS_MAX_CHN_NUM] = { TD_TRUE, TD_TRUE, TD_FALSE, TD_FALSE };
    ot_vpss_grp vpss_grp;
    td_s32 ret;

    (td_void)memset_s(&vpss_grp_attr, sizeof(ot_vpss_grp_attr), 0, sizeof(ot_vpss_grp_attr));
    sample_comm_vpss_get_default_grp_attr(&vpss_grp_attr);
    vpss_grp_attr.max_width = pic_size[0].width;
    vpss_grp_attr.max_height = pic_size[0].height;
    /* VPSS only onle channel0 support compress seg mode */
    sample_comm_vpss_get_default_chn_attr(&vpss_chn_attr[0]);
    vpss_chn_attr[0].width = pic_size[0].width;
    vpss_chn_attr[0].height = pic_size[0].height;
    vpss_chn_attr[0].depth = 1;

    for (i = 1; i < vpss_chn_num; i++) {
        (td_void)memset_s(&vpss_chn_attr[i], sizeof(ot_vpss_chn_attr), 0, sizeof(ot_vpss_chn_attr));
        sample_comm_vpss_get_default_chn_attr(&vpss_chn_attr[i]);
        vpss_chn_attr[i].width = pic_size[i].width;
        vpss_chn_attr[i].height = pic_size[i].height;
        vpss_chn_attr[i].compress_mode = OT_COMPRESS_MODE_NONE;
        vpss_chn_attr[i].depth = 1;
    }

    for (vpss_grp = 0; vpss_grp < vpss_grp_cnt; vpss_grp++) {
        ret = sample_common_vpss_start(vpss_grp, chn_enable, &vpss_grp_attr, vpss_chn_attr, OT_VPSS_MAX_CHN_NUM);
        if (ret != TD_SUCCESS) {
            sample_svp_trace_err("failed with %#x!\n", ret);
            return TD_FAILURE;
        }
    }
    return TD_SUCCESS;
}

/* function : Stop Vpss */
static td_void sample_common_svp_stop_vpss(td_s32 vpss_grp_cnt)
{
    ot_vpss_grp vpss_grp = 0;
    td_bool chn_enable[OT_VPSS_MAX_CHN_NUM] = { TD_TRUE, TD_TRUE, TD_FALSE, TD_FALSE };
    td_s32 i;

    for (i = 0; (i < vpss_grp_cnt) && (i < OT_VPSS_MAX_CHN_NUM); i++) {
        sample_common_vpss_stop(vpss_grp, chn_enable, OT_VPSS_MAX_CHN_NUM);
        vpss_grp++;
    }
}

/*
 * function : Start Vo
 */
td_s32 sample_common_svp_start_vo(sample_vo_cfg *vo_cfg)
{
    sample_svp_check_exps_return(vo_cfg == TD_NULL, TD_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR, "vo_cfg can't be null\n");
    return sample_comm_vo_start_vo(vo_cfg);
}

/*
 * function : Stop Vo
 */
td_void sample_common_svp_stop_vo(sample_vo_cfg *vo_cfg)
{
    sample_svp_check_exps_return_void(vo_cfg == TD_NULL, SAMPLE_SVP_ERR_LEVEL_ERROR, "vo_cfg can't be null\n");
    (td_void)sample_comm_vo_stop_vo(vo_cfg);
}

static td_s32 sample_common_svp_vi_bind_multi_vpss(td_s32 vpss_grp_cnt, td_s32 vi_chn_cnt,
    td_s32 vi_chn_interval)
{
    td_s32 ret;
    td_s32 loop;
    ot_vi_chn vi_chn;
    ot_vpss_grp vpss_grp = 0;

    for (loop = 0; loop < vi_chn_cnt  && vpss_grp < vpss_grp_cnt; loop++) {
        vi_chn = loop * vi_chn_interval;
        ret = sample_comm_vi_bind_vpss(0, vi_chn, vpss_grp, 0);
        if (ret != TD_SUCCESS) {
            sample_svp_trace_err("vi bind vpss failed!\n");
            return ret;
        }
        vpss_grp++;
    }

    return TD_SUCCESS;
}

static td_s32 sample_common_svp_vi_unbind_multi_vpss(td_s32 vpss_grp_cnt, td_s32 vi_chn_cnt,
    td_s32 vi_chn_interval)
{
    td_s32 ret;
    td_s32 loop;
    ot_vi_chn vi_chn;
    ot_vpss_grp vpss_grp = 0;

    for (loop = 0; loop < vi_chn_cnt && vpss_grp < vpss_grp_cnt; loop++) {
        vi_chn = loop * vi_chn_interval;
        ret = sample_comm_vi_un_bind_vpss(0, vi_chn, vpss_grp, 0);
        if (ret != TD_SUCCESS) {
            sample_svp_trace_err("vi bind vpss failed!\n");
            return ret;
        }

        vpss_grp++;
    }

    return TD_SUCCESS;
}

static td_s32 sample_common_svp_set_vi_cfg(sample_vi_cfg *vi_cfg, ot_pic_size *pic_type,
    td_u32 pic_type_len, ot_pic_size *ext_pic_size_type, sample_sns_type sns_type)
{
    sample_comm_vi_get_default_vi_cfg(sns_type, vi_cfg);
    sample_svp_check_exps_return(pic_type_len < OT_VPSS_CHN_NUM,
        TD_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR, "pic_type_len is illegal!\n");
    pic_type[1] = *ext_pic_size_type;

    return TD_SUCCESS;
}

static td_s32 sample_common_svp_get_def_vo_cfg(sample_vo_cfg *vo_cfg)
{
    ot_rect  def_disp_rect  = {0, 0, 1920, 1080};
    ot_size  def_img_size = {1920, 1080};

    if (vo_cfg == TD_NULL) {
        sample_svp_trace_err("error:argument can not be TD_NULL\n");
        return TD_FAILURE;
    }

    vo_cfg->vo_dev  = SAMPLE_VO_DEV_UHD;

    vo_cfg->vo_intf_type      = OT_VO_INTF_HDMI;
    vo_cfg->intf_sync         = OT_VO_OUT_1080P30;
    vo_cfg->bg_color          = COLOR_RGB_BLACK;
    vo_cfg->pix_format        = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    vo_cfg->disp_rect         = def_disp_rect;
    vo_cfg->image_size        = def_img_size;
    vo_cfg->vo_part_mode      = OT_VO_PARTITION_MODE_SINGLE;
    vo_cfg->dis_buf_len       = 3; /* 3: def buf len for single */
    vo_cfg->dst_dynamic_range = OT_DYNAMIC_RANGE_SDR8;
    vo_cfg->vo_mode           = VO_MODE_1MUX;
    vo_cfg->compress_mode     = OT_COMPRESS_MODE_NONE;

    return TD_SUCCESS;
}

static td_s32 sample_common_svp_set_and_start_vo(const ot_sample_svp_switch *switch_ptr,
    sample_vo_cfg *vo_cfg)
{
    td_s32 ret;

    /*  Start Vo */
    if (switch_ptr->is_vo_open == TD_TRUE) {
        sample_common_svp_get_def_vo_cfg(vo_cfg);
        ret = sample_common_svp_start_vo(vo_cfg);
        sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error(%#x),sample_common_svp_start_vo failed!\n", ret);
    }
    return TD_SUCCESS;
}

static td_s32 sample_common_svp_start_venc(const ot_sample_svp_switch *switch_ptr,
    sample_vo_cfg *vo_cfg)
{
    td_s32 ret = TD_SUCCESS;
    ot_venc_chn h264_chn = 0;
    sample_comm_venc_chn_param chn_param;
    ot_size ven_size = {1920, 1080};

    chn_param.frame_rate = 30; /* 30 is a number */
    chn_param.stats_time = 1; /* 1 is a number */
    chn_param.gop = 30; /* 30 is a number */
    chn_param.venc_size = ven_size;
    chn_param.size = PIC_1080P;
    chn_param.profile = 0;
    chn_param.is_rcn_ref_share_buf = TD_TRUE;

    chn_param.type = OT_PT_H264;
    chn_param.rc_mode = SAMPLE_RC_CBR;

    if (switch_ptr->is_venc_open  == TD_TRUE) {
        ret = sample_comm_venc_get_gop_attr(OT_VENC_GOP_MODE_NORMAL_P, &chn_param.gop_attr);
        sample_svp_check_exps_goto(ret != TD_SUCCESS, end_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error(%#x),sample_comm_venc_get_gop_attr failed!\n", ret);

        ret = sample_comm_venc_start(h264_chn, &chn_param);
        sample_svp_check_exps_goto(ret != TD_SUCCESS, end_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error(%#x),sample_comm_venc_start failed!\n", ret);

        ret = sample_comm_venc_start_get_stream(&h264_chn, 1);
        sample_svp_check_exps_goto(ret != TD_SUCCESS, end_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error(%#x),sample_comm_venc_start_get_stream failed!\n", ret);
    }
    return ret;

end_1:
    if (switch_ptr->is_venc_open == TD_TRUE) {
        sample_comm_venc_stop(h264_chn);
    }
end_0:
    if (switch_ptr->is_vo_open == TD_TRUE) {
        sample_common_svp_stop_vo(vo_cfg);
    }
    return ret;
}

static td_s32 sample_common_svp_get_pic_type_by_sns_type(sample_sns_type sns_type, ot_pic_size size[], td_u32 num)
{
    sample_svp_check_exps_return(num > OT_VPSS_CHN_NUM, TD_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "num(%u) can't be larger than (%u)\n", num, OT_VPSS_CHN_NUM);
    switch (sns_type) {
        case OV_OS08A20_MIPI_8M_30FPS_12BIT:
        case OV_OS08A20_MIPI_8M_30FPS_12BIT_WDR2TO1:
            size[0] = PIC_3840X2160;
            break;
        default:
            size[0] = PIC_3840X2160;
            break;
    }
    return TD_SUCCESS;
}

/*
 * function : Start Vi/Vpss/Venc/Vo
 */
td_s32 sample_common_svp_start_vi_vpss_venc_vo(sample_vi_cfg *vi_cfg,
    ot_sample_svp_switch *switch_ptr, ot_pic_size *ext_pic_size_type)
{
    ot_size pic_size[OT_VPSS_CHN_NUM];
    ot_pic_size pic_type[OT_VPSS_CHN_NUM];
    sample_vo_cfg vo_cfg;

    const td_s32 vpss_grp_cnt = 1;
    td_s32 ret = TD_FAILURE;
    sample_sns_type sns_type = SENSOR0_TYPE;

    sample_svp_check_exps_return(vi_cfg == TD_NULL, ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "vi_cfg can't be null\n");
    sample_svp_check_exps_return(switch_ptr == TD_NULL, ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "switch_ptr can't be null\n");
    sample_svp_check_exps_return(ext_pic_size_type == TD_NULL, ret,
        SAMPLE_SVP_ERR_LEVEL_ERROR, "ext_pic_size_type can't be null\n");

    ret = sample_common_svp_get_pic_type_by_sns_type(sns_type, pic_type, OT_VPSS_CHN_NUM);
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "sample_common_svp_get_pic_type_by_sns_type failed!\n");
    ret = sample_common_svp_set_vi_cfg(vi_cfg, pic_type, OT_VPSS_CHN_NUM, ext_pic_size_type, sns_type);
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "sample_common_svp_set_vi_cfg failed,Error:%#x\n", ret);

    // TODO: why need these two line? added by ly
    vi_cfg[0].mipi_info.divide_mode = LANE_DIVIDE_MODE_1;
    vi_cfg[0].sns_info.bus_id = 5;

    /* step  1: Init vb */
    ret = sample_common_svp_vb_init(pic_type, pic_size, OT_VPSS_CHN_NUM);
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),sample_common_svp_vb_init failed!\n", ret);

    /* step 2: Start vi */
    ret = sample_comm_vi_start_vi(vi_cfg);
    sample_svp_check_exps_goto(ret != TD_SUCCESS, end_init_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),sample_comm_vi_start_vi failed!\n", ret);

    /* step 3: Bind vpss to vi */
    ret = sample_common_svp_vi_bind_multi_vpss(vpss_grp_cnt, 1, 1);
    sample_svp_check_exps_goto(ret != TD_SUCCESS, end_init_2, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),sample_common_vi_bind_multi_vpss failed!\n", ret);

    /* step 4: Start vpss */
    ret = sample_common_svp_start_vpss(vpss_grp_cnt, pic_size, OT_VPSS_CHN_NUM);
    sample_svp_check_exps_goto(ret != TD_SUCCESS, end_init_3, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),sample_common_svp_start_vpss failed!\n", ret);

    /* step 5: Set vi frame, Start Vo */
    ret = sample_common_svp_set_and_start_vo(switch_ptr, &vo_cfg);
    sample_svp_check_exps_goto(ret != TD_SUCCESS, end_init_4, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),sample_common_svp_set_vi_frame failed!\n", ret);
    sample_svp_trace_debug("\n vo_cfg size, width: %d, height %d", vo_cfg.image_size.width, vo_cfg.image_size.height);

    /* step 6: Start Venc */
    ret = sample_common_svp_start_venc(switch_ptr, &vo_cfg);
    sample_svp_check_exps_goto(ret != TD_SUCCESS, end_init_4, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),sample_common_svp_start_vencb failed!\n", ret);

    return TD_SUCCESS;
end_init_4:
    sample_common_svp_stop_vpss(vpss_grp_cnt);
end_init_3:
    ret = sample_common_svp_vi_unbind_multi_vpss(vpss_grp_cnt, 1, 1);
    sample_svp_check_exps_trace(ret != TD_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR, "svp_vi_unbind_multi_vpss failed!\n");
end_init_2:
    sample_comm_vi_stop_vi(vi_cfg);
end_init_1:  /*  system exit */
    sample_comm_sys_exit();
    (td_void)memset_s(vi_cfg, sizeof(sample_vi_cfg), 0, sizeof(sample_vi_cfg));
    return ret;
}

/*
 * function : Stop Vi/Vpss/Venc/Vo
 */
td_void sample_common_svp_stop_vi_vpss_venc_vo(sample_vi_cfg *vi_cfg,
    ot_sample_svp_switch *switch_ptr)
{
    sample_vo_cfg vo_cfg = {0};
    td_s32 ret;
    const td_s32 vpss_grp_cnt = 1;

    sample_svp_check_exps_return_void(vi_cfg == TD_NULL, SAMPLE_SVP_ERR_LEVEL_ERROR, "vi_cfg can't be null\n");
    sample_svp_check_exps_return_void(switch_ptr == TD_NULL, SAMPLE_SVP_ERR_LEVEL_ERROR, "switch_ptr can't be null\n");

    if (switch_ptr->is_venc_open == TD_TRUE) {
        sample_comm_venc_stop_get_stream(1);
        sample_comm_venc_stop(0);
    }

    if (switch_ptr->is_vo_open == TD_TRUE) {
        (td_void)sample_common_svp_get_def_vo_cfg(&vo_cfg);
        sample_common_svp_stop_vo(&vo_cfg);
    }

    ret = sample_common_svp_vi_unbind_multi_vpss(vpss_grp_cnt, 1, 1);
    sample_svp_check_exps_trace(ret != TD_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "sample_common_svp_vi_unbind_multi_vpss failed\n");
    sample_common_svp_stop_vpss(vpss_grp_cnt);
    sample_comm_vi_stop_vi(vi_cfg);
    sample_comm_sys_exit();

    (td_void)memset_s(vi_cfg, sizeof(sample_vi_cfg), 0, sizeof(sample_vi_cfg));
}

/*
 * Load bin
 */
td_s32 sample_comm_svp_load_core_binary(ot_svp_dsp_id core_id)
{
    td_s32 ret;
    td_char *bin[OT_SVP_DSP_ID_BUTT][SAMPLE_SVP_DSP_BIN_NUM_PER] = {
        { "./dsp_bin/dsp0/ot_sram.bin", "./dsp_bin/dsp0/ot_iram0.bin",
            "./dsp_bin/dsp0/ot_dram0.bin", "./dsp_bin/dsp0/ot_dram1.bin" },
        { "./dsp_bin/dsp1/ot_sram.bin", "./dsp_bin/dsp1/ot_iram0.bin",
            "./dsp_bin/dsp1/ot_dram0.bin", "./dsp_bin/dsp1/ot_dram1.bin" } };

    ret = ss_mpi_svp_dsp_power_on(core_id);
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x):ss_mpi_svp_dsp_power_on failed!\n", ret);

    ret = ss_mpi_svp_dsp_load_bin(bin[core_id][SAMPLE_SVP_DSP_MEM_TYPE_IRAM],
        core_id * SAMPLE_SVP_DSP_BIN_NUM_PER + SAMPLE_SVP_DSP_MEM_TYPE_IRAM);
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x): ss_mpi_svp_dsp_load_bin failed!\n", ret);

    ret = ss_mpi_svp_dsp_load_bin(bin[core_id][SAMPLE_SVP_DSP_MEM_TYPE_SYS_DDR],
        core_id * SAMPLE_SVP_DSP_BIN_NUM_PER + SAMPLE_SVP_DSP_MEM_TYPE_SYS_DDR);
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x): ss_mpi_svp_dsp_load_bin failed!\n", ret);
    ret = ss_mpi_svp_dsp_load_bin(bin[core_id][SAMPLE_SVP_DSP_MEM_TYPE_DRAM_0],
        core_id * SAMPLE_SVP_DSP_BIN_NUM_PER + SAMPLE_SVP_DSP_MEM_TYPE_DRAM_0);
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x): ss_mpi_svp_dsp_load_bin failed!\n", ret);
    ret = ss_mpi_svp_dsp_load_bin(bin[core_id][SAMPLE_SVP_DSP_MEM_TYPE_DRAM_1],
        core_id * SAMPLE_SVP_DSP_BIN_NUM_PER + SAMPLE_SVP_DSP_MEM_TYPE_DRAM_1);
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x): ss_mpi_svp_dsp_load_bin failed!\n", ret);

    ret = ss_mpi_svp_dsp_enable_core(core_id);
    sample_svp_check_exps_return(ret != TD_SUCCESS, ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x): ss_mpi_svp_dsp_enable_core failed!\n", ret);
    return ret;
}

/*
 * UnLoad bin
 */
void sample_comm_svp_unload_core_binary(ot_svp_dsp_id core_id)
{
    td_s32 ret;
    ret = ss_mpi_svp_dsp_disable_core(core_id);
    sample_svp_check_exps_return_void(ret != TD_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x):ss_mpi_svp_dsp_disable_core failed!\n", ret);

    ret = ss_mpi_svp_dsp_power_off(core_id);
    sample_svp_check_exps_return_void(ret != TD_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x):ss_mpi_svp_dsp_power_off failed!\n", ret);
}

