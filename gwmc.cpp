
int main(int argc, char *argv[]) {
  const int64_t warp_error = av1_refine_integerized_param(
      &tmp_wm_params, tmp_wm_params.wmtype, is_cur_buf_hbd(xd), xd->bd,
      ref_buf[frame]->y_buffer, ref_buf[frame]->y_width,
      ref_buf[frame]->y_height, ref_buf[frame]->y_stride,
      cpi->source->y_buffer, src_width, src_height, src_stride,
      GM_REFINEMENT_COUNT, best_warp_error, segment_map, segment_map_w,
      erroradv_threshold);
}