// SPDX-License-Identifier: MIT
// Thin D2R aggregation unit; implementation lives in d2r_parts/.
#include "d2r_parts/00_layout_and_load_helpers.inc"
#include "d2r_parts/01_mma_mainloop.inc"
#include "d2r_parts/02_moe_kernels.inc"
#include "d2r_parts/03_moe_launchers.inc"
#include "d2r_parts/04_dense_q8_path.inc"
