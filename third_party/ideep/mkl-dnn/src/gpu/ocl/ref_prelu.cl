/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/ocl/ocl_eltwise.h"
#include "gpu/ocl/ocl_types.h"

#if IS_FWD

__kernel void ref_prelu_fwd(const __global SRC_DATA_T *src,
        const __global WEI_DATA_T *weights, __global DST_DATA_T *dst) {

    const int d0 = GWS_GET_D0();
    const int d1 = GWS_GET_D1();
    const int d2 = GWS_GET_D2();
    const int d3 = GWS_GET_D3();
    const int d4 = GWS_GET_D4();
    const int d5 = GWS_GET_D5();

    const unsigned data_off = OFF_MD(SRC, d0, d1, d2, d3, d4, d5);
    const unsigned wei_off = OFF_MD(WEI, d0 % WEI_D0, d1 % WEI_D1, d2 % WEI_D2,
            d3 % WEI_D3, d4 % WEI_D4, d5 % WEI_D5);

    const float src_data = SRC_TO_REF(src[data_off]);

    const float wei_data = WEI_TO_REF(weights[wei_off]);

    const float res_data = relu_fwd(src_data, wei_data);

    dst[data_off] = TO_DST(res_data);
}

#else // #if IS_FWD

__kernel void ref_prelu_bwd(const __global SRC_DATA_T *src,
        const __global WEI_DATA_T *weights, const __global DST_DATA_T *diff_dst,
        __global DIFF_SRC_DATA_T *diff_src,
        __global DIFF_WEI_DATA_T *diff_weights) {
    const int d0 = GWS_GET_D0();
    const int d1 = GWS_GET_D1();
    const int d2 = GWS_GET_D2();
    const int d3 = GWS_GET_D3();
    const int d4 = GWS_GET_D4();
    const int d5 = GWS_GET_D5();

    const unsigned data_off = OFF_MD(SRC, d0, d1, d2, d3, d4, d5);
    const unsigned wei_off = OFF_MD(WEI, d0 % WEI_D0, d1 % WEI_D1, d2 % WEI_D2,
            d3 % WEI_D3, d4 % WEI_D4, d5 % WEI_D5);

    const float src_data = SRC_TO_REF(src[data_off]);
    const float diff_dst_data = DST_TO_REF(diff_dst[data_off]);
    const float wei_data = WEI_TO_REF(weights[wei_off]);

    const float diff_src_data
            = src_data > 0 ? diff_dst_data : diff_dst_data * wei_data;

    const float diff_wei_data = src_data > 0 ? 0 : diff_dst_data * src_data;

    diff_src[data_off] = TO_SRC(diff_src_data);

    const unsigned diff_wei_off = OFF_MD(DIFF_WEI, d0 % DIFF_WEI_D0,
            d1 % DIFF_WEI_D1, d2 % DIFF_WEI_D2, d3 % DIFF_WEI_D3,
            d4 % DIFF_WEI_D4, d5 % DIFF_WEI_D5);

#if DIFF_WEI_DT_F32
    diff_weights[diff_wei_off] = diff_wei_data;
#else // #if DIFF_WEI_DT_F32
    diff_weights[diff_wei_off] = TO_DIFF_WEI(diff_wei_data);
#endif // #else // #if DIFF_WEI_DT_F32
}

#endif // #else // #if IS_FWD
