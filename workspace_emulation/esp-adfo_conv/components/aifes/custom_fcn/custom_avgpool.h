#ifndef CUSTOM_AVGPOOL_H
#define CUSTOM_AVGPOOL_H

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "core/aifes_core.h"
#include "basic/base/aimath/aimath_f32.h"
#include "basic/base/aimath/aimath_basic.h"
#include "basic/default/aimath/aimath_f32_default.h"


#define HW(h, w)        {h, w}
#define CUSTOM_AVGPOOL2D_F32_A(pool_size, stride, padding) \
            {{0,},pool_size,stride,padding,}

typedef struct custom_avgpool2d custom_avgpool2d_f32_t;

struct custom_avgpool2d {
    ailayer_t base;
    uint16_t pool_size[2];
    uint16_t stride[2];
    uint16_t padding[2];
    int8_t channel_axis;
    void *optimem[2];
    void (*forward_pool)(
        const aitensor_t *input,
        const uint16_t pool_size[2],
        const uint16_t stride[2],
        const uint16_t padding[2],
        aitensor_t *output);
    void (*backward_pool)(
        const aitensor_t *delta_out,
        const uint16_t pool_size[2],
        const uint16_t stride[2],
        const uint16_t padding[2],
        aitensor_t *delta_in);
    uint16_t result_shape[4];
};

ailayer_t *custom_avgpool2d_chw_f32_default(
    custom_avgpool2d_f32_t *layer,
    ailayer_t *input_layer);

#endif
