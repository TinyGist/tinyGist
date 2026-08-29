#include "custom_fcn/custom_avgpool.h"

static const aicore_layertype_t custom_avgpool2d_type = {0, 0};

static void custom_avgpool2d_init_zeros(aitensor_t *tensor)
{
    uint32_t size = 1;
    uint32_t index;

    for (index = 0; index < tensor->dim; index++) {
        size *= tensor->shape[index];
    }

    for (index = 0; index < size; index++) {
        ((float *)tensor->data)[index] = 0.0f;
    }
}

static void custom_avgpool2d_calc_result_shape(ailayer_t *self)
{
    custom_avgpool2d_f32_t *layer =
        (custom_avgpool2d_f32_t *)self->layer_configuration;

    self->result.shape[0] = self->input_layer->result.shape[0];
    self->result.shape[1] = self->input_layer->result.shape[1];
    self->result.shape[2] =
        (self->input_layer->result.shape[2] + 2 * layer->padding[0]
         - layer->pool_size[0]) / layer->stride[0] + 1;
    self->result.shape[3] =
        (self->input_layer->result.shape[3] + 2 * layer->padding[1]
         - layer->pool_size[1]) / layer->stride[1] + 1;
}

static void custom_avgpool2d_fwd(
    const aitensor_t *input,
    const uint16_t pool_size[2],
    const uint16_t stride[2],
    const uint16_t padding[2],
    aitensor_t *output)
{
    const uint16_t batches = input->shape[0];
    const uint16_t channels = input->shape[1];
    const uint16_t input_height = input->shape[2];
    const uint16_t input_width = input->shape[3];
    const uint16_t output_height = output->shape[2];
    const uint16_t output_width = output->shape[3];
    const float pool_area = (float)(pool_size[0] * pool_size[1]);
    uint16_t batch;
    uint16_t channel;
    uint16_t output_row;
    uint16_t output_column;
    uint16_t pool_row;
    uint16_t pool_column;

    for (batch = 0; batch < batches; batch++) {
        for (channel = 0; channel < channels; channel++) {
            for (output_row = 0; output_row < output_height; output_row++) {
                for (output_column = 0; output_column < output_width; output_column++) {
                    float sum = 0.0f;

                    for (pool_row = 0; pool_row < pool_size[0]; pool_row++) {
                        for (pool_column = 0; pool_column < pool_size[1]; pool_column++) {
                            const int32_t input_row =
                                (int32_t)(output_row * stride[0] + pool_row)
                                - padding[0];
                            const int32_t input_column =
                                (int32_t)(output_column * stride[1] + pool_column)
                                - padding[1];

                            if (input_row >= 0 && input_column >= 0
                                && input_row < input_height
                                && input_column < input_width) {
                                const uint32_t input_index =
                                    ((uint32_t)batch * channels * input_height * input_width)
                                    + ((uint32_t)channel * input_height * input_width)
                                    + ((uint32_t)input_row * input_width)
                                    + (uint32_t)input_column;
                                sum += ((const float *)input->data)[input_index];
                            }
                        }
                    }

                    ((float *)output->data)[
                        ((uint32_t)batch * channels * output_height * output_width)
                        + ((uint32_t)channel * output_height * output_width)
                        + ((uint32_t)output_row * output_width)
                        + output_column] = sum / pool_area;
                }
            }
        }
    }
}

static void custom_avgpool2d_bwd(
    const aitensor_t *delta_out,
    const uint16_t pool_size[2],
    const uint16_t stride[2],
    const uint16_t padding[2],
    aitensor_t *delta_in)
{
    const uint16_t batches = delta_in->shape[0];
    const uint16_t channels = delta_in->shape[1];
    const uint16_t input_height = delta_in->shape[2];
    const uint16_t input_width = delta_in->shape[3];
    const uint16_t output_height = delta_out->shape[2];
    const uint16_t output_width = delta_out->shape[3];
    const float pool_area = (float)(pool_size[0] * pool_size[1]);
    uint16_t batch;
    uint16_t channel;
    uint16_t output_row;
    uint16_t output_column;
    uint16_t pool_row;
    uint16_t pool_column;

    custom_avgpool2d_init_zeros(delta_in);

    for (batch = 0; batch < batches; batch++) {
        for (channel = 0; channel < channels; channel++) {
            for (output_row = 0; output_row < output_height; output_row++) {
                for (output_column = 0; output_column < output_width; output_column++) {
                    const float gradient = ((const float *)delta_out->data)[
                        ((uint32_t)batch * channels * output_height * output_width)
                        + ((uint32_t)channel * output_height * output_width)
                        + ((uint32_t)output_row * output_width)
                        + output_column] / pool_area;

                    for (pool_row = 0; pool_row < pool_size[0]; pool_row++) {
                        for (pool_column = 0; pool_column < pool_size[1]; pool_column++) {
                            const int32_t input_row =
                                (int32_t)(output_row * stride[0] + pool_row)
                                - padding[0];
                            const int32_t input_column =
                                (int32_t)(output_column * stride[1] + pool_column)
                                - padding[1];

                            if (input_row >= 0 && input_column >= 0
                                && input_row < input_height
                                && input_column < input_width) {
                                const uint32_t input_index =
                                    ((uint32_t)batch * channels * input_height * input_width)
                                    + ((uint32_t)channel * input_height * input_width)
                                    + ((uint32_t)input_row * input_width)
                                    + (uint32_t)input_column;
                                ((float *)delta_in->data)[input_index] += gradient;
                            }
                        }
                    }
                }
            }
        }
    }
}

static void custom_avgpool2d_forward(ailayer_t *self)
{
    custom_avgpool2d_f32_t *layer =
        (custom_avgpool2d_f32_t *)self->layer_configuration;

    layer->forward_pool(
        &self->input_layer->result,
        layer->pool_size,
        layer->stride,
        layer->padding,
        &self->result);
}

static void custom_avgpool2d_backward(ailayer_t *self)
{
    custom_avgpool2d_f32_t *layer =
        (custom_avgpool2d_f32_t *)self->layer_configuration;

    layer->backward_pool(
        &self->output_layer->deltas,
        layer->pool_size,
        layer->stride,
        layer->padding,
        &self->deltas);
}

static uint32_t custom_avgpool2d_sizeof_trainmem(const ailayer_t *self)
{
    (void)self;
    return 0;
}

static void custom_avgpool2d_set_trainmem(ailayer_t *self, void *memory)
{
    (void)self;
    (void)memory;
}

static ailayer_t *custom_avgpool2d_initialize(
    custom_avgpool2d_f32_t *layer,
    ailayer_t *input_layer)
{
    if (layer == 0 || input_layer == 0
        || layer->pool_size[0] == 0 || layer->pool_size[1] == 0
        || layer->stride[0] == 0 || layer->stride[1] == 0
        || layer->padding[0] >= layer->pool_size[0]
        || layer->padding[1] >= layer->pool_size[1]) {
        return 0;
    }

    layer->base.layer_type = &custom_avgpool2d_type;
    layer->base.layer_configuration = layer;
    layer->base.settings = 0;
    layer->base.input_layer = input_layer;
    layer->base.output_layer = 0;
    layer->base.result.dim = 4;
    layer->base.result.shape = layer->result_shape;
    layer->base.deltas.dim = 4;
    layer->base.deltas.shape = input_layer->result.shape;
    layer->base.forward = custom_avgpool2d_forward;
    layer->base.backward = custom_avgpool2d_backward;
    layer->base.calc_result_shape = custom_avgpool2d_calc_result_shape;
    layer->base.calc_result_tensor_params = 0;
    layer->base.init_params = 0;
    layer->base.sizeof_trainmem = custom_avgpool2d_sizeof_trainmem;
    layer->base.set_trainmem = custom_avgpool2d_set_trainmem;
    layer->base.sizeof_fwdmem = 0;
    layer->base.sizeof_bwdmem = 0;
    layer->base.trainable_params_count = 0;
    input_layer->output_layer = &layer->base;

    custom_avgpool2d_calc_result_shape(&layer->base);

    return &layer->base;
}

ailayer_t *custom_avgpool2d_chw_f32_default(
    custom_avgpool2d_f32_t *layer,
    ailayer_t *input_layer)
{
    if (layer == 0) {
        return 0;
    }

    layer->base.result.dtype = aif32;
    layer->base.deltas.dtype = aif32;
    layer->channel_axis = 1;
    layer->forward_pool = custom_avgpool2d_fwd;
    layer->backward_pool = custom_avgpool2d_bwd;

    return custom_avgpool2d_initialize(layer, input_layer);
}
