#include "freertos/FreeRTOS.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include "esp_random.h"

#include "aifes.h"
#include "adfo-model.h"
#include "adfo-com.h"

#include "Gesture_test_data.h"
#include "Gesture_test_data_label.h"
#include "Gesture_train_data.h"
#include "Gesture_train_data_label.h"
#include "Gesture_val_data.h"
#include "Gesture_val_data_label.h"

#define BATCH_SIZE          50
#define EPOCHS              1

#define image_size            8
#define Gesture_OUTPUTS       4
#define Gesture_LAYER_CNT     10

aiopti_adam_f32_t adam_opti = AIOPTI_ADAM_F32(0.01f, 0.9f, 0.999f, 1e-8f);

uint16_t input_layer_shape[] = {1, 1, image_size, image_size};
ailoss_crossentropy_f32_t model_loss;

ailayer_input_f32_t		input_layer	        = AILAYER_INPUT_F32_A(4, input_layer_shape);
ailayer_conv2d_f32_t    conv1               = AILAYER_CONV2D_F32_A(8, HW(3, 3), HW(1, 1), HW(1,1), HW(1, 1)); //8x8x8
custom_avgpool2d_f32_t avg1                = CUSTOM_AVGPOOL2D_F32_A(HW(2, 2), HW(2, 2), HW(0, 0)); //8x4x4
ailayer_relu_f32_t		relu1 	            = AILAYER_RELU_F32_A();

ailayer_conv2d_f32_t    conv2               = AILAYER_CONV2D_F32_A(16, HW(3, 3), HW(1, 1), HW(1,1), HW(1, 1)); //16x4x4
custom_avgpool2d_f32_t avg2                = CUSTOM_AVGPOOL2D_F32_A(HW(2, 2), HW(2, 2), HW(0, 0)); //16x2x2
ailayer_relu_f32_t		relu2 	            = AILAYER_RELU_F32_A();

ailayer_flatten_t       flat3               = AILAYER_FLATTEN_F32_A();

ailayer_dense_f32_t		dense_layer4	    = AILAYER_DENSE_F32_A(32);
ailayer_relu_f32_t		relu4 	            = AILAYER_RELU_F32_A();
ailayer_dense_f32_t		dense_layer5	    = AILAYER_DENSE_F32_A(Gesture_OUTPUTS);
ailayer_softmax_f32_t	softmax_layer5	    = AILAYER_SOFTMAX_F32_A();

void aggregate_arrays(float* local_array){
    for(int i=LOG_PARAM; i<LOG_PARAM+50; i++){
        printf("idx: %d, Local: %f, Received Arr: %f, divison value: %d\n", i, local_array[i], total_receives[i], division_array[i]);
    }

    for(int i=0;i<total_params;i++){
        if(division_array[i] == 0) { continue; }
        if(SB_AGGR) {
            int local_success = (int)(latest_success*1000);
            local_array[i] = ((local_array[i] * local_success) + total_receives[i]) / (division_array[i] + local_success);
        } else {
            local_array[i] = (local_array[i] + total_receives[i]) / (division_array[i]+1);
        }
    }
    LOG("Aggregated with %d tensors.\n", received_segments);
}

void show_labels() {
    int training_size = sizeof(Gesture_training_data) / sizeof(Gesture_training_data[0]) / (image_size*image_size);
    int labels[4] = {0};

    for (int i = 0; i < training_size*4; i++) {
        if (Gesture_training_data_label[i] == 1.0f) {
            labels[i%4]++;
        }
    }

    INFO("Labels On Device:\n");
    for (int i=0; i<4; i++){
        INFO("Label %d: %d\n", i, labels[i]);
    }
}

bool valueinarray(int val, int *arr, size_t n) {
    for(size_t i = 0; i < n; i++) {
        if(arr[i] == val)
            return true;
    }
    return false;
}

void copy_tensors(const aitensor_t *from_tensor, aitensor_t *to_tensor, int *params_cnt){
    int tensor_elements = aimath_tensor_elements(from_tensor);
    *params_cnt += tensor_elements;

    *to_tensor = *from_tensor;
}

void copy_tensor_to_float(const aitensor_t *from_tensor, float *to_array, int *params_cnt)
{
    static int count = 0;
    memcpy(to_array+(*params_cnt),
           from_tensor->data,
           aimath_tensor_elements(from_tensor) * (from_tensor->dtype->size));
    (*params_cnt) = (*params_cnt) + aimath_tensor_elements(from_tensor);
    count++;
    DEBUG("Copy count: %d\n", count);
}

void copy_float_to_tensor(const float *from_array, aitensor_t *to_tensor, int *params_cnt)
{
    memcpy(to_tensor->data,
           from_array+(*params_cnt),
           aimath_tensor_elements(to_tensor) * (to_tensor->dtype->size));
    (*params_cnt) = (*params_cnt) + aimath_tensor_elements(to_tensor);
}

void flatten_array_to_tensor(aitensor_t* tensor, float* flatten, int* total_params_arr){
    INFO("\n-------------------- Flatten array to tensor --------------------\n");
    int cnt = 0;
    for(int i=0; i<NUMB_TRAINABLE_PARAMETERS;i++){
        int tensor_elements = total_params_arr[i];
        DEBUG("tensor elements: %d\n", tensor_elements);
        for(int j=0; j<tensor_elements;j++){
            ((float *)(&tensor[i])->data)[j] = flatten[cnt++];
        }
    }
}

void print_tensor(aitensor_t* tensor){
    int max_print=0;
    uint16_t i, j;

    if (tensor->dim == 1) {
        for (j = 0; j < tensor->shape[0]; j++)
        {
            ANALYZE("%f;", ((float *)tensor->data)[j]);
        }
    } else if (tensor->dim == 2) {
        for (i = 0; i < tensor->shape[0]; i++)
        {
            for (j = 0; j < tensor->shape[1]; j++)
            {
                if(max_print > 1000 && max_print < 1050){
                    ANALYZE("idx %d: %f\n", max_print, ((float *)tensor->data)[i * tensor->shape[1] + j]);
                }
                max_print++;
            }
        }
    }
    ANALYZE("\n");

}

aitensor_t *aialgo_forward_model_print(aimodel_t *model, aitensor_t *input_data)
{
	uint16_t i;
	ailayer_t *layer_ptr = model->input_layer;

	model->input_layer->result.data = input_data->data;
	for(i = 0; i < model->layer_count; i++)
	{
	    if(layer_ptr->layer_type == ailayer_batch_norm_type)
        {
            ailayer_batch_norm_t *layer = (ailayer_batch_norm_t *)(layer_ptr->layer_configuration);
            aitensor_t *moving_means = &(layer->moving_means);
            aitensor_t *moving_variances = &(layer->moving_variances);
            aitensor_t *means = layer->means;
            aitensor_t *variances = layer->variances;
            aitensor_t *betas = &(layer->betas);
            aitensor_t *gammas = &(layer->gammas);
            printf("Means Forward Mode\n");
            print_tensor(means);
            printf("-----------------------\n");
            printf("variances Forward Mode\n");
            print_tensor(variances);
            printf("-----------------------\n");
        }
		layer_ptr->forward(layer_ptr);
		if(layer_ptr->layer_type == ailayer_batch_norm_type)
        {
            ailayer_batch_norm_t *layer = (ailayer_batch_norm_t *)(layer_ptr->layer_configuration);
            aitensor_t *moving_means = &(layer->moving_means);
            aitensor_t *moving_variances = &(layer->moving_variances);
            aitensor_t *means = layer->means;
            aitensor_t *variances = layer->variances;
            aitensor_t *betas = &(layer->betas);
            aitensor_t *gammas = &(layer->gammas);
            printf("Means Forward Mode\n");
            print_tensor(means);
            printf("-----------------------\n");

            printf("moving_means Forward Mode\n");
            print_tensor(moving_means);
            printf("-----------------------\n");

            printf("variances Forward Mode\n");
            print_tensor(variances);
            printf("-----------------------\n");
            //print_tensor(&layer_ptr->result);
        }
		layer_ptr = layer_ptr->output_layer;
	}
	return &(model->output_layer->result);
}

void create_bitmap(aitensor_t* params, float val){
    INFO("\n-------------------- Create Bitmap --------------------\n");
    int param_loc = 0;
    for(int i=0;i<NUMB_TRAINABLE_PARAMETERS;i++){
        int tensor_elements = aimath_tensor_elements(&params[i]);
        for(int j=0;j<tensor_elements;j++){
            float param = *(float*)((&params[i])->data+j);
            param_loc++;
        }
    }
    for(int i=0;i<BIT_ARRAY_SIZE;i++){
        ANALYZE("%d",READ_BIT(bitArray, i));
    }
    ANALYZE("\n");
}

void free_pointer_array(void **array) {
    for (int i = 0; i < SEGMENT_COUNT; i++) {
        free(array[i]);
    }
}

void create_segments(float* all_params, int total_params, float **segments, int* segment_sizes, char **bitArrays){
    LOG("\n-------------------- Create %d Segments, with total params : %d\n", SEGMENT_COUNT, total_params);
    
    float threshold_values[SEGMENT_COUNT];
    int segment_indexes[SEGMENT_COUNT];
    // Find threshold values
    for(int i=0; i<SEGMENT_COUNT; i++){
        threshold_values[i] = calculate_percentile(all_params, 100*i/(SEGMENT_COUNT),total_params);
        if(SEGMENT_COUNT == 1) {threshold_values[i] = calculate_percentile(all_params, 0,total_params);}
        INFO("Theshold %d value: %f\n", i, threshold_values[i]);
        segment_sizes[i] = 0;
        segment_indexes[i] = 0;
    }
    // Calculate how many parameters needed in each segment
    for(int i=0; i<total_params; i++){
        for (int j = SEGMENT_COUNT-1; j >= 0; j--) {
            if(RANDOM_SEGMENTATION){
                int choice = i % 3;
                if(choice == j){
                    segment_sizes[j] += 1;
                    break;                
                }
            } else {
                if (fabs(all_params[i]) >= threshold_values[j]) {
                    segment_sizes[j] += 1;
                    break; 
                }
            }
        }
    }

    // Allocate memory for each segment
    for (int i = 0; i < SEGMENT_COUNT; i++) {
        // Malloc as below so the array is a mulitple of chunk size
        segments[i] = (float*)malloc(ceil((float)segment_sizes[i]/CHUNK_SIZE) * CHUNK_SIZE * sizeof(float));
        memset(segments[i], 0, ceil((float)segment_sizes[i]/CHUNK_SIZE) * CHUNK_SIZE * sizeof(float));
        if (!segments[i]) {
            LOG("Memory allocation failed!\n");
            exit(-1);
        }
    }
    
    // Assign each parameter to a segment
    for(int i=0; i<total_params; i++){
        for (int j = SEGMENT_COUNT-1; j >= 0; j--) {
            if(RANDOM_SEGMENTATION){
                int choice = i % 3;
                if(choice == j){
                    segments[j][segment_indexes[j]] = all_params[i];   
                    segment_indexes[j] += 1;
                    break;                
                }
            } else {
                if (fabs(all_params[i]) >= threshold_values[j]) {
                    segments[j][segment_indexes[j]] = all_params[i];
                    segment_indexes[j] += 1;
                    break;
                }
            }
        }
    }
    
    for(int i=0; i<SEGMENT_COUNT; i++){
        char *temp_bitarray = (char*)malloc((BIT_ARRAY_SIZE / BYTE_SIZE * sizeof(char)));
        memset(temp_bitarray, 0, (BIT_ARRAY_SIZE / BYTE_SIZE * sizeof(char)));
        for(int j=0; j<total_params; j++){
            if(RANDOM_SEGMENTATION){
                int choice = j % 3;
                if(choice == i){
                    SET_BIT(temp_bitarray, j);                    
                }
            } else {
                if(i == SEGMENT_COUNT -1){
                    if(fabs(all_params[j]) >= threshold_values[i]){
                        SET_BIT(temp_bitarray, j);
                    }
                } else {
                    if(fabs(all_params[j]) >= threshold_values[i] && fabs(all_params[j]) < threshold_values[i+1]){
                        SET_BIT(temp_bitarray, j);
                    }
                }
            }
        }
        temp_bitarray[(BIT_ARRAY_SIZE / BYTE_SIZE)-1] = 0;
        bitArrays[i] = temp_bitarray;
    }
    // }
}

int compare(const void *a, const void *b) {
    float diff = (*(float *)a) - (*(float *)b);
    return (diff > 0) - (diff < 0); // Returns 1, 0, or -1
}

float calculate_percentile(float* arr, int q, int size){
    float *temp_array = (float *)malloc(size * sizeof(float));
    if (!temp_array) {
        LOG("Memory allocation failed!\n");
        exit(-1);
    }

    // Copy elements using memcpy
    memcpy(temp_array, arr, size * sizeof(float));

    DEBUG("Calculate Percentile, percentage %d!\n", q);
    if (q < 0 || q > 100) {
        fprintf(stderr, "Percentile q must be between 0 and 100.\n");
        return -1;
    }
    
    // Convert to absolute values
    for(int i=0;i<size;i++){
        temp_array[i] = fabs(temp_array[i]);
    }

    // Sort the data array
    qsort(temp_array, size, sizeof(float), compare);

    // Calculate the index for the q-th percentile
    float position = (q / 100.0) * (size - 1);
    size_t lower_index = (size_t)position;
    size_t upper_index = lower_index + 1;
    float fraction = position - lower_index;

    if (upper_index < size) {
        float ret_val = temp_array[lower_index] + fraction * (temp_array[upper_index] - temp_array[lower_index]);
        printf("before temp array free\n");
        free(temp_array);
        return ret_val;
    } else {
        float ret_val = temp_array[lower_index];
        printf("other temp array free\n");
        free(temp_array);
        return ret_val; // When position is the last index
    }
}

void eval_test_data(float *output_data, int test_size){
    int r = (rand() % (test_size - 5)/4)*4; // Pick a random number up to test size as multipliers of 4
    DEBUG("\n----- Test data starting from %d, Predicted Values | Target Values -----\n", r);
    for (int i=r; i < r+4; i++){
        DEBUG("\t%f\t%f |", output_data[i],Gesture_test_data_label[i]);
        DEBUG("\t%f\t%f |", output_data[i+4],Gesture_test_data_label[i+4]);
        DEBUG("\t%f\t%f |", output_data[i+8],Gesture_test_data_label[i+8]);
        DEBUG("\t%f\t%f |", output_data[i+12],Gesture_test_data_label[i+12]);
        DEBUG("\t%f\t%f |", output_data[i+16],Gesture_test_data_label[i+16]);
        DEBUG("\n");
    }
}

void calc_success_rate(float *output_data, int test_size){
    int predicted_index, target_index;
    float success=0;
    for (int i=0; i < test_size*4; i+=4){
        predicted_index=i;
        target_index=i;

        for (int j=0; j<4;j++){
            if (output_data[i+j] >= output_data[predicted_index]){
                    predicted_index=i+j;
            }
        }
        for (int j=0; j<4;j++){
            if (Gesture_val_data_label[i+j] >= Gesture_val_data_label[target_index]){
                    target_index=i+j;
            }
        }

        if (target_index == predicted_index) success++;
    }
    float success_rate = success / test_size;
    latest_success = success_rate;
    LOG("Total Success: %f, Test Size: %d, Success rate: %f\n", success, test_size, success_rate);
}

void calc_success_rate_val(float *output_data, int test_size){
    int predicted_index, target_index;
    float success=0;
    for (int i=0; i < test_size*4; i+=4){
        predicted_index=i;
        target_index=i;

        for (int j=0; j<4;j++){
            if (output_data[i+j] >= output_data[predicted_index]){
                    predicted_index=i+j;
            }
        }
        for (int j=0; j<4;j++){
            if (Gesture_test_data_label[i+j] >= Gesture_test_data_label[target_index]){
                    target_index=i+j;
            }
        }
        if (target_index == predicted_index) success++;
    }
    float success_rate = success / test_size;
    latest_success = success_rate;
    LOG("Total Success: %f, Test Size: %d, Success rate: %f\n", success, test_size, success_rate);
    printf("Outside macro\n");
}


void init_model(aimodel_t *local_model, aiopti_t *model_optimizer, void* parameter_memory, void* working_memory)
{
    INFO("\n-------------------- Init Model -------------------- \n");
    ailayer_t *x;

    local_model->input_layer = ailayer_input_f32_default(&input_layer);
    
    x = ailayer_conv2d_chw_f32_default(&conv1, local_model->input_layer);
    x = custom_avgpool2d_chw_f32_default(&avg1, x);
    x = ailayer_relu_f32_default(&relu1, x);

    x = ailayer_conv2d_chw_f32_default(&conv2, x);
    x = custom_avgpool2d_chw_f32_default(&avg2, x);
    x = ailayer_relu_f32_default(&relu2, x);

    x = ailayer_flatten_f32_default(&flat3, x);
    
    x = ailayer_dense_f32_default(&dense_layer4, x);
    x = ailayer_relu_f32_default(&relu4, x);
    x = ailayer_dense_f32_default(&dense_layer5, x);
    x = ailayer_softmax_f32_default(&softmax_layer5, x);
    
    local_model->output_layer = x;

    local_model->loss = ailoss_crossentropy_f32_default(&model_loss, local_model->output_layer); // ailoss_crossentropy_f32_default

    aialgo_compile_model(local_model);

    //------------------Memory allocation----------------------------
    // Allocate memory for result and temporal data
    uint32_t model_parameter_memory_size = aialgo_sizeof_parameter_memory(local_model);
    parameter_memory = malloc(model_parameter_memory_size);
    DEBUG("model_parameter_memory_size: %li\n", model_parameter_memory_size);

    // Distribute the memory to the trainable parameters of the model
    aialgo_distribute_parameter_memory(local_model, parameter_memory, model_parameter_memory_size);
    srand(42+device_order); //26
    aialgo_initialize_parameters_model(local_model);

    // -------------------------------- Define the optimizer for training ---------------------
    model_optimizer = aiopti_adam_f32_default(&adam_opti);

    // -------------------------------- Allocate and schedule the working memory for training ---------

    uint32_t model_training_memory_size = aialgo_sizeof_training_memory(local_model, model_optimizer);
    DEBUG("model_training_memory_size: %li\n", model_training_memory_size);

    working_memory = malloc(model_training_memory_size);

    if(working_memory == NULL)
    {
        LOG("Not enough memory available for parameter memory of the neural network.\n");
        return;
    }

    // Schedule the memory over the model
    aialgo_schedule_training_memory(local_model, model_optimizer, working_memory, model_training_memory_size);
    aialgo_init_model_for_training(local_model, model_optimizer);
    INFO("-----------------------------------------------------\n\n");

    // Params after init
    ailayer_t *layer_ptr = local_model->input_layer;

    for (int l = 0; l < local_model->layer_count; l++)
    {
        for (int j = 0; j < layer_ptr->trainable_params_count; j++)
        {
            if (aimath_tensor_elements(layer_ptr->trainable_params[j]) > 0)
            {
                print_tensor(layer_ptr->trainable_params[j]);
            }
        }
        layer_ptr = layer_ptr->output_layer;
    }

    return;
}

void train_model(aimodel_t *local_model, aiopti_t *model_optimizer){
    // Pick only 'subset_size' of training dataset to train in this round
    int subset_size = SUBSET_SIZE;
    float *subset_training_data = (float*)malloc(subset_size * sizeof(float) * image_size * image_size); // Dont forget to free
    float *subset_training_data_label = (float*)malloc(subset_size * sizeof(float) * 4);
    int selected_index[subset_size] = {};

    for (int i=0; i<subset_size; i++){
        int index = ((uint32_t) esp_random()) % 600;
        while(valueinarray(index, selected_index, subset_size)){
            index = ((uint32_t) esp_random()) % 600;
        }
        selected_index[i] = index;
        int starting_point = index * (image_size  * image_size);
        for(int j = 0; j < (image_size  * image_size); j++){
            subset_training_data[i * (image_size  * image_size) + j] = Gesture_training_data[starting_point + j];
        }
        int starting_point_label = index * 4;
        for(int j = 0; j < 4; j++){
            subset_training_data_label[i * 4 + j] = Gesture_training_data_label[starting_point_label + j];
        }
    }
    INFO("Subset creation completed with subset size %d!\n", subset_size);

    // ---------------------------------- Input Sizes ---------------------------------------
    int training_size = subset_size;
    // int training_size = sizeof(Gesture_training_data) / sizeof(Gesture_training_data[0]) / (image_size*image_size);
    int test_size = sizeof(Gesture_val_data) / sizeof(Gesture_val_data[0]) / (image_size*image_size);

    INFO("Training Data Size: %d\n", training_size);
    INFO("Test Data Size for: %d\n", test_size);
    INFO("Batch Size: %d\n", BATCH_SIZE);

    uint32_t i;
    uint16_t input_shape[] = {training_size, 1, image_size, image_size}; // [sample count, channels, height, width]
    aitensor_t input_tensor = AITENSOR_4D_F32(input_shape, subset_training_data);
    uint16_t target_shape[] = {training_size, 4};
	aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, subset_training_data_label);
	

    uint16_t test_shape[] = {test_size, 1, image_size, image_size}; // [sample count, channels, height, width]
	aitensor_t test_tensor = AITENSOR_4D_F32(test_shape, Gesture_val_data);

    float output_data[test_size*4];
    memset(output_data, 0, sizeof(output_data));
	uint16_t output_shape[2] = {test_size, 4};
	aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);


    LOG("\n-------------------- Train Model --------------------\n");

	float loss;
    printf("Learning Rate in the round %d: %f\n", current_round, adam_opti.learning_rate);
    if((current_round == (COMM_ROUNDS*0.4) || current_round == (COMM_ROUNDS*0.65) || current_round == (COMM_ROUNDS*0.75)) && ADASTAIR){
        adam_opti.learning_rate = adam_opti.learning_rate / 2;
    }
    model_optimizer = aiopti_adam_f32_default(&adam_opti);
    
    for(i = 0; i < EPOCHS; i++)
    {
        // One epoch of training. Iterates through the whole data once
        aialgo_train_model(local_model, &input_tensor, &target_tensor, model_optimizer, BATCH_SIZE);

        aialgo_calc_loss_model_f32(local_model, &input_tensor, &target_tensor, &loss);
        LOG("Epoch %ld: loss: %.6f\n", i+1, loss/(float)training_size);
        LOG("Evaluation after training!\n");
        aialgo_inference_model(local_model, &test_tensor, &output_tensor);
        calc_success_rate(output_data, test_size);
    }

    
    //calc_success_rate(output_data, test_size);
    LOG("-------------------- Training Done --------------------\n");
    free(subset_training_data);
    free(subset_training_data_label);

}

void get_parameters(aitensor_t* local_params, aimodel_t* local_model){
    LOG("\n-------------------- Get Parameters -------------------- \n");

    ailayer_t *layer_ptr = local_model->input_layer;
    int params_cnt = 0;
    int tensor_cnt = 0;
    aitensor_t *moving_means;
    aitensor_t *moving_variances;
    aitensor_t *means;
    aitensor_t *variances;


    for (int l = 0; l < local_model->layer_count; l++)
    {
        if(layer_ptr->layer_type == ailayer_batch_norm_type)
        {
            ailayer_batch_norm_t *layer = (ailayer_batch_norm_t *)(layer_ptr->layer_configuration);
            moving_means = &(layer->moving_means);
            moving_variances = &(layer->moving_variances);
            means = layer->means;
            variances = layer->variances;

            copy_tensors(means, &local_params[tensor_cnt++], &params_cnt);
            copy_tensors(variances, &local_params[tensor_cnt++], &params_cnt);
            copy_tensors(moving_means, &local_params[tensor_cnt++], &params_cnt);
            copy_tensors(moving_variances, &local_params[tensor_cnt++], &params_cnt);
        }
        for (int j = 0; j < layer_ptr->trainable_params_count; j++)
        {
            if (aimath_tensor_elements(layer_ptr->trainable_params[j]) > 0)
            {
                ANALYZE("Get Parameters layer: %d\n", l);
                // print_tensor(layer_ptr->trainable_params[j]);
                copy_tensors(layer_ptr->trainable_params[j], &local_params[tensor_cnt++], &params_cnt);
                // print_tensor(&local_params[tensor_cnt - 1]);
            }
            else
            {
                params_cnt++;
            }
        }
        layer_ptr = layer_ptr->output_layer;
    }
}

void load_params(aitensor_t *trainable_parameters, aimodel_t *local_model){
    INFO("----- Load Model----- \n");

    ailayer_t *layer_ptr = local_model->input_layer;
    int params_cnt = 0;
    int tensor_cnt = 0;
    aitensor_t *moving_means;
    aitensor_t *moving_variances;
    aitensor_t *means;
    aitensor_t *variances;

    int test_size = sizeof(Gesture_test_data) / sizeof(Gesture_test_data[0]) / (image_size*image_size);

    uint16_t test_shape[] = {test_size, 1, image_size, image_size}; // [sample count, channels, height, width]
	aitensor_t test_tensor = AITENSOR_4D_F32(test_shape, Gesture_test_data);

    float output_data[test_size*4];
	uint16_t output_shape[2] = {test_size, 4};
	aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);
    
    for (int l = 0; l <local_model->layer_count; l++)
    {
        if(layer_ptr->layer_type == ailayer_batch_norm_type)
        {
            ailayer_batch_norm_t *layer = (ailayer_batch_norm_t *)(layer_ptr->layer_configuration);
            moving_means = &(layer->moving_means);
            moving_variances = &(layer->moving_variances);
            means = layer->means;
            variances = layer->variances;

            copy_tensors(&trainable_parameters[tensor_cnt++], means, &params_cnt);
            copy_tensors(&trainable_parameters[tensor_cnt++], variances, &params_cnt);
            copy_tensors(&trainable_parameters[tensor_cnt++], moving_means, &params_cnt);
            copy_tensors(&trainable_parameters[tensor_cnt++], moving_variances, &params_cnt);
        }
        for (int j = 0; j < layer_ptr->trainable_params_count; j++)
        {
            if (aimath_tensor_elements(layer_ptr->trainable_params[j]) > 0)
            {
                ANALYZE("LOAD PARAMS: \n");
                // print_tensor(layer_ptr->trainable_params[j]);
                copy_tensors(&trainable_parameters[tensor_cnt++], layer_ptr->trainable_params[j], &params_cnt);
                // print_tensor(layer_ptr->trainable_params[j]);
            }
            else
            {
                params_cnt++;
            }
        }
        layer_ptr = layer_ptr->output_layer;
    }

    LOG("\n----- Evaluate test data with loaded model: ----- \n");

    aialgo_inference_model(local_model, &test_tensor, &output_tensor);
	eval_test_data(output_data, test_size);
    calc_success_rate_val(output_data, test_size);
    LOG("### Success Rate: %f\n", latest_success);
}
