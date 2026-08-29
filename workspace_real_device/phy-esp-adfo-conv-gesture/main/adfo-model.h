#ifndef ADFO_MODEL
#define ADFO_MODEL

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "aifes.h"


// Function declarations
void AIfES_mnist_run(aitensor_t*);
void eval_test_data(float *output_data, int test_size);
void calc_success_rate(float *output_data, int test_size);
void load_params(aitensor_t*, aimodel_t*);
void init_model(aimodel_t *, aiopti_t *, void*, void*);
void train_model(aimodel_t *, aiopti_t *);
void get_parameters(aitensor_t*, aimodel_t *);
void create_segments(float*, int, float**, int*, char**);
float calculate_percentile(float*, int, int);
void print_tensor(aitensor_t*);
void flatten_array_to_tensor(aitensor_t*, float*, int*);
void aggregate_arrays(float*);
void show_labels();
void free_pointer_array(void**);

#endif