/* Emulation example.
   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "nvs_flash.h"
#include <unistd.h>

#include <esp_event.h>
#include <esp_log.h>
#include <esp_system.h>
#include <sys/param.h>
#include <esp_http_server.h>
#include "esp_netif.h"
#include "esp_eth.h"
#include "esp_random.h"
#include <string.h>
#include <time.h>
#include "esp_timer.h"


#include "protocol_examples_common.h"
#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include <lwip/netdb.h>
#include "adfo-model.h"
#include "aifes.h"
#include "adfo-com.h"

#include "esp_clk_tree.h"
#include "soc/clk_tree_defs.h"
#include "esp_log.h"

#define STACK_SIZE 8192
#define TASK_PRIORITY 1

SemaphoreHandle_t addMutex;
int total_params=-1;

void app_main(void)
{   
    // Create tensor to copy model parameters
    aitensor_t local_params[NUMB_TRAINABLE_PARAMETERS];
    aitensor_t received_params[NUMB_TRAINABLE_PARAMETERS];
    aimodel_t local_model;
    aiopti_t model_optimizer;
    void *local_model_parameter_memory = NULL;
    void *local_model_working_memory = NULL;
    addMutex = xSemaphoreCreateMutex();

    int free_heap = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    printf("HEAP Size in the beginning: %d\n", free_heap);

    // Allocate address struct
    adresses_available = (struct adresses *) malloc(MAX_DEVICES * sizeof(struct adresses));

    if (adresses_available == NULL) {
        LOG("Failed to allocate adresses_available in PSRAM!");
    }
    memset(adresses_available, 0, MAX_DEVICES * sizeof(struct adresses));

#ifdef DATASET
    device_id = 10+DATASET;
#else
    LOG("DATASET is not defined!\n");
#endif
	LOG("This Device's ID: %ld!\n", device_id);


    // Start Ethernet and TCP server task
    register_ethernet();
    sleep(20);
    xTaskCreatePinnedToCore(tcp_server_task, "tcp_server", STACK_SIZE, (void*)AF_INET, TASK_PRIORITY, NULL, 0);
    LOG("TCP Server Created!\n");

    LOG("Wait for 60 seconds until all devices are ready...\n");
    sleep(60);
    
    int addr_count=0;
    while(addr_count == 0){
        // Check other ports and print out found devices
        find_available_ports();

#if MAXIMUM_NUM_DEVICES > 0
        for(int device_num=0; device_num<MAXIMUM_NUM_DEVICES; device_num++){
            adresses_available[device_num].device_id=10+device_num; 
            snprintf(adresses_available[device_num].addr, 13, "172.17.0.%d", device_num+10);
        }
        LOG("MAXIMUM_NUM_DEVICES is %d", MAXIMUM_NUM_DEVICES);
#else
        LOG("MAXIMUM_NUM_DEVICES is NULL");
#endif

        LOG("------------------ FOUND PORTS and DEVICES -----------------------\n");
        for(addr_count=0;addr_count<MAX_DEVICES;addr_count++){
            if (adresses_available[addr_count].addr[0] == '\0') break; 
            LOG("Addr: %s, Device ID: %lu\n", adresses_available[addr_count].addr, adresses_available[addr_count].device_id);
            if(adresses_available[addr_count].device_id == device_id) { device_order = addr_count; };
        }
        LOG("-------------------------------------------------------------------\n");
        //sleep(2);
        LOG("Order of this device %d\n", device_order);
    }

    // Allocate space for global arrays
    division_array = (int *) malloc(RECEIVE_ARRAY_SIZE * sizeof(int));
    true_labels_array = (int *) malloc(100 * sizeof(int));
    pred_labels_array = (int *) malloc(100 * sizeof(int));

    if (division_array == NULL) {
        LOG("Failed to allocate division_array in PSRAM\n");
        abort();
    }
    memset(division_array, 0, RECEIVE_ARRAY_SIZE * sizeof(int));
    
    free_heap = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    printf("HEAP Size before init model: %d\n", free_heap);

    show_labels();
    init_model(&local_model, &model_optimizer, local_model_parameter_memory, local_model_working_memory);

    bool end = false;
    for(int round=0; round<COMM_ROUNDS;round++){
        free_heap = heap_caps_get_free_size(MALLOC_CAP_8BIT);
        printf("HEAP Size at the beginning of round: %d\n", free_heap);
        current_round = round;
        //sleep(15); // Imitate data collection
        LOG("####### Round: %d\n", round);
        // train model for 3 epochs
        train_model(&local_model, &model_optimizer);
        train_model(&local_model, &model_optimizer);
        train_model(&local_model, &model_optimizer);

        get_parameters(&local_params[0], &local_model);
        //sleep(1);

        int param_counts_arr[NUMB_TRAINABLE_PARAMETERS];
        total_params = count_total_params(&local_params[0], &param_counts_arr[0]);
        float* flatten = (float *)malloc(total_params * sizeof(float));
        flatten_tensor_to_array(&local_params[0], flatten, param_counts_arr);

        free_heap = heap_caps_get_free_size(MALLOC_CAP_8BIT);
        printf("HEAP Size before creating segments: %d\n", free_heap);
        char *bitArray_send[SEGMENT_COUNT];
        int selected_devices[SHARE_COUNT];
        float *segments[SEGMENT_COUNT];
        int segment_sizes[SEGMENT_COUNT];
        create_segments(flatten, total_params, segments, segment_sizes, bitArray_send);
        free_heap = heap_caps_get_free_size(MALLOC_CAP_8BIT);
        printf("HEAP Size after creating segments: %d\n", free_heap);

        INFO("Segments Created!");
        for(int i=0; i<SEGMENT_COUNT; i++){
            INFO("Segment %d size : %d\n", i, segment_sizes[i]);
        }

        memset(selected_devices, -1, sizeof(selected_devices));
        for (int share=0; share < SHARE_COUNT; share++) {
            free_heap = heap_caps_get_free_size(MALLOC_CAP_8BIT);
            printf("HEAP Size  in send loop: %d\n", free_heap);
            printf("Send progress : %d/%d\n", share, SHARE_COUNT);
            int selected_segment;
            if(RANDOM_SEGMENTATION){
                selected_segment = ((uint32_t) esp_random()) % (SEGMENT_COUNT);
            } else if(ONLY_2) {
                selected_segment = SEGMENT_COUNT - 1;
            } else {
                selected_segment = select_segment(segments, segment_sizes);
            }
            for(int i=0; i<100; i++){
                ANALYZE("Bitmap: %d, flatten - value %d: %f\n", READ_BIT(bitArray_send[selected_segment], i), i, flatten[i]);
            }
            for(int i=0; i<10; i++){
                ANALYZE("Shared Segment: %f\n", segments[selected_segment][i]);
            }
            int index = ((uint32_t) esp_random()) % (addr_count);

            while(adresses_available[index].device_id == device_id) { 
                index = ((uint32_t) esp_random()) % (addr_count);
            }
            selected_devices[share] = index;
            LOG("Send Segment %d to index: %d\n", selected_segment, index);
            send_array(index, segments[selected_segment], bitArray_send[selected_segment], segment_sizes[selected_segment]);
            printf("After send array!\n");
        }
        free_heap = heap_caps_get_free_size(MALLOC_CAP_8BIT);
        printf("HEAP Size before free nested arrays: %d\n", free_heap);
        free_pointer_array((void**)segments);
        free_pointer_array((void**)bitArray_send);
        free_heap = heap_caps_get_free_size(MALLOC_CAP_8BIT);
        printf("HEAP Size after free nested arrays: %d\n", free_heap);
        
        for(int i=LOG_PARAM; i<LOG_PARAM+50; i++){
            ANALYZE("total_receives - value %d: %f\n", i, total_receives[i]);
        }

        time_t start_time, current_time;
        start_time = time(NULL);
        LOG("Waiting until 5 Segments received...\n");
        do {
            current_time = time(NULL);
            sleep(1);
            if(difftime(current_time, start_time) > 90) {
                //end = true;
                break;
            }
        } while (received_segments < SHARE_COUNT-1);
        xSemaphoreTake(addMutex, portMAX_DELAY);

        INFO("%d Segments received!\n", received_segments);

        total_params = count_total_params(&local_params[0], &param_counts_arr[0]);
        aggregate_arrays(&flatten[0]);
        clear_received_array();
        received_segments = 0;
        xSemaphoreGive(addMutex);

        flatten_array_to_tensor(&local_params[0], &flatten[0], param_counts_arr);
        load_params(&local_params[0], &local_model);
        free(flatten);
        //sleep(1);
        LOG("Total sent communication cost %ld\n", comm_cost);
        if(end) {break;}
    }
}
