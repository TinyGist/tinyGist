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

#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include <lwip/netdb.h>
#include "adfo-model.h"
#include "aifes.h"
#include "adfo-com.h"

#define STACK_SIZE 8192
#define TASK_PRIORITY 1

SemaphoreHandle_t addMutex;
int total_params=-1;

void app_main(void)
{   
    division_array = (int *) malloc(RECEIVE_ARRAY_SIZE * sizeof(int));
    if (division_array == NULL) {
        printf("Failed to allocate division_array in PSRAM\n");
        abort();
    }
    memset(division_array, 0, RECEIVE_ARRAY_SIZE * sizeof(int));
    // Create tensor to copy model parameters
    aitensor_t local_params[NUMB_TRAINABLE_PARAMETERS];
    aitensor_t received_params[NUMB_TRAINABLE_PARAMETERS];
    aimodel_t local_model;
    aiopti_t model_optimizer;
    void *local_model_parameter_memory = NULL;
    void *local_model_working_memory = NULL;
    addMutex = xSemaphoreCreateMutex();

#ifdef DATASET
	device_id = 40+DATASET;
#else
	LOG("DATASET is not defined");
#endif
    LOG("This Device's ID: %ld!\n", device_id);


    // Start Ethernet and TCP server task
    register_wifi();
    sleep(6);
    xTaskCreatePinnedToCore(tcp_server_task, "tcp_server", STACK_SIZE, (void*)AF_INET, TASK_PRIORITY, NULL, 0);
    LOG("TCP Server Created!\n");
    LOG("Wait for 60 seconds until all devices are ready...\n");
    sleep(60);
    
    int addr_count=0;

    while(addr_count < 1){
        // Check other ports and print out found devices
        find_available_ports();

        adresses_available[0].device_id=40;
        snprintf(adresses_available[0].addr, 13, "%s", "192.168.1.40");
        adresses_available[1].device_id=41;
        snprintf(adresses_available[1].addr, 13, "%s", "192.168.1.41");
        adresses_available[2].device_id=42;
        snprintf(adresses_available[2].addr, 13, "%s", "192.168.1.42");
        /*
        adresses_available[3].device_id=43;
        snprintf(adresses_available[3].addr, 13, "%s", "192.168.1.43");
        adresses_available[4].device_id=44;
        snprintf(adresses_available[4].addr, 13, "%s", "192.168.1.44");
        */
        /*
        adresses_available[5].device_id=45;
        snprintf(adresses_available[5].addr, 13, "%s", "192.168.1.45");
        adresses_available[6].device_id=46;
        snprintf(adresses_available[6].addr, 13, "%s", "192.168.1.46");
        adresses_available[7].device_id=47;
        snprintf(adresses_available[7].addr, 13, "%s", "192.168.1.47");
        adresses_available[8].device_id=48;
        snprintf(adresses_available[8].addr, 13, "%s", "192.168.1.48");
        adresses_available[9].device_id=49;
        snprintf(adresses_available[9].addr, 13, "%s", "192.168.1.49");
        */
        

        LOG("------------------ FOUND PORTS and DEVICES -----------------------\n");
        for(addr_count=0;addr_count<30;addr_count++){
            if (adresses_available[addr_count].addr[0] == '\0') break; 
            LOG("Addr: %s, Device ID: %lu\n", adresses_available[addr_count].addr, adresses_available[addr_count].device_id);
            if(adresses_available[addr_count].device_id == device_id) { device_order = addr_count; };
        }
        LOG("-------------------------------------------------------------------\n");
        LOG("Order of this device %d\n", device_order);
    }

    LOG("Wait another 60 seconds to aviod discory issue\n");
    sleep(60);
    
    show_labels();
    init_model(&local_model, &model_optimizer, local_model_parameter_memory, local_model_working_memory);
    // time_t start_time, current_time;
    // start_time = time(NULL);
    bool end = false;
    for(int round=0; round<300;round++){
        current_round = round;
        LOG("####### Round: %d\n", round);
        train_model(&local_model, &model_optimizer);
        train_model(&local_model, &model_optimizer);
        train_model(&local_model, &model_optimizer);

        get_parameters(&local_params[0], &local_model);

        int param_counts_arr[NUMB_TRAINABLE_PARAMETERS];
        total_params = count_total_params(&local_params[0], &param_counts_arr[0]);
        float* flatten = (float *)malloc(total_params * sizeof(float));
        flatten_tensor_to_array(&local_params[0], flatten, param_counts_arr);

        char *bitArray_send[SEGMENT_COUNT];
        int selected_devices[SHARE_COUNT];
        float *segments[SEGMENT_COUNT];
        int segment_sizes[SEGMENT_COUNT];
        create_segments(flatten, total_params, segments, segment_sizes, bitArray_send);

        INFO("Segments Created!");
        for(int i=0; i<SEGMENT_COUNT; i++){
            INFO("Segment %d size : %d\n", i, segment_sizes[i]);
        }

        memset(selected_devices, -1, sizeof(selected_devices));
        for (int share=0; share < SHARE_COUNT; share++) {
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
        }
        free_pointer_array(segments);
        free_pointer_array(bitArray_send);
        
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
        LOG("Total sent communication cost %ld\n", comm_cost);
        if(end) {break;}
    }
}
