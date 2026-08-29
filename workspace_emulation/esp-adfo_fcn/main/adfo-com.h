#ifndef ADFO_COM
#define ADFO_COM

#include <stdio.h>
#include <stdlib.h>
#include "aifes.h"
#include "esp_netif.h"

#define DEBUG_LEVEL 2

#if DEBUG_LEVEL >= 1
    #define LOG(fmt, ...)   printf("[ROUND %d] " fmt "", current_round, ##__VA_ARGS__)
#else
    #define LOG(fmt, ...)   // nothing
#endif

#if DEBUG_LEVEL >= 2
    #define INFO(fmt, ...)  printf("[INFO] " fmt "", ##__VA_ARGS__)
#else
    #define INFO(fmt, ...)  // nothing
#endif

#if DEBUG_LEVEL >= 3
    #define DEBUG(fmt, ...) printf("[DEBUG] " fmt "", ##__VA_ARGS__)
#else
    #define DEBUG(fmt, ...) // nothing
#endif

#if DEBUG_LEVEL >= 4
    #define ANALYZE(fmt, ...) printf("[DEBUG] " fmt "", ##__VA_ARGS__)
#else
    #define ANALYZE(fmt, ...) // nothing
#endif

#define CHUNK_SIZE                  50
#define NUMB_TRAINABLE_PARAMETERS   4
#define RECEIVE_ARRAY_SIZE          10000
#define PORT                        3333
#define KEEPALIVE_IDLE              5
#define KEEPALIVE_INTERVAL          5
#define KEEPALIVE_COUNT             3
#define HOST_IP_ADDR                "172.17.0.10"
#define COMM_ROUNDS                 220
#define ENCODING                    true
#define MAX_DEVICES                 200
#define MAX_IP_LEN                  17

#define BIT_ARRAY_SIZE              10000   // Number of bits in the array
#define BYTE_SIZE                   8
#define SHARE_COUNT                 6

#define LOG_PARAM                   1000
#define SUBSET_SIZE                 150

// 1: DFA, 2: SDFA, 3: Gist_Ada
#if METHOD == 1
    #define SEGMENT_COUNT               1
    #define RANDOM_SEGMENTATION     false
    #define ADASTAIR                false
    #define SB_AGGR                 false
#elif METHOD == 2
    #define SEGMENT_COUNT               3
    #define RANDOM_SEGMENTATION     true
    #define ADASTAIR                false
    #define SB_AGGR                 false
#elif METHOD == 3
    #define SEGMENT_COUNT               3
    #define RANDOM_SEGMENTATION     false
    #define ADASTAIR                true
    #define SB_AGGR                 true
#else
    #define SEGMENT_COUNT               3
    #define RANDOM_SEGMENTATION     false
    #define ADASTAIR                true
    #define SB_AGGR                 true
#endif


#define ONLY_2                  false

#define SET_BIT(arr, n)    (arr[(n) / BYTE_SIZE] |= (1 << ((n) % BYTE_SIZE)))
#define CLEAR_BIT(arr, n)  (arr[(n) / BYTE_SIZE] &= ~(1 << ((n) % BYTE_SIZE)))
#define TOGGLE_BIT(arr, n) (arr[(n) / BYTE_SIZE] ^= (1 << ((n) % BYTE_SIZE)))
#define READ_BIT(arr, n)   ((arr[(n) / BYTE_SIZE] >> ((n) % BYTE_SIZE)) & 1)

struct adresses {
    char addr[MAX_IP_LEN];
    uint32_t device_id;
};

// extern struct adresses adresses_available[MAX_DEVICES];
extern struct adresses *adresses_available;
extern uint32_t device_id;
extern float total_receives[RECEIVE_ARRAY_SIZE];
extern float received_flatten[RECEIVE_ARRAY_SIZE];
// extern unsigned short division_array[RECEIVE_ARRAY_SIZE];
extern int *division_array;
extern int *true_labels_array;   // allocate depending on max test_size
extern int *pred_labels_array;
extern int total_params;
extern volatile int received_segments;
extern SemaphoreHandle_t addMutex;
extern esp_netif_t *eth_netif;
extern float latest_success;
extern long comm_cost;
extern int current_round;
extern int device_order;
typedef char byte;

void register_ethernet(void);
void send_array(int, const float*, char*, int);
int find_available_ports();
float decode(char *);
void encode(float, char*);
void tcp_server_task(void*);
int count_total_params(aitensor_t*, int*);
void flatten_tensor_to_array(aitensor_t*, float*, int*);
void set_bitarray(float*, char*);
void clear_received_array();
bool value_in_array(int, int*, int);
void add_received();
int select_segment(float**, int*);

#endif
