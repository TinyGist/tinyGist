#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include <unistd.h>

#include <esp_event.h>
#include <esp_log.h>
#include <esp_system.h>
#include <sys/param.h>
#include <esp_http_server.h>
#include "esp_netif.h"
#include "esp_eth.h"
#include "esp_mac.h"
#include "esp_random.h"
#include <string.h>

#include "protocol_examples_common.h"
#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include <lwip/netdb.h>
#include "adfo-model.h"
#include "aifes.h"
#include "adfo-com.h"
#include <fcntl.h>
#include "esp_heap_caps.h"

esp_eth_handle_t eth_handle = NULL;
esp_netif_t *eth_netif = NULL;

struct adresses *adresses_available = NULL;
// struct adresses adresses_available[MAX_DEVICES];
uint32_t device_id = 0;
int device_order = 0;
// bool comm_in_progress = false;
float total_receives[RECEIVE_ARRAY_SIZE] = {0};
float received_flatten[RECEIVE_ARRAY_SIZE] = {0};
// unsigned short division_array[RECEIVE_ARRAY_SIZE] = {0};
int *division_array = NULL;
int *true_labels_array = NULL;   // allocate depending on max test_size
int *pred_labels_array = NULL;
char received_success[3], received_bitmap[BIT_ARRAY_SIZE / BYTE_SIZE];
volatile int received_segments = 0;
char device_str[4];
char success_str[4];
char float_str[5];
char host_ip[MAX_IP_LEN] = HOST_IP_ADDR;
long comm_cost=0;
int try_port = PORT;
float latest_success = 0;
int received_success_count = 0;
int current_round = 0;

void encode(float l_param, char *packet)
{
    if(!ENCODING){
        snprintf(packet, 32, "%.*f", 5, l_param);
    } else {
        int param = (int) round(((l_param)*1000));

        char temp[3];
        snprintf(temp, 3,  "%02x", (byte) (param >> 8 * 1) & 0xFF);
        snprintf(packet, 5,  "%s%02x", temp, (byte) (param >> 8 * 0) & 0xFF);
    }
}

float decode(char *recv_packet)
{
    if(!ENCODING){
        return atof(recv_packet);
    } else {
        char test_hexstring[5] = "";

        test_hexstring[0] = recv_packet[0];
        test_hexstring[1] = recv_packet[1];
        test_hexstring[2] = recv_packet[2];
        test_hexstring[3] = recv_packet[3];		

        int value = 0;
        value = (int)strtol(test_hexstring, NULL, 16);			

        if (value & (1<<(15)))
            value -= 1 << 16;	

        return value/1000.0;
    }
}

int select_segment(float** segments, int* segment_sizes){
    int normalized_averages[SEGMENT_COUNT] = {0};
    float segment_averages[SEGMENT_COUNT] = {0};
    int picked_segment;

    for (int i = 0; i < SEGMENT_COUNT; i++) {
        for (int j = 0; j < segment_sizes[i]; j++) {
            segment_averages[i] += fabs(segments[i][j]);
        }
    }

    for (int i = 0; i < SEGMENT_COUNT; i++) {
        if(i>0){
            normalized_averages[i] = normalized_averages[i-1] + (int)(segment_averages[i]*1000);
        }else {
            normalized_averages[i] = (int)(segment_averages[i]*1000);
        }
    }

    int picked_val = ((uint32_t) esp_random()) % (normalized_averages[SEGMENT_COUNT-1]);
    printf("Picked_val: %d\n", picked_val);
    //printf("Averages: %d, %d, %d\n", normalized_averages[0], normalized_averages[1], normalized_averages[2]);
    for (int i = 0; i < SEGMENT_COUNT; i++) {
        if(picked_val <= normalized_averages[i]){
            picked_segment = i;
            break;
        }
    }
    return picked_segment;
}

void clear_received_array() {
    memset(total_receives, 0, sizeof(total_receives));
    memset(division_array, 0, RECEIVE_ARRAY_SIZE * sizeof(int));

}

bool value_in_array(int value, int *array, int size) {
    for (int i = 0; i < size; i++) {
        if (array[i] == value) {
            return true; // Value found
        }
    }
    return false; // Value not found
}

int find_available_ports()
{
    char recv_buffer[11+1];
    memset(recv_buffer, 0, sizeof(recv_buffer));
    int addr_family = 0;
    int ip_protocol = 0;
    const char* payload = "who";

    struct sockaddr_in dest_addr;
    dest_addr.sin_port = htons(try_port);
    dest_addr.sin_family = AF_INET;
    addr_family = AF_INET;
    ip_protocol = IPPROTO_IP;

    for(int i = 0; i < MAX_DEVICES; i++) {
        inet_pton(AF_INET, host_ip, &dest_addr.sin_addr);
        
        int sock =  socket(addr_family, SOCK_STREAM, ip_protocol);

        int flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, flags | O_NONBLOCK);

        if (sock < 0) {
            LOG("Unable to create socket: errno %d\n", errno);
            return -1;
        }
        LOG("Socket created, connecting to %s:%d\n", host_ip, try_port);

        int err = connect(sock, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
        if (err < 0 && errno != EINPROGRESS) {
            LOG("Connect failed immediately: errno %d\n", errno);
            close(sock);
            return try_port;
        }
        fd_set write_fds;
        FD_ZERO(&write_fds);
        FD_SET(sock, &write_fds);

        struct timeval conn_timeout;
        conn_timeout.tv_sec = 30;  // 4 seconds timeout
        conn_timeout.tv_usec = 0;

        int sel_res = select(sock + 1, NULL, &write_fds, NULL, &conn_timeout);
        if (sel_res <= 0) {
            LOG("Connect timed out or failed.\n");
            close(sock);
            return try_port;
        }

        // Restore socket to blocking mode
        fcntl(sock, F_SETFL, flags);

        err = send(sock, payload, strlen(payload), 0);
        if (err < 0) {
            LOG("Error occurred during sending: errno %d\n", errno);
            break;
        }

        int len = recv(sock, recv_buffer, sizeof(recv_buffer)-1, 0);

        // check recv result *before* parsing
        if (len <= 0) {
            if (len < 0) {
                LOG("recv failed: errno %d\n", errno);
            } else {
                LOG("peer closed connection, no data\n");
            }
            close(sock);
            break;  // try next IP or break, depending on your logic
        }

        // make C string
        recv_buffer[len] = '\0';

        // parse
        char *ret = strtok(recv_buffer, "|");
        char *device_id_str = NULL;

        if (ret != NULL) {
            device_id_str = strtok(NULL, "|");
        }

        if (!device_id_str) {
            LOG("Invalid response (no device id): '%s'\n", recv_buffer);
            close(sock);
            continue;
        }

        uint32_t converted_device_id = (uint32_t)strtoul(device_id_str, NULL, 10);

        // Error occurred during receiving
        if (len < 0) {
            LOG("recv failed: errno %d\n", errno);
            break;
        }
        // Data received
        else {
            strcpy(adresses_available[i].addr, host_ip);
            adresses_available[i].device_id = converted_device_id;

            // Increment IP addr on every connect
            int last_octet;
            char *last_dot = strrchr(host_ip, '.');         
            if (last_dot != NULL) {
                last_octet = atoi(last_dot + 1);            
                last_octet++;
                INFO("Last octet %d\n", last_octet);
                
                if (snprintf(last_dot + 1, sizeof(host_ip) - (last_dot - host_ip), "%d", last_octet) < sizeof(host_ip) - (last_dot - host_ip)) {
                    INFO("Updated IP: %s\n", host_ip);
                } else {
                    LOG("Buffer size is too small for the updated IP address.\n");
                }
            } else {
                LOG("Invalid IP address format\n");
            }
        }

        // Close connections to the socket
        if (shutdown(sock, SHUT_RDWR) == -1) { // SHUT_RDWR disables both send and receive
            LOG("SHUTDOWN ERROR\n");
            perror("shutdown");
        }

        // Close the socket
        if (close(sock) == -1) {
            LOG("CLOSE ERROR\n");
            perror("close");
        }
    }
    return 0;
}

void flatten_tensor_to_array(aitensor_t* tensor, float* flatten, int* total_params_arr){
    INFO("\n-------------------- Flatten tensor to array --------------------\n");
    int cnt = 0;
    for(int i=0; i<NUMB_TRAINABLE_PARAMETERS;i++){
        int tensor_elements = total_params_arr[i];
        for(int j=0; j<tensor_elements;j++){
            flatten[cnt++] = ((float *)(&tensor[i])->data)[j];
        }
    }
}

int count_total_params(aitensor_t* trainable_parameters, int* param_counts_arr){
    int _total_params=0;

    // TODO: tensor_elements are calculated weirdly, this might need to be fixed from copy in adfo-model
    for(int i=0; i<NUMB_TRAINABLE_PARAMETERS;i++){
        int tensor_elements = aimath_tensor_elements(&trainable_parameters[i]);
        param_counts_arr[i] = tensor_elements;
        _total_params += tensor_elements;
    }

    return _total_params;
}

bool wait_for_socket_writable(int sock, int timeout_ms) {
    fd_set wfds;
    struct timeval tv;
    
    FD_ZERO(&wfds);
    FD_SET(sock, &wfds);
    
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    
    int result = select(sock + 1, NULL, &wfds, NULL, &tv);
    return result > 0 && FD_ISSET(sock, &wfds);
}

int send_with_retry(int sock, const char* data, size_t len) {
    size_t total_sent = 0;
    while (total_sent < len) {
        int sent = send(sock, data + total_sent, len - total_sent, 0);
        if (sent < 0) {
            if (errno == EINTR) continue;  // Interrupted, try again
            return -1;  // Error
        }
        total_sent += sent;
    }
    return total_sent;
}

void send_array(int addr_index, const float* flatten, char* bitArray_send, int segment_size)
{
    printf("Inside send array!\n");
    char rx_buffer[128];
    char addr_choice[MAX_IP_LEN];
    int addr_family = 0;
    int ip_protocol = 0;
    int port_choice = PORT;

    struct sockaddr_in dest_addr;
    dest_addr.sin_port = htons(port_choice);
    dest_addr.sin_family = AF_INET;
    addr_family = AF_INET;
    ip_protocol = IPPROTO_IP;

    strcpy(addr_choice, adresses_available[addr_index].addr);

    inet_pton(AF_INET, addr_choice, &dest_addr.sin_addr);
        
    int sock =  socket(addr_family, SOCK_STREAM, ip_protocol);
    if (sock < 0) {
        LOG("Unable to create socket: errno %d\n", errno);
        return;
    }
    // printf("After socket!\n");
    struct timeval tv;
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
    ANALYZE("Print bitmap:\n");
    for(int i=0; i<BIT_ARRAY_SIZE; i++){
        ANALYZE("%d:", READ_BIT(bitArray_send, i));
    }
    ANALYZE("\nArray with size %d: \n", (int)ceil((float)segment_size/CHUNK_SIZE) * CHUNK_SIZE);
    for(int i=0; i<ceil((float)segment_size/CHUNK_SIZE) * CHUNK_SIZE; i++){
        ANALYZE("%f:", flatten[i]);
    }
    ANALYZE("\n");

    INFO("Socket created, array is being sent to %s:%d for segment size: %d\n", addr_choice, port_choice, segment_size);
    int err = connect(sock, (struct sockaddr *)&dest_addr, sizeof(dest_addr));

    if (err != 0) {
        LOG("Socket unable to connect: err %d errno %d\n", err, errno);
        LOG("connect failed: %s\n", strerror(errno));
        return;
    }
    // +1 for device id and bitmap
    for(int i=0; i<(int)ceil((float)segment_size/CHUNK_SIZE)+1; i++){
        char *chunk;
        size_t chunk_size;
        if(!i){
            // device id - delimeter - bitarray size - delimeter
            chunk_size = 3 + 1 + (BIT_ARRAY_SIZE / BYTE_SIZE) + 1;
            chunk = (char *)malloc(chunk_size);
            memset(chunk, 0, chunk_size);
            sprintf(device_str, "%d", (int)(latest_success*1000));
            // encode(latest_success, success_str);
            printf("sr to send: %s\n", device_str);
            strcat(chunk, device_str);
            strcat(chunk, "|");
            for(int b=0; b<BIT_ARRAY_SIZE/BYTE_SIZE;b++){
                chunk[4+b] = bitArray_send[b];
            }
            ANALYZE("chunk: %s\n", chunk);
        } else {
            chunk_size = CHUNK_SIZE * 5;
            chunk = (char *)malloc(chunk_size + 1);
            chunk[0] = '\0';
            chunk[chunk_size] = 0;
        }
        
        // Make a string package to send
        if(i){
            for(int j=0; j<CHUNK_SIZE;j++){
                encode(flatten[(i-1)*CHUNK_SIZE + j], float_str);
                strcat(chunk, float_str);
                strcat(chunk, ";");
            }
        }
        const TickType_t xDelay = 50 / portTICK_PERIOD_MS;
        vTaskDelay( xDelay );
        wait_for_socket_writable(sock, 30000);
        int sent = -1;
        while(sent < 0){
            comm_cost += chunk_size/3*2;
            if(RANDOM_SEGMENTATION && !i){
                comm_cost = comm_cost - (1 + (BIT_ARRAY_SIZE / BYTE_SIZE) + 1);
            }
            sent = send_with_retry(sock, chunk, chunk_size);
            if(sent<0){
                LOG("Failed, exit send_array!\n");
                if (chunk != NULL) {
                    free(chunk);
                    chunk = NULL;    // Prevent future double-free
                }
                // Close the socket
                if (shutdown(sock, SHUT_RDWR) == -1) {
                    perror("shutdown");
                }

                if (close(sock) == -1) {
                    perror("close");
                }
                return;
            }
        }

        ANALYZE("##SENDER: After send: first char %c -- sent length: %d, Progress: %d / %d\n", chunk[0], sent, i, (int)ceil((float)segment_size/CHUNK_SIZE));
        if (chunk != NULL) {
            free(chunk);
            chunk = NULL;    // Prevent future double-free
        }

        if (sent == -1) {
            perror("Send failed");
            close(sock);
        }
        int len = recv(sock, rx_buffer, sizeof(rx_buffer) - 1, 0);

        if (len < 0) {
            LOG("recv failed: errno %d\n", errno);
            break;
        } else {
            rx_buffer[len] = 0; // Null-terminate whatever we received and treat like a string

            if(strcmp(rx_buffer, "fail\n") == 0){
                i--;
            }
        }
    }
    INFO("#### Send Complete!\n");
    // Close connections to the socket
    if (shutdown(sock, SHUT_RDWR) == -1) { // SHUT_RDWR disables both send and receive
        perror("shutdown");
    }

    // Close the socket
    if (close(sock) == -1) {
        perror("close");
    }
}

void add_received(){
    INFO("--------- Add received function!\n");

    xSemaphoreTake(addMutex, portMAX_DELAY);
    received_segments += 1;
    int flat_cnt = 0;
    for(int i=0; i<RECEIVE_ARRAY_SIZE; i++){
        int bitval = (int)READ_BIT(received_bitmap, i);
        if(bitval){
            if(flat_cnt < 10) { DEBUG("idx: %d, Received flatten Segment: %f\n", i, received_flatten[flat_cnt]); }
            if(SB_AGGR) {
                division_array[i] += received_success_count;
                total_receives[i] += (float)received_flatten[flat_cnt++] * received_success_count;
            } else {
                division_array[i] += 1;
                total_receives[i] += (float)received_flatten[flat_cnt++];
            }

        }
    }

    memset(received_bitmap, 0, sizeof(received_bitmap));
    xSemaphoreGive(addMutex);
}

static void do_retransmit(const int sock)
{
    int len;
    char rx_buffer[3+1+(BIT_ARRAY_SIZE / BYTE_SIZE)+1+(CHUNK_SIZE*6)+1];
    int array_cnt=0;
    int packet_cnt=0;
    bool bitmap_received = false;

    do {
        len = recv(sock, rx_buffer, sizeof(rx_buffer) - 1, 0);
        DEBUG("### Received len: %d - Packet Number: %d", len, packet_cnt++);
        DEBUG("  |  ");
        if (len < 0) {
            LOG("Error occurred during receiving: errno %d, received_param: %d, total_params: %d\n", errno, array_cnt, total_params);
            if(array_cnt == total_params){
                DEBUG("Segment received! received_param: %d\n", array_cnt);
                add_received();
            }
        } else if (len == 0) {
            if(bitmap_received) {
                DEBUG("Segment received! received_param: %d\n", array_cnt);
                add_received();
            }
            INFO("Connection closed!\n");
        } else {
            rx_buffer[len] = 0; // Null-terminate whatever is received and treat it like a string
            ANALYZE("Received: %s\n", rx_buffer);

            if (strcmp(rx_buffer, "who") == 0){
                char payload[11];  // Allocate enough space for the string and the device ID
                sprintf(payload, "success|%lu", device_id);
                send(sock, payload,  strlen(payload), 0);
            } else {
                char* element;
                if(!bitmap_received){
                    strncpy(received_success, rx_buffer, 3);
                    received_success_count = atoi(received_success);
                    // LOG("Receive in progress with success: %s\n", received_success);
                    LOG("Receive in progress with success: %d\n", received_success_count);
                    for(int b=0; b<BIT_ARRAY_SIZE/BYTE_SIZE;b++){
                        received_bitmap[b] = rx_buffer[4+b];
                    }
                    bitmap_received = true;
                    element = strtok(NULL, ";"); // so that below check satisfies element == NULL
                } else {
                    element = strtok(rx_buffer, ";");
                }
                
                while (element != NULL) {
                    float num = decode(element);
                    received_flatten[array_cnt] = num;
                    array_cnt++;
                    element = strtok(NULL, ";");
                }
                char payload[11];
                sprintf(payload, "success|%lu", device_id);
                int to_write = 11;
                send(sock, payload,  strlen(payload), 0);
            }
        }
    } while (len > 0);
}

void tcp_server_task(void *pvParameters)
{
    char addr_str[128];
    int addr_family = (int)pvParameters;
    int ip_protocol = 0;
    int keepAlive = 1;
    int keepIdle = KEEPALIVE_IDLE;
    int keepInterval = KEEPALIVE_INTERVAL;
    int keepCount = KEEPALIVE_COUNT;
    struct sockaddr_storage dest_addr;

    if (addr_family == AF_INET) {
        struct sockaddr_in *dest_addr_ip4 = (struct sockaddr_in *)&dest_addr;
        dest_addr_ip4->sin_addr.s_addr = htonl(INADDR_ANY);
        dest_addr_ip4->sin_family = AF_INET;
        dest_addr_ip4->sin_port = htons(PORT);
        ip_protocol = IPPROTO_IP;
    }

    int listen_sock = socket(addr_family, SOCK_STREAM, ip_protocol);
    if (listen_sock < 0) {
        LOG("Unable to create socket: errno %d\n", errno);
        vTaskDelete(NULL);
        return;
    }
    int opt = 1;
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    // int rcvbuf = 262144;  // 256KB
    // setsockopt(listen_sock, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    
    int err = bind(listen_sock, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
    if (err != 0) {
        LOG("Socket unable to bind: errno %d\n", errno);
        LOG("IPPROTO: %d\n", addr_family);
        goto CLEAN_UP;
    }
    err = listen(listen_sock, 1);
    if (err != 0) {
        LOG("Error occurred during listen: errno %d\n", errno);
        goto CLEAN_UP;
    }

    while (1) {
        struct sockaddr_storage source_addr; // Large enough for both IPv4 or IPv6
        socklen_t addr_len = sizeof(source_addr);
        int sock = accept(listen_sock, (struct sockaddr *)&source_addr, &addr_len);
        DEBUG("New sock started receiving!\n");
        if (sock < 0) {
            LOG("Unable to accept connection: errno %d\n", errno);
            break;
        }

        struct timeval tv;
        tv.tv_sec = 5;
        tv.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &keepAlive, sizeof(int));
        setsockopt(sock, IPPROTO_TCP, TCP_KEEPIDLE, &keepIdle, sizeof(int));
        setsockopt(sock, IPPROTO_TCP, TCP_KEEPINTVL, &keepInterval, sizeof(int));
        setsockopt(sock, IPPROTO_TCP, TCP_KEEPCNT, &keepCount, sizeof(int));
        // Convert ip address to string
        if (source_addr.ss_family == PF_INET) {
            inet_ntoa_r(((struct sockaddr_in *)&source_addr)->sin_addr, addr_str, sizeof(addr_str) - 1);
        }
        if (source_addr.ss_family == PF_INET6) {
            inet6_ntoa_r(((struct sockaddr_in6 *)&source_addr)->sin6_addr, addr_str, sizeof(addr_str) - 1);
        }
        DEBUG("Socket accepted ip address: %s\n", addr_str);

        do_retransmit(sock);
        LOG("Receiving completed!\n");

        close(sock);
    }

CLEAN_UP:
    close(listen_sock);
    vTaskDelete(NULL);
}

static void event_handler(void *arg, esp_event_base_t event_base,
                          int32_t event_id, void *event_data)
{
    if (event_base == ETH_EVENT) {
        if (event_id == ETHERNET_EVENT_START) return;
        if (event_id == ETHERNET_EVENT_STOP) return;
    }
    if (event_base == IP_EVENT)
        if (event_id == IP_EVENT_ETH_GOT_IP)
		return;
}

void register_ethernet(void)
{
#ifdef DATASET
    uint8_t base_mac[6] = { 0x02, 0x00, 0x00, 0x00,
                       (uint8_t)((DATASET >> 8) & 0xFF),
                       (uint8_t)(DATASET & 0xFF) };
    ESP_ERROR_CHECK(esp_base_mac_addr_set(base_mac));
#endif
#ifdef METHOD
    LOG("Used method is %d\n", METHOD);
#else
    LOG("METHOD is not defined!\n");
#endif
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    esp_netif_config_t cfg = ESP_NETIF_DEFAULT_ETH();
    esp_netif_t *eth_netif = esp_netif_new(&cfg);

    eth_mac_config_t mac_config = ETH_MAC_DEFAULT_CONFIG();
    eth_phy_config_t phy_config = ETH_PHY_DEFAULT_CONFIG();

    esp_eth_mac_t *mac = esp_eth_mac_new_openeth(&mac_config);
    esp_eth_phy_t *phy = esp_eth_phy_new_dp83848(&phy_config);

    esp_eth_handle_t eth_handle = NULL;
    esp_eth_config_t config = ETH_DEFAULT_CONFIG(mac, phy);
    ESP_ERROR_CHECK(esp_eth_driver_install(&config, &eth_handle));

    ESP_ERROR_CHECK(esp_netif_attach(eth_netif, esp_eth_new_netif_glue(eth_handle)));

    ESP_ERROR_CHECK(esp_event_handler_register(ETH_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_ETH_GOT_IP, &event_handler, NULL));

    esp_err_t err = esp_netif_dhcpc_stop(eth_netif);
    if (err != ESP_OK && err != ESP_ERR_ESP_NETIF_DHCP_ALREADY_STOPPED) {
        ESP_ERROR_CHECK(err);
    }

    int last_octet = 10;
#ifdef DATASET
    last_octet += DATASET;  // DATASET=0..199 -> 10..209
    LOG("ETH IP ends with: %d\n", last_octet);
#else
    LOG("DATASET is not defined!\n");
#endif

    esp_netif_ip_info_t ip_info;
    memset(&ip_info, 0, sizeof(ip_info));
    IP4_ADDR(&ip_info.ip,      172, 17, 0, last_octet);
    IP4_ADDR(&ip_info.gw,      172, 17, 0, 1);
    IP4_ADDR(&ip_info.netmask, 255, 255, 255, 0);

    ESP_ERROR_CHECK(esp_netif_set_ip_info(eth_netif, &ip_info));

    ESP_ERROR_CHECK(esp_eth_start(eth_handle));

    LOG("Ethernet started with static IP: 172.17.0.%d\n", last_octet);
}


/*
void register_ethernet(void)
{
    uint8_t random_mac[6];  // Example MAC address
    random_mac[0] = 0x02;  // Set the first byte as '02' (locally administered address)
    random_mac[1] = esp_random() & 0xFF;  // Random byte 1
    random_mac[2] = esp_random() & 0xFF;  // Random byte 2
    random_mac[3] = esp_random() & 0xFF;  // Random byte 3
    random_mac[4] = esp_random() & 0xFF;  // Random byte 4
    random_mac[5] = esp_random() & 0xFF;  // Random byte 5
    esp_err_t err = esp_base_mac_addr_set(random_mac);

    // tcpip_adapter_init();
    nvs_flash_init();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_config_t cfg = ESP_NETIF_DEFAULT_ETH();
    eth_netif = esp_netif_new(&cfg);

    eth_mac_config_t mac_config = ETH_MAC_DEFAULT_CONFIG();
    mac_config.rx_task_stack_size = 4096;
    
    eth_phy_config_t phy_config = ETH_PHY_DEFAULT_CONFIG();
    esp_eth_mac_t *mac = esp_eth_mac_new_openeth(&mac_config);

    esp_eth_phy_t *phy = esp_eth_phy_new_dp83848(&phy_config);

    esp_eth_config_t config = ETH_DEFAULT_CONFIG(mac, phy);
    ESP_ERROR_CHECK(esp_eth_driver_install(&config, &eth_handle)); // THIS FAILS
    ESP_ERROR_CHECK(esp_netif_attach(eth_netif, esp_eth_new_netif_glue(eth_handle)));
    ESP_ERROR_CHECK(esp_event_handler_register(ETH_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_ETH_GOT_IP, &event_handler, NULL));
    ESP_ERROR_CHECK(esp_eth_start(eth_handle));
}
*/
