#ifndef LOAD_SAFETENSORS_H
#define LOAD_SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>


// Data Structures & Enums


typedef struct FileMapping {
    uint8_t *data; 
    size_t size;
    int fd;
} FileMapping;

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} GGUFType;

typedef struct {
    char *str;
    uint64_t len;
} GGUFString;

typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
} ggml_type;

typedef struct {
    int found;
    ggml_type type;
    uint32_t n_dims;
    uint64_t dims[4];
    uint64_t relative_offset;
} TargetTensorInfo;

typedef struct __attribute__((packed)){
    uint16_t d;       
    int8_t  qs[32];   
} block_q8_0;


// Function Declarations


const char* get_ggml_type_name(ggml_type type);
FileMapping map_file(const char *filePath);
void unmap_file(FileMapping *mf);
GGUFString read_gguf_string(const uint8_t *data, uint64_t *offset);
void print_gguf_value(const uint8_t *data, uint64_t *offset, uint32_t type);

uint64_t tensor_table_of_contents(const FileMapping *mf, uint64_t tensor_count, 
                                  uint64_t tensor_offset, int ShouldPrint, 
                                  const char *targetName, TargetTensorInfo *targetInfo);

uint64_t display_gloabal_dump(const FileMapping *mf, int ShouldPrint, 
                            const char *targetName, TargetTensorInfo *targetInfo);

float fp16_to_fp32(uint16_t h);
void print_q8_0_weights(const uint8_t *tensor_bytes_ptr, int count, int dequantize);
void print_f32_weights(const uint8_t *tensor_bytes_ptr, int count);
void specific_tensor_data(const FileMapping *mf, const char *tensorName);

#endif 