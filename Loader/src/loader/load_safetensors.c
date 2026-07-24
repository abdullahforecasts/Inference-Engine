#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h> 
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include "../../inc/loader/load_safetensors.h"

//implemenaion of methods 

const char* get_ggml_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return "F32 (Float32)";
        case GGML_TYPE_F16:  return "F16 (Float16)";
        case GGML_TYPE_Q4_0: return "Q4_0 (4-bit Quantized)";
        case GGML_TYPE_Q4_1: return "Q4_1 (4-bit Quantized)";
        case GGML_TYPE_Q5_0: return "Q5_0 (5-bit Quantized)";
        case GGML_TYPE_Q5_1: return "Q5_1 (5-bit Quantized)";
        case GGML_TYPE_Q8_0: return "Q8_0 (8-bit Quantized)";
        case GGML_TYPE_Q8_1: return "Q8_1 (8-bit Quantized)";
        case GGML_TYPE_Q2_K: return "Q2_K (2-bit K-Quantized)";
        case GGML_TYPE_Q3_K: return "Q3_K (3-bit K-Quantized)";
        case GGML_TYPE_Q4_K: return "Q4_K (4-bit K-Quantized)";
        case GGML_TYPE_Q5_K: return "Q5_K (5-bit K-Quantized)";
        case GGML_TYPE_Q6_K: return "Q6_K (6-bit K-Quantized)";
        case GGML_TYPE_Q8_K: return "Q8_K (8-bit K-Quantized)";
        default:             return "Unknown GGML Type";
    }
}

FileMapping map_file(const char * filePath){
    FileMapping mf = {NULL, 0, -1};
    mf.fd = open(filePath, O_RDONLY);

    if (mf.fd < 0) {
        perror("Error opening file");
        exit(1);
    }

    struct stat sb;
    if (fstat(mf.fd, &sb) < 0)
    {
        perror("Error getting file size");
        close(mf.fd);
        exit(1);
    }

    mf.size = sb.st_size;
    mf.data = mmap(NULL, mf.size, PROT_READ, MAP_SHARED, mf.fd, 0);
    if (mf.data == MAP_FAILED)
    {
        perror("Error mmapping file"); 
        close(mf.fd); 
        exit(1);
    }
    
    return mf;
}

void unmap_file(FileMapping *mf) {
    if (mf->data && mf->data != MAP_FAILED) {
        munmap(mf->data, mf->size);
    }
    if (mf->fd >= 0) {
        close(mf->fd);
    }
}

GGUFString read_gguf_string(const uint8_t *data, uint64_t *offset) {
    GGUFString gstr;
    gstr.len = *(uint64_t *)(data + *offset);
    *offset += 8;
    
    gstr.str = malloc(gstr.len + 1);
    memcpy(gstr.str, data + *offset, gstr.len);
    gstr.str[gstr.len] = '\0';
    *offset += gstr.len;
    
    return gstr;
}

void print_gguf_value(const uint8_t *data, uint64_t *offset, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT32: {
            uint32_t val = *(uint32_t *)(data + *offset);
            printf("%u\n", val);
            *offset += 4;
            break;
        }
        case GGUF_TYPE_INT32: {
            int32_t val = *(int32_t *)(data + *offset);
            printf("%d\n", val);
            *offset += 4;
            break;
        }
        case GGUF_TYPE_FLOAT32: {
            float val = *(float *)(data + *offset);
            printf("%f\n", val);
            *offset += 4;
            break;
        }
        case GGUF_TYPE_BOOL: {
            uint8_t val = *(uint8_t *)(data + *offset);
            printf("%s\n", val ? "true" : "false");
            *offset += 1;
            break;
        }
        case GGUF_TYPE_STRING: {
            GGUFString val_str = read_gguf_string(data, offset);
            printf("\"%s\"\n", val_str.str);
            free(val_str.str);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t item_type = *(uint32_t *)(data + *offset);
            *offset += 4;
            uint64_t array_len = *(uint64_t *)(data + *offset);
            *offset += 8;
            
            printf("[Array of type %u, length %lu]: ", item_type, array_len);
            
            if (array_len > 0) {
                if (item_type == GGUF_TYPE_STRING) {
                    GGUFString first_item = read_gguf_string(data, offset);
                    printf("[\"%s\", ...]\n", first_item.str);
                    free(first_item.str);
                    
                    for (uint64_t i = 1; i < array_len; i++) {
                        uint64_t skip_len = *(uint64_t *)(data + *offset);
                        *offset += 8 + skip_len;
                    }
                } else {
                    uint32_t element_sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
                    uint32_t size = element_sizes[item_type];
                    printf("(...raw data...)\n");
                    *offset += (array_len * size);
                }
            } else {
                printf("[]\n");
            }
            break;
        }
        default:
            printf("Unknown type %u\n", type);
            exit(1);
    }
}

uint64_t tensor_table_of_contents(const FileMapping *mf, uint64_t tensor_count, uint64_t tensor_offset, int ShouldPrint, const char *targetName, TargetTensorInfo *targetInfo)
{
    if (targetInfo) {
        targetInfo->found = 0;
    }
    

    for (uint64_t i = 0; i < tensor_count; i++)
    {
        GGUFString t_name = read_gguf_string(mf->data, &tensor_offset);

        uint32_t n_dims = *(uint32_t *)(mf->data + tensor_offset);
        tensor_offset += 4;

        uint64_t *dims = malloc(n_dims * sizeof(uint64_t));
        for (uint32_t j = 0; j < n_dims; j++)
        {
           dims[j] = *(uint64_t*)(mf->data + tensor_offset); 
           tensor_offset += 8;
        }

        uint32_t t_type = *(uint32_t*)(mf->data + tensor_offset);
        tensor_offset += 4;

        uint64_t relative_data_offset = *(uint64_t *)(mf->data + tensor_offset);
        tensor_offset += 8; 

        // if (ShouldPrint)
        // printf("Tensor %lu: Name: %s, Type: %u, Dims: %u\n", i, t_name.str, t_type, n_dims);

        if (ShouldPrint) 
        printf("Tensor %lu: Name: %s, Type: %u (%s), Dims: %u\n", i, t_name.str, t_type, get_ggml_type_name((ggml_type)t_type), n_dims);

    
    
        if (targetName && targetInfo && strcmp(t_name.str, targetName) == 0) 
        {
            targetInfo->found = 1;
            targetInfo->type = t_type;
            targetInfo->n_dims = n_dims;

            for (uint32_t d = 0; d < n_dims && d < 4; d++) {
                targetInfo->dims[d] = dims[d];
            }

            targetInfo->relative_offset = relative_data_offset;
        }

        free(t_name.str);
        free(dims);
    }

    if (ShouldPrint)
        printf("Here ends the tensor_table_of_contents at offset : %lu \n", tensor_offset);

    return tensor_offset;
}

uint64_t display_gloabal_dump(const FileMapping *mf, int ShouldPrint, const char *targetName, TargetTensorInfo *targetInfo) {
    // Read Header
    uint32_t version = *(uint32_t *)(mf->data + 4);
    uint64_t tensor_count = *(uint64_t *)(mf->data + 8);
    uint64_t metadata_kv_count = *(uint64_t *)(mf->data + 16);

    if (ShouldPrint) {
        printf("Format Version: %u\n", version);
        printf("Tensor Count: %lu\n", tensor_count);
        printf("Metadata KV Count: %lu\n\n", metadata_kv_count);
    }
    
    uint64_t offset = 24; 
    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        GGUFString key = read_gguf_string(mf->data, &offset);
        uint32_t value_type = *(uint32_t *)(mf->data + offset);
        offset += 4;

        if (ShouldPrint) {
            printf("%s = ", key.str);
            print_gguf_value(mf->data, &offset, value_type);
        } else {
          
            switch (value_type) {
                case GGUF_TYPE_UINT8:   case GGUF_TYPE_INT8:   offset += 1; break;
                case GGUF_TYPE_UINT16:  case GGUF_TYPE_INT16:  offset += 2; break;
                case GGUF_TYPE_UINT32:  case GGUF_TYPE_INT32:  case GGUF_TYPE_FLOAT32: offset += 4; break;
                case GGUF_TYPE_BOOL:    offset += 1; break;
                case GGUF_TYPE_STRING: {
                    uint64_t len = *(uint64_t *)(mf->data + offset);
                    offset += 8 + len;
                    break;
                }
                case GGUF_TYPE_ARRAY: {
                    uint32_t item_type = *(uint32_t *)(mf->data + offset); offset += 4;
                    uint64_t array_len = *(uint64_t *)(mf->data + offset); offset += 8;
                    if (item_type == GGUF_TYPE_STRING) {
                        for (uint64_t a = 0; a < array_len; a++) {
                            uint64_t skip_len = *(uint64_t *)(mf->data + offset);
                            offset += 8 + skip_len;
                        }
                    } else {
                        uint32_t element_sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
                        offset += (array_len * element_sizes[item_type]);
                    }
                    break;
                }
                default: offset += 8; break; 
            }
        }

        free(key.str);
    }

    if (ShouldPrint) {
        printf("\n[Metadata Section Ends at Byte Offset: %lu]\n \n", offset);
    }
    
    uint64_t tensor_offset = offset;
    return tensor_table_of_contents(mf, tensor_count, tensor_offset, ShouldPrint, targetName, targetInfo);
}

/*

visualization of the conversion 

 ┌───┬───────────────────┬──────────────────────────────────────────┐
 │ S │   Exponent (E)    │               Mantissa (M)               │
 └───┴───────────────────┴──────────────────────────────────────────┘
 bit   bits 14 to 10           bits 9 to 0
 15    (5 bits total)          (10 bits total)



 ┌───┬───────────────────────────────┬─────────────────────────────────────────────────────────┐
 │ S │         Exponent (E)          │                      Mantissa (M)                       │
 └───┴───────────────────────────────┴─────────────────────────────────────────────────────────┘
 bit   bits 30 to 23                   bits 22 to 0
 31    (8 bits total)                  (23 bits total)

*/

float fp16_to_fp32(uint16_t h) {
    _Float16 f16;
    memcpy(&f16, &h, sizeof(uint16_t));
    return (float)f16;
}

void print_q8_0_weights(const uint8_t *tensor_bytes_ptr, int count, int dequantize) {

    block_q8_0 *blocks = (block_q8_0 *)tensor_bytes_ptr;
    
    for (int i = 0; i < count; i++) {
        int block_idx = i / 32;
        int local_idx = i % 32;
        
        // block_q8_0 block = blocks[block_idx];
        // int8_t raw_val = block.qs[local_idx];

        // lil optimization hehehe

        const block_q8_0 *block = &blocks[block_idx];
        int8_t raw_val = block->qs[local_idx];  
        
        if (dequantize) {
            float scale = fp16_to_fp32(block->d);
            float dequantized_val = raw_val * scale;
            printf("Weight[%d] (Dequantized) = %.8f\n", i, dequantized_val);
        } else {
            printf("Weight[%d] (Raw Q8) = %d\n", i, raw_val);
        }
    }


}

void print_f32_weights(const uint8_t *tensor_bytes_ptr, int count) {
    float *weights = (float *)tensor_bytes_ptr;
    for (int i = 0; i < count; i++) {
        printf("Weight[%d] = %f\n", i, weights[i]);
    }
}

void specific_tensor_data(const FileMapping *mf, const char * tensorName)
{
    TargetTensorInfo info;
   
    uint64_t end_of_toc = display_gloabal_dump(mf, 0, tensorName, &info);
    
    if (!info.found) {
        printf("Error: Tensor '%s' not found in this GGUF file.\n", tensorName);
        return;
    }

    uint64_t alignment = 32;
    uint64_t raw_data_start = ((end_of_toc + alignment - 1) / alignment) * alignment;

    uint64_t absolute_coordinate = raw_data_start + info.relative_offset;
    uint8_t *tensor_bytes_ptr = mf->data + absolute_coordinate;

     const char * type = get_ggml_type_name(info.type);
    printf("\n--- Target Tensor Found ---\n");
    printf("Name:                           %s\n", tensorName);
    printf("Type ID:                        %u (%s)\n", info.type, type);
    printf("Dimensions:                     %u [", info.n_dims);

    for(uint32_t d = 0; d < info.n_dims; d++) {
        printf("%lu%s", info.dims[d], (d == info.n_dims - 1) ? "" : " x ");
    }


    printf("]\n");
    printf("Binary Pool Begins at Offset:   %lu\n", raw_data_start);
    printf("Relative Tensor Offset:         %lu\n", info.relative_offset);
    printf("Absolute File Byte Position:    %lu\n", absolute_coordinate);
    printf("Direct Memory Pointer Address:  %p\n", (void*)tensor_bytes_ptr);


   // branch and call 
if (info.type == GGML_TYPE_F32)
    {
       print_f32_weights(tensor_bytes_ptr, 20);
    }
    else if (info.type == GGML_TYPE_Q8_0)
    {
        print_q8_0_weights(tensor_bytes_ptr, 20, 1); 
    }
    else
    {
        printf("Type %s not directly printable yet.\n", type);
    }
    
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Input format is as \n 1-  ./gguf_dump model.gguf  \n 2-  ./gguf_dump model.gguf blk.0.attn_q.weight \n");
        return 1;
    }

    char *filePath = argv[1];

    FileMapping mf = map_file(filePath);
    printf("SuccessFull File mapping %zu\n", mf.size);

    if (mf.size < 24) {
        printf("Error: File too small.\n");
        unmap_file(&mf);
        return 1;
    }

    if (argc == 2)
    {
        
        display_gloabal_dump(&mf, 1, NULL, NULL);
    }
    else if (argc == 3)
    {
        char * tensorName = argv[2];
        specific_tensor_data(&mf, tensorName);
    }

    unmap_file(&mf);
    return 0;

    
}

