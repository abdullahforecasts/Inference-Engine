#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
 

typedef struct FileMapping
{
    __uint8_t * data;
    size_t size;
    int fd;
}FileMapping;

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

FileMapping mf = {NULL,-1,-1};
uint32_t general_alignment = 32;

typedef struct {
    char *str;
    uint64_t len;
} GGUFString;

FileMapping map_file(const char *  filePath)
{
    FileMapping mf={NULL,0,-1};
    mf.fd = open(filePath,O_RDONLY);
    if (mf.fd < 0) {
        perror("Error opening file");
        exit(1);
    }

    struct stat sb;
    if(fstat(mf.fd,&sb) < 0)
    {
        perror("Error getting file size");
        close(mf.fd);
        exit(1);
    }

    mf.size = sb.st_size;
    mf.data = mmap(NULL,mf.size,PROT_READ,MAP_SHARED,mf.fd,0);
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
            
            // For safety and brevity in the dump, we print the first few items if it's strings/numbers
            if (array_len > 0) {
                if (item_type == GGUF_TYPE_STRING) {
                    // Printing just the first element as an example if it's a massive array (like tokens)
                    GGUFString first_item = read_gguf_string(data, offset);
                    printf("[\"%s\", ...]\n", first_item.str);
                    free(first_item.str);
                    
                    // Skiping the remaining strings in the array to advance the offset correctly
                    for (uint64_t i = 1; i < array_len; i++) {
                        uint64_t skip_len = *(uint64_t *)(data + *offset);
                        *offset += 8 + skip_len;
                    }
                } else {
                    // For numeric arrays, advance the offset past the block of elements
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

void tensor_table_of_contents(const char * filePath , uint64_t tensor_count, uint64_t tensor_offset)
{

    for (uint64_t i = 0; i < tensor_count; i++)
    {
        GGUFString  t_name = read_gguf_string(mf.data,&tensor_offset);

        uint32_t n_dims = *(uint32_t *)(mf.data +tensor_offset);
        tensor_offset += 4;

        uint64_t * dims   = malloc(n_dims * sizeof(uint64_t));
        for (uint32_t i = 0; i < n_dims; i++)
        {
           dims[i] = *(uint64_t*)(mf.data +tensor_offset); 
           tensor_offset += 8;
        }

        uint32_t t_type = *(uint32_t*)(mf.data +tensor_offset);
        tensor_offset+=4;

        uint64_t relative_data_offset = *(uint64_t *)(mf.data + tensor_offset);
        tensor_offset += 8; 

        printf("Tensor %lu: Name: %s, Type: %u, Dims: %u\n", i, t_name.str, t_type, n_dims);
    
        free(t_name.str);
        free(dims);
        
    }

    printf("Here ends the tensor_table_of_contents at offset : %zu \n", tensor_offset);
    
}

void display_gloabal_dump(const char *filePath) {
    mf = map_file(filePath);
    printf("SuccessFull File mapping %zu\n", mf.size);

    if (mf.size < 24) {
        printf("Error: File too small.\n");
        unmap_file(&mf);
        return;
    }

    // Read Header
    uint32_t version = *(uint32_t *)(mf.data + 4);
    uint64_t tensor_count = *(uint64_t *)(mf.data + 8);
    uint64_t metadata_kv_count = *(uint64_t *)(mf.data + 16);

    printf("Format Version: %u\n", version);
    printf("Tensor Count: %lu\n", tensor_count);
    printf("Metadata KV Count: %lu\n\n", metadata_kv_count);

    
    uint64_t offset = 24; 
    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        GGUFString key = read_gguf_string(mf.data, &offset);
        uint32_t value_type = *(uint32_t *)(mf.data + offset);
        offset += 4;

        if (strcmp(key.str, "general.alignment") == 0) {
        // Since it's a UINT32, read it directly
        general_alignment = *(uint32_t *)(mf.data + offset);
         }
        printf("%s = ", key.str);
        print_gguf_value(mf.data, &offset, value_type);

        free(key.str);
    }

    // now Moiz knows from where to pick up..... 
    printf("\n[Metadata Section Ends at Byte Offset: %lu]\n", offset);
    uint64_t tensor_offset = offset;

     printf("\nGeneral_Alignment : %"PRIu32"\n", general_alignment);

   
    tensor_table_of_contents(filePath,tensor_count,tensor_offset);


    unmap_file(&mf);

}

void specific_tensor_data(const char * filePath,const char * tensorName )
{
    printf("called : specific_tensor_data \n");
}

int main(int argc , char **argv)
{
    if(argc < 2)
    {
    printf("Input format is as \n 1-  ./gguf_dump model.gguf  \n 2-  ./gguf_dump model.gguf blk.0.attn_q.weight \n");
    return 1;
    }

    char *filePath = argv[1];
    if(argc == 2)
    {
        display_gloabal_dump(filePath);
    }
    else if (argc == 3)
    {
        char * tensorName = argv[2];
        specific_tensor_data(filePath, tensorName);
    }

    return 0;
}



