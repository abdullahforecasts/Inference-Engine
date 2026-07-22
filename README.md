DataSet link :  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf

Compile : gcc loader.c -o loader

Run : ./loader tl.gguf
      ./loader tl.gguf token_embd.weight
      ./loader tl.gguf output_norm.weight
      ./loader tl.gguf blk.9.attn_q.weight
