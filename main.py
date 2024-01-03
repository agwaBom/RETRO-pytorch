import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from retro_pytorch import RETRO, TrainingWrapper
if __name__ == "__main__":
    retro = RETRO(
        chunk_size = 64,                         # the chunk size that is indexed and retrieved (needed for proper relative positions as well as causal chunked cross attention)
        max_seq_len = 2048,                      # max sequence length
        enc_dim = 896,                           # encoder model dim
        enc_depth = 2,                           # encoder depth
        dec_dim = 796,                           # decoder model dim
        dec_depth = 12,                          # decoder depth
        dec_cross_attn_layers = (3, 6, 9, 12),   # decoder cross attention layers (with causal chunk cross attention)
        heads = 8,                               # attention heads
        dim_head = 64,                           # dimension per head
        dec_attn_dropout = 0.25,                 # decoder attention dropout
        dec_ff_dropout = 0.25,                   # decoder feedforward dropout
        use_deepnet = True                       # turn on post-normalization with DeepNet residual scaling and initialization, for scaling to 1000 layers
    ).cuda()

    wrapper = TrainingWrapper(
        retro = retro,                                 # path to retro instance
        knn = 2,                                       # knn (2 in paper was sufficient)
        chunk_size = 64,                               # chunk size (64 in paper)
        documents_path = './sample_text_folder',              # path to folder of text
        glob = '**/*.txt',                             # text glob
        chunks_memmap_path = './train.chunks.dat',     # path to chunks
        seqs_memmap_path = './train.seq.dat',          # path to sequence data
        doc_ids_memmap_path = './train.doc_ids.dat',   # path to document ids per chunk (used for filtering neighbors belonging to same document)
        max_chunks = 1_000_000,                        # maximum cap to chunks
        max_seqs = 100_000,                            # maximum seqs
        knn_extra_neighbors = 100,                     # num extra neighbors to fetch
        max_index_memory_usage = '100m',
        current_memory_available = '1G'
    )

    import IPython; IPython.embed(); exit(1)


    seq = torch.randint(0, 20000, (2, 2048 + 1))      # plus one since it is split into input and labels for training
    retrieved = torch.randint(0, 20000, (2, 32, 2, 128)) # retrieved tokens - (batch, num chunks, num retrieved neighbors, retrieved chunk with continuation)

    loss = retro(seq, retrieved, return_loss = True)
    loss.backward()

# do above for many steps