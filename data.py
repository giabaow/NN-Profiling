import torch

def get_synthetic_data(batch_size=32, input_shape=(3, 32, 32), seq_len=16, embed_dim=128):
    # For CNN
    cnn_data = torch.randn(batch_size, *input_shape)
    # For MLP
    mlp_data = torch.randn(batch_size, input_shape[0]*input_shape[1]*input_shape[2])
    # For Transformer
    transformer_data = torch.randn(seq_len, batch_size, embed_dim)
    return mlp_data, cnn_data, transformer_data