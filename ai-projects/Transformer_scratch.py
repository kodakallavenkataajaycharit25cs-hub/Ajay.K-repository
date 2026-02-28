import numpy as np

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.minimum(scores, mask)
    attn_weights = softmax(scores)
    return np.matmul(attn_weights, V), attn_weights

def layer_norm(X, eps=1e-6):
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    return (X - mean) / np.sqrt(var + eps)

def residual_layer_norm(X, sublayer_out):
    residual = X + sublayer_out
    return layer_norm(residual)

def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads=2, mask=None):
    B, T, C = X.shape
    head_size = C // num_heads
    
    Q = np.matmul(X, Wq).reshape(B, T, num_heads, head_size).transpose(0, 2, 1, 3)
    K = np.matmul(X, Wk).reshape(B, T, num_heads, head_size).transpose(0, 2, 1, 3)
    V = np.matmul(X, Wv).reshape(B, T, num_heads, head_size).transpose(0, 2, 1, 3)
    
    out_heads, attn = scaled_dot_product_attention(Q, K, V, mask)
    out = out_heads.transpose(0, 2, 1, 3).reshape(B, T, C)
    return np.matmul(out, Wo), attn

def feed_forward(X, W1, b1, W2, b2):
    hidden = np.maximum(0, np.matmul(X, W1) + b1)
    return np.matmul(hidden, W2) + b2

def encoder_layer(X, Wq, Wk, Wv, Wo, W1, b1, W2, b2, num_heads=2):
    attn_out, _ = multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads)
    X = residual_layer_norm(X, attn_out)
    ffn_out = feed_forward(X, W1, b1, W2, b2)
    X = residual_layer_norm(X, ffn_out)
    return X

def decoder_layer(X, enc_out, self_Wq, self_Wk, self_Wv, self_Wo, encdec_Wq, encdec_Wk, encdec_Wv, encdec_Wo, W1, b1, W2, b2, num_heads=2):
    B, T, C = X.shape
    head_size = C // num_heads
    mask = np.triu(np.ones((T, T)) * -100, k=1)[np.newaxis, np.newaxis, :, :]
    
    self_attn, _ = multi_head_attention(X, self_Wq, self_Wk, self_Wv, self_Wo, num_heads, mask)
    X = residual_layer_norm(X, self_attn)
    
    encdec_Q = np.matmul(X, encdec_Wq).reshape(B, T, num_heads, head_size).transpose(0, 2, 1, 3)
    encdec_K = np.matmul(enc_out, encdec_Wk).reshape(B, T, num_heads, head_size).transpose(0, 2, 1, 3)
    encdec_V = np.matmul(enc_out, encdec_Wv).reshape(B, T, num_heads, head_size).transpose(0, 2, 1, 3)
    encdec_out, _ = scaled_dot_product_attention(encdec_Q, encdec_K, encdec_V)
    encdec_out = encdec_out.transpose(0, 2, 1, 3).reshape(B, T, C)
    encdec_out = np.matmul(encdec_out, encdec_Wo)
    
    X = residual_layer_norm(X, encdec_out)
    ffn_out = feed_forward(X, W1, b1, W2, b2)
    X = residual_layer_norm(X, ffn_out)
    return X

np.random.seed(42)
B, T, C, vocab, d_ff, N, num_heads = 1, 3, 4, 10, 8, 2, 2

src_ids = np.random.randint(0, vocab, (B, T))
tgt_ids = np.random.randint(0, vocab, (B, T))

embed_W_src = np.random.randn(vocab, C) * 0.02
embed_W_tgt = np.random.randn(vocab, C) * 0.02
gen_W = np.random.randn(C, vocab) * 0.02

Wq = np.random.randn(C, C) * 0.02
Wk = Wq.copy()
Wv = Wq.copy()
Wo = np.eye(C) * 0.02
self_Wq = np.random.randn(C, C) * 0.02
self_Wk = self_Wq.copy()
self_Wv = self_Wq.copy()
self_Wo = np.eye(C) * 0.02
encdec_Wq = np.random.randn(C, C) * 0.02
encdec_Wk = encdec_Wq.copy()
encdec_Wv = encdec_Wq.copy()
encdec_Wo = np.eye(C) * 0.02
W1 = np.random.randn(C, d_ff) * 0.02
b1 = np.zeros((1, d_ff))
W2 = np.random.randn(d_ff, C) * 0.02
b2 = np.zeros((1, C))

pe = positional_encoding(T, C)[np.newaxis, :]

src_emb = np.take(embed_W_src, src_ids, axis=0) + pe
tgt_emb = np.take(embed_W_tgt, tgt_ids, axis=0) + pe

for _ in range(N):
    src_emb = encoder_layer(src_emb, Wq, Wk, Wv, Wo, W1, b1, W2, b2, num_heads)

for _ in range(N):
    tgt_emb = decoder_layer(tgt_emb, src_emb, self_Wq, self_Wk, self_Wv, self_Wo, encdec_Wq, encdec_Wk, encdec_Wv, encdec_Wo, W1, b1, W2, b2, num_heads)

logits = np.matmul(tgt_emb, gen_W)
probs = softmax(logits, axis=-1)

print("src_emb shape:", src_emb.shape)
print("tgt_emb shape:", tgt_emb.shape)
print("probs shape:", probs.shape)
print("Sample probs:", probs[0,0,:3])
print("All finite:", np.all(np.isfinite(probs)))
