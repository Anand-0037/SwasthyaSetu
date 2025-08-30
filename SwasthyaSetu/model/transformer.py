import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        return output, attn_probs # Return attention probabilities for copy mechanism

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        output, attn_probs = self.scaled_dot_product_attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        return output, attn_probs # Return attention probabilities

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        cross_attn_output, cross_attn_probs = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x, cross_attn_probs # Return cross-attention probabilities

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_len=5000):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)
        seq_len = tgt.shape[1]
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src) * math.sqrt(self.d_model)))
        enc_output = src_embedded
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_output, src_mask)

        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt) * math.sqrt(self.d_model)))
        dec_output = tgt_embedded
        cross_attn_probs_list = []
        for decoder_layer in self.decoder_layers:
            dec_output, cross_attn_probs = decoder_layer(dec_output, enc_output, src_mask, tgt_mask)
            cross_attn_probs_list.append(cross_attn_probs)

        output = self.fc_out(dec_output)
        return output, cross_attn_probs_list

# Tiny CPU variant for demo
class TinyTransformer(Transformer):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=4, num_layers=2, d_ff=512, dropout=0.1, max_len=5000):
        super(TinyTransformer, self).__init__(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_len)


class CopyMechanism(nn.Module):
    def __init__(self, d_model):
        super(CopyMechanism, self).__init__()
        self.linear_p_gen = nn.Linear(d_model * 2, 1) # Concatenate decoder output and context vector

    def forward(self, decoder_output, cross_attn_probs_list, encoder_output):
        # cross_attn_probs_list: list of (batch_size, num_heads, tgt_len, src_len)
        # Average cross-attention probabilities across heads and layers
        # This will be (batch_size, tgt_len, src_len)
        # Stack and then mean over the layer dimension (dim=0) and head dimension (dim=1)
        # The current avg_cross_attn_probs has shape (batch_size, tgt_len, src_len)
        # The bmm operation requires (batch_size, M, K) and (batch_size, K, N) -> (batch_size, M, N)
        # Here, M = tgt_len, K = src_len, N = d_model
        # So, avg_cross_attn_probs is (batch_size, tgt_len, src_len)
        # and encoder_output is (batch_size, src_len, d_model)
        # This is correct for bmm.

        # The error was likely due to `torch.mean(torch.stack(cross_attn_probs_list), dim=[0, 1])`
        # where dim=[0,1] was averaging over batch and heads, which is incorrect.
        # It should average over the layers (dim=0 after stack) and then heads (dim=1).
        # Let's fix the averaging for cross_attn_probs_list

        # Stack all cross_attn_probs from different decoder layers
        # Shape: (num_layers, batch_size, num_heads, tgt_len, src_len)
        stacked_cross_attn_probs = torch.stack(cross_attn_probs_list)

        # Average across layers (dim=0) and heads (dim=1)
        # Resulting shape: (batch_size, tgt_len, src_len)
        avg_cross_attn_probs = torch.mean(stacked_cross_attn_probs, dim=[0, 2]) # Average over layers and heads

        context_vector = torch.bmm(avg_cross_attn_probs, encoder_output)

        combined_input = torch.cat((decoder_output, context_vector), dim=-1)
        p_gen = torch.sigmoid(self.linear_p_gen(combined_input)) # (batch_size, tgt_len, 1)
        return p_gen, avg_cross_attn_probs

class ProvenanceHead(nn.Module):
    def __init__(self, d_model):
        super(ProvenanceHead, self).__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, decoder_output):
        # Simple linear layer to predict provenance score for each token
        return torch.sigmoid(self.linear(decoder_output))

class FaithfulnessClassifier(nn.Module):
    def __init__(self, d_model):
        super(FaithfulnessClassifier, self).__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, decoder_output):
        # Simple linear layer to predict faithfulness score for the summary
        # This would typically operate on a pooled representation of the summary
        # For now, we will take the mean of the decoder output
        pooled_output = torch.mean(decoder_output, dim=1)
        return torch.sigmoid(self.linear(pooled_output))


class MedicalSummarizer(Transformer):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=4, num_layers=6, d_ff=512, dropout=0.1, max_len=5000):
        super(MedicalSummarizer, self).__init__(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_len)
        self.copy_mechanism = CopyMechanism(d_model)
        self.provenance_head = ProvenanceHead(d_model)
        self.faithfulness_classifier = FaithfulnessClassifier(d_model)

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src) * math.sqrt(self.d_model)))
        enc_output = src_embedded
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_output, src_mask)

        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt) * math.sqrt(self.d_model)))
        dec_output = tgt_embedded
        cross_attn_probs_list = []
        for decoder_layer in self.decoder_layers:
            dec_output, cross_attn_probs = decoder_layer(dec_output, enc_output, src_mask, tgt_mask)
            cross_attn_probs_list.append(cross_attn_probs)

        # Generate output vocabulary distribution
        output_vocab_dist = self.fc_out(dec_output)

        # Copy mechanism
        p_gen, attn_dist = self.copy_mechanism(dec_output, cross_attn_probs_list, enc_output) # (batch_size, tgt_len, 1)

        # Provenance head
        provenance_scores = self.provenance_head(dec_output) # (batch_size, tgt_len, 1)

        # Faithfulness classifier
        faithfulness_score = self.faithfulness_classifier(dec_output) # (batch_size, 1)

        return output_vocab_dist, p_gen, provenance_scores, faithfulness_score, attn_dist


class TinyMedicalSummarizer(MedicalSummarizer):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=4, num_layers=2, d_ff=512, dropout=0.1, max_len=5000):
        super(TinyMedicalSummarizer, self).__init__(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_len)


