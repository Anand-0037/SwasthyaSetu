import torch
import torch.nn as nn
import math
from typing import Tuple, List, Optional

class PerspectiveEmbedding(nn.Module):
    """
    Embedding layer that incorporates perspective information
    """
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000):
        super(PerspectiveEmbedding, self).__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Perspective embeddings
        self.perspective_embedding = nn.Embedding(3, d_model)  # 0: neutral, 1: patient, 2: clinician
        
        # Positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, tokens: torch.Tensor, perspective: int = 0) -> torch.Tensor:
        """
        Args:
            tokens: Token IDs [batch_size, seq_len]
            perspective: 0=neutral, 1=patient, 2=clinician
        """
        batch_size, seq_len = tokens.shape
        
        # Token embeddings
        token_emb = self.token_embedding(tokens) * math.sqrt(self.d_model)
        
        # Perspective embeddings
        perspective_tensor = torch.full((batch_size, seq_len), perspective, 
                                      dtype=torch.long, device=tokens.device)
        perspective_emb = self.perspective_embedding(perspective_tensor)
        
        # Positional encoding
        pos_emb = self.pe[:, :seq_len]
        
        # Combine embeddings
        embeddings = token_emb + perspective_emb + pos_emb
        return self.dropout(embeddings)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        return output, attn_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # Final linear layer
        output = self.w_o(attn_output)
        return output, attn_weights

class PerspectiveAwareDecoderLayer(nn.Module):
    """
    Decoder layer with perspective-aware attention
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(PerspectiveAwareDecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Perspective-aware feed forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Perspective gate - learns to weight information based on perspective
        self.perspective_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None, perspective_emb=None):
        # Self attention
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross attention
        cross_attn_output, cross_attn_weights = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        
        # Apply perspective gate if perspective embedding is provided
        if perspective_emb is not None:
            # Combine cross attention with perspective information
            combined = torch.cat([cross_attn_output, perspective_emb], dim=-1)
            gate = self.perspective_gate(combined)
            cross_attn_output = cross_attn_output * gate
        
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed forward
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, cross_attn_weights

class PerspectiveAwareMedicalSummarizer(nn.Module):
    """
    Enhanced medical summarizer with perspective awareness
    """
    def __init__(self, vocab_size: int, d_model: int = 256, num_heads: int = 4, 
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, 
                 d_ff: int = 512, dropout: float = 0.1, max_len: int = 5000):
        super(PerspectiveAwareMedicalSummarizer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.src_embedding = PerspectiveEmbedding(vocab_size, d_model, max_len)
        self.tgt_embedding = PerspectiveEmbedding(vocab_size, d_model, max_len)
        
        # Encoder layers (standard transformer encoder)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
            for _ in range(num_encoder_layers)
        ])
        
        # Perspective-aware decoder layers
        self.decoder_layers = nn.ModuleList([
            PerspectiveAwareDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Perspective-specific heads
        self.patient_head = nn.Linear(d_model, vocab_size)
        self.clinician_head = nn.Linear(d_model, vocab_size)
        
        # Copy mechanism
        self.copy_gate = nn.Linear(d_model * 2, 1)
        
        # Provenance and faithfulness heads
        self.provenance_head = nn.Linear(d_model, 1)
        self.faithfulness_head = nn.Linear(d_model, 1)
        
        # Perspective classifier (to determine which perspective to use)
        self.perspective_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),  # neutral, patient, clinician
            nn.Softmax(dim=-1)
        )
        
    def generate_masks(self, src, tgt):
        """Generate attention masks"""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                perspective: int = 0) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with perspective awareness
        
        Args:
            src: Source tokens [batch_size, src_len]
            tgt: Target tokens [batch_size, tgt_len]
            perspective: 0=neutral, 1=patient, 2=clinician
        """
        src_mask, tgt_mask = self.generate_masks(src, tgt)
        
        # Encode source with neutral perspective
        src_embedded = self.src_embedding(src, perspective=0)
        encoder_output = src_embedded
        
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, src_key_padding_mask=None)
        
        # Decode with specified perspective
        tgt_embedded = self.tgt_embedding(tgt, perspective=perspective)
        decoder_output = tgt_embedded
        
        # Get perspective embedding for gating
        batch_size, seq_len = tgt.shape
        perspective_tensor = torch.full((batch_size, seq_len), perspective, 
                                      dtype=torch.long, device=tgt.device)
        perspective_emb = self.tgt_embedding.perspective_embedding(perspective_tensor)
        
        cross_attn_weights_list = []
        
        for decoder_layer in self.decoder_layers:
            decoder_output, cross_attn_weights = decoder_layer(
                decoder_output, encoder_output, src_mask, tgt_mask, perspective_emb
            )
            cross_attn_weights_list.append(cross_attn_weights)
        
        # Generate outputs based on perspective
        if perspective == 1:  # Patient perspective
            vocab_dist = self.patient_head(decoder_output)
        elif perspective == 2:  # Clinician perspective
            vocab_dist = self.clinician_head(decoder_output)
        else:  # Neutral or auto-detect
            vocab_dist = self.output_projection(decoder_output)
        
        # Copy mechanism
        avg_cross_attn = torch.mean(torch.stack(cross_attn_weights_list), dim=[0, 2])
        context_vector = torch.bmm(avg_cross_attn, encoder_output)
        
        combined_context = torch.cat([decoder_output, context_vector], dim=-1)
        copy_gate = torch.sigmoid(self.copy_gate(combined_context))
        
        # Provenance scores
        provenance_scores = torch.sigmoid(self.provenance_head(decoder_output))
        
        # Faithfulness score
        pooled_output = torch.mean(decoder_output, dim=1)
        faithfulness_score = torch.sigmoid(self.faithfulness_head(pooled_output))
        
        # Perspective classification (for auto-detection)
        perspective_probs = self.perspective_classifier(pooled_output)
        
        return (vocab_dist, copy_gate, provenance_scores, faithfulness_score, 
                avg_cross_attn, perspective_probs)
    
    def generate_summary(self, src: torch.Tensor, max_length: int = 64, 
                        perspective: int = 0, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate summary with specified perspective
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # Start with start token
        generated = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)  # Assuming 2 is START token
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                outputs = self.forward(src, generated, perspective)
                vocab_dist = outputs[0]
                
                # Apply temperature
                next_token_logits = vocab_dist[:, -1, :] / temperature
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(next_token_probs, 1)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if end token is generated
                if next_token.item() == 3:  # Assuming 3 is END token
                    break
        
        return generated

class TinyPerspectiveAwareMedicalSummarizer(PerspectiveAwareMedicalSummarizer):
    """Tiny version for demo and fast training"""
    def __init__(self, vocab_size: int, d_model: int = 256, num_heads: int = 4, 
                 num_encoder_layers: int = 2, num_decoder_layers: int = 2, 
                 d_ff: int = 512, dropout: float = 0.1, max_len: int = 5000):
        super(TinyPerspectiveAwareMedicalSummarizer, self).__init__(
            vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers,
            d_ff, dropout, max_len
        )

