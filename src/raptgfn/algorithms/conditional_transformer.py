"""Conditional Transformer for aptamer sequence generation."""

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(self, in_dim, out_dim, hidden_layers, dropout_prob, init_drop=False):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        layers = [nn.Linear(in_dim, hidden_layers[0]), nn.ReLU()]
        layers += [nn.Dropout(dropout_prob)] if init_drop else []
        for i in range(1, len(hidden_layers)):
            layers.extend(
                [
                    nn.Linear(hidden_layers[i - 1], hidden_layers[i]),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                ]
            )
        layers.append(nn.Linear(hidden_layers[-1], out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x, with_uncertainty=False):
        return self.model(x)


class CondGFNTransformer(nn.Module):
    """Transformer policy conditioned on GFlowNet objectives."""

    def __init__(
        self,
        max_len,
        vocab_size,
        num_actions,
        num_hid=64,
        num_layers=3,
        num_head=8,
        dropout=0.0,
        batch_size=128,
        bidirectional=False,
        **kwargs
    ):
        super().__init__()

        # Compute the conditioning dimension from the enabled encodings.
        # Thermometer encoding expands each scalar into therm_n_bins features.
        n_objectives = len(kwargs.get("objectives", ["length", "gc_content"]))
        therm_n_bins = kwargs.get("therm_n_bins", 50)
        beta_use_therm = kwargs.get("beta_use_therm", True)
        pref_use_therm = kwargs.get("pref_use_therm", True)

        beta_dim = therm_n_bins if beta_use_therm else 1
        pref_dim = n_objectives * therm_n_bins if pref_use_therm else n_objectives
        self.cond_dim = beta_dim + pref_dim

        self.use_cond = True
        self.num_hid = num_hid

        self.pos = PositionalEncoding(num_hid, dropout=dropout, max_len=max_len + 2)
        self.embedding = nn.Embedding(vocab_size, num_hid)
        encoder_layers = nn.TransformerEncoderLayer(
            num_hid, num_head, num_hid, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Use the conditional representation.
        self.output = MLP(num_hid + num_hid, num_actions, [4 * num_hid, 4 * num_hid], dropout)
        self.cond_embed = nn.Linear(self.cond_dim, num_hid)
        # Represent log Z with the configured linear, MLP, or bias-only head.
        self.Z_type = kwargs.get("Z_type", "linear")
        if self.Z_type == "mlp":
            self.Z_mod = MLP(self.cond_dim, num_hid, [2 * num_hid, num_hid], dropout)
        elif self.Z_type == "bias_only":
            self.Z_mod = nn.Linear(self.cond_dim, 1, bias=True)
        else:
            self.Z_mod = nn.Linear(self.cond_dim, num_hid)

        self.logsoftmax2 = torch.nn.LogSoftmax(2)
        # Optional learned backward policy.
        self.enable_backward = kwargs.get("enable_backward", False)
        if self.enable_backward:
            # Predict backward-action probabilities from the conditioned state representation.
            self.bwd_head = MLP(num_hid + num_hid, num_actions, [2 * num_hid, num_hid], dropout)

    def Z(self, cond_var):
        """Return the conditional log-partition representation."""
        z = self.Z_mod(cond_var)
        if self.Z_type == "bias_only":
            return z.squeeze(1)
        return z.sum(1)

    def model_params(self):
        """Return parameters belonging to the policy model."""
        params = (
            list(self.pos.parameters())
            + list(self.embedding.parameters())
            + list(self.encoder.parameters())
            + list(self.output.parameters())
            + list(self.cond_embed.parameters())
        )
        if getattr(self, "enable_backward", False):
            params += list(self.bwd_head.parameters())
        return params

    def Z_param(self):
        """Return parameters belonging to the partition-function head."""
        return self.Z_mod.parameters()

    def forward(self, x, cond, mask, return_all=False, lens=None, logsoftmax=False):
        """Run the forward pass.
        
        Args:
            x: Input tokens with shape [batch_size, seq_len].
            cond: Conditions with shape [batch_size, cond_dim].
            mask: Padding mask with shape [batch_size, seq_len].
            return_all: Whether to return outputs at every position.
            lens: Sequence lengths with shape [batch_size].
            logsoftmax: Whether to apply log softmax.
        """
        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, num_hid]
        x = self.pos(x)

        # Transformer encoder
        # Convert the mask for the batch-first layout.
        src_key_padding_mask = mask

        # Causal mask
        seq_len = x.shape[1]
        causal_mask = generate_square_subsequent_mask(seq_len).to(x.device)

        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)

        # Select the last valid position for each sequence.
        if lens is not None:
            batch_indices = torch.arange(x.shape[0], device=x.device)
            pooled_x = x[batch_indices, lens - 1]
        else:
            pooled_x = x[:, -1]  # Use the final position.

        # Embed the conditioning vector.
        cond_var = self.cond_embed(cond)  # [batch_size, num_hid]

        if return_all:
            # Broadcast the conditions over all positions.
            cond_var_expanded = cond_var.unsqueeze(1).expand(-1, x.shape[1], -1)
            final_rep = torch.cat((x, cond_var_expanded), axis=-1)
            out = self.output(final_rep)
            return self.logsoftmax2(out) if logsoftmax else out
        else:
            final_rep = torch.cat((pooled_x, cond_var), axis=-1)
            y = self.output(final_rep)
            return y

    def backward_logits(self, x, cond, mask, return_all=False, lens=None, logsoftmax=False):
        """Compute the backward-policy output.
        
        Raise RuntimeError when enable_backward is false.
        Output shapes follow the forward method's return_all convention.
        """
        if not getattr(self, "enable_backward", False):
            raise RuntimeError("Backward head is disabled. Set enable_backward=True to use it.")
        # Reuse the embedding, positional encoding, and Transformer encoder.
        x = self.embedding(x)
        x = self.pos(x)
        src_key_padding_mask = mask
        seq_len = x.shape[1]
        causal_mask = generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        if lens is not None:
            batch_indices = torch.arange(x.shape[0], device=x.device)
            pooled_x = x[batch_indices, lens - 1]
        else:
            pooled_x = x[:, -1]
        cond_var = self.cond_embed(cond)
        if return_all:
            cond_var_expanded = cond_var.unsqueeze(1).expand(-1, x.shape[1], -1)
            final_rep = torch.cat((x, cond_var_expanded), axis=-1)
            out = self.bwd_head(final_rep)
            return self.logsoftmax2(out) if logsoftmax else out
        else:
            final_rep = torch.cat((pooled_x, cond_var), axis=-1)
            y = self.bwd_head(final_rep)
            return y


def generate_square_subsequent_mask(sz: int):
    """Construct a causal attention mask."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Reshape the encoding for batch-first inputs.
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
