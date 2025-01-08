import torch
import torch.nn as nn
import torch.nn.init as init
from pyctcdecode import build_ctcdecoder

torch.autograd.set_detect_anomaly(True)

device = "cuda" if torch.cuda.is_available() else "cpu"



class Swish(nn.SiLU):
    pass

class SpecAugment(nn.Module):
    """
    Zeroes out (cuts) random continuous horizontal or vertical segments of the spectrogram
    as described in SpecAugment (https://arxiv.org/abs/1904.08779).

    Args:
        freq_masks (int): Number of frequency segments to cut.
        time_masks (int): Number of time segments to cut.
        freq_width (int): Maximum number of frequencies to cut in one segment.
        time_width (float or int): Maximum number of time steps to cut in one segment.
            If a positive integer, defines the maximum number of time steps to cut.
            If a float in [0, 1], defines the maximum percentage of time steps to cut adaptively.
        mask_value (float): Value to use for masking.

    -- From NeMo
    """

    def __init__(
        self, freq_masks=0, time_masks=0, freq_width=10, time_width=0.1, mask_value=0.0
    ):
        super().__init__()

        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width
        self.mask_value = mask_value

        if isinstance(time_width, float):
            if not (0 <= time_width <= 1):
                raise ValueError(
                    "If `time_width` is a float, it must be in the range [0, 1]."
                )
            self.adaptive_temporal_width = True
        else:
            self.adaptive_temporal_width = False

    def forward(self, input_spec, length):
        """
        Apply SpecAugment to the input spectrogram.

        Args:
            input_spec (torch.Tensor): Input spectrogram of shape [B, F, T].
            length (torch.Tensor): Length of each sequence in the batch.

        Returns:
            torch.Tensor: Spectrogram with SpecAugment applied.
        """
        device = input_spec.device
        batch_size, freq_dim, time_dim = input_spec.shape

        # Frequency masking
        if self.freq_masks > 0:
            freq_starts = torch.randint(
                0,
                max(1, freq_dim - self.freq_width + 1),
                (batch_size, self.freq_masks),
                device=device,
            )
            freq_lengths = torch.randint(
                1, self.freq_width + 1, (batch_size, self.freq_masks), device=device
            )
            for b in range(batch_size):
                for f_start, f_width in zip(freq_starts[b], freq_lengths[b]):
                    f_end = f_start + f_width
                    input_spec[b, f_start:f_end, :] = self.mask_value

        # Time masking
        if self.time_masks > 0:
            if self.adaptive_temporal_width:
                time_widths = (length * self.time_width).int().clamp(min=1)
            else:
                time_widths = torch.full(
                    (batch_size,), self.time_width, dtype=torch.int64, device=device
                )

            time_starts = [
                torch.randint(
                    0, max(1, l - t_width + 1), (self.time_masks,), device=device
                )
                for l, t_width in zip(length, time_widths)
            ]
            time_lengths = [
                torch.randint(1, t_width + 1, (self.time_masks,), device=device)
                for t_width in time_widths
            ]
            for b in range(batch_size):
                for t_start, t_width in zip(time_starts[b], time_lengths[b]):
                    t_end = t_start + t_width
                    input_spec[b, :, t_start:t_end] = self.mask_value

        return input_spec


def calc_length(lengths, all_paddings, kernel_size, stride, repeat_num=1):
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = (
            torch.div(
                lengths.to(dtype=torch.float) + add_pad, stride, rounding_mode="floor"
            )
            + one
        )
    return lengths.to(dtype=torch.int)


class ConvSubsampling(nn.Module):
    """
    Convsubsampling -> Linear -> Dropout.
    """

    def __init__(self, input_dim, output_dim, dropout=0):
        super(ConvSubsampling, self).__init__()
        self.conv1 = nn.Conv2d(1, output_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(
            output_dim, output_dim, kernel_size=3, stride=2, padding=1
        )
        self.linear = nn.Linear(output_dim * 20, output_dim)  # 20 is 80 // 4
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, lengths):
        """
        Args:
            x: Input tensor of shape `(batch_size, feature_dim, time_steps)`.
            lengths: Tensor of shape `(batch_size,)` representing original sequence lengths.
        Returns:
            subsampled_x: Output tensor of shape `(batch_size, time_steps//4, output_dim)`.
            subsampled_lengths: Adjusted lengths after subsampling.
        """
        x = x.unsqueeze(1)  # (B, 1, F, T)
        x = self.activation(self.conv1(x))  # (B, C, F', T')
        x = self.activation(self.conv2(x))  # (B, C, F'', T'')
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T'', C * F'')
        x = self.linear(x)  # (B, T'', D)
        x = self.dropout(x)

        # # Adjust lengths for subsampling
        # lengths = torch.div(lengths - 1, 2, rounding_mode="floor") + 1  # After first conv
        # lengths = torch.div(lengths - 1, 2, rounding_mode="floor") + 1  # After second conv

        lengths = calc_length(
            lengths=lengths, all_paddings=2, kernel_size=3, stride=2, repeat_num=2
        )  # repeat_num = 2 because we have 2 conv layers

        return x, lengths.to(dtype=torch.int)


# Blocks
class FFN(nn.Module):
    """
    LayerNorm -> Linear -> Swish activation -> Dropout -> Linear -> Dropout
    => With Residual connection
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish(), use_bias=True):
        super(FFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias

        self.norm = nn.LayerNorm(normalized_shape=d_model)
        self.linear1 = nn.Linear(d_model, d_model * d_ff, bias=self.use_bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_model * d_ff, d_model, bias=self.use_bias)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = 0.5 * x
        x = x + residual

        return x


class MHSA(nn.Module):
    """
    LayerNorm -> Multi-Head Attention with Relative Positional Encoding -> Dropout
    => With Residual connection
    """

    def __init__(self, d_model, num_heads, dropout, dropout_att):
        super(MHSA, self).__init__()

        self.norm = nn.LayerNorm(normalized_shape=d_model)

        # self.q_proj = nn.Linear(d_model, d_model)
        # self.k_proj = nn.Linear(d_model, d_model)
        # self.v_proj = nn.Linear(d_model, d_model)

        self.multihead_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout_att, batch_first=True
        )  # no RoPE for now
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        residual = x
        x = self.norm(x)

        # q = self.q_proj(x)
        # k = self.k_proj(x)
        # v = self.v_proj(x)
        # x, _ = self.multihead_attn(q, k, v)

        x, _ = self.multihead_attn(x, x, x)

        x = self.dropout(x)
        x = x + residual

        return x


class Conv(nn.Module):
    """
    LayerNorm -> Pointwise Conv -> Glu activation -> 1D depthwise conv -> BN -> Swish activation -> Pointwise Conv -> Dropout
    => With Residual connection
    """

    def __init__(
        self, d_model, kernel_size, pointwise_activation="glu_", use_bias=True
    ):
        super(Conv, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "Kernel size must be odd."

        self.pointwise_activation = (
            nn.GLU(dim=1)
            if pointwise_activation == "glu_"
            else getattr(nn, pointwise_activation)()
        )
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2 if pointwise_activation == "glu_" else d_model,
            kernel_size=1,
            bias=use_bias,
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=(kernel_size - 1) // 2,
            bias=use_bias,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            bias=use_bias,
        )
        self.activation = nn.SiLU()  # or we can use Swish() from above

    def forward(self, x):
        """
        Forward pass of the Conv module.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, time_steps, d_model)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, time_steps, d_model)`.
        """
        residual = x
        x = x.transpose(1, 2)  # [batch, d_model, time_steps]
        x = self.pointwise_conv1(x)
        x = self.pointwise_activation(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)  # [batch, time_steps, d_model]
        x = x + residual
        return x


class ConformerBlock(nn.Module):
    def __init__(self, config):
        super(ConformerBlock, self).__init__()

        self.ffn1 = FFN(
            d_model=config["d_model"],
            d_ff=config["ff_expansion_factor"],
            dropout=config["dropout"],
        )

        self.mhsa = MHSA(
            d_model=config["d_model"],
            num_heads=config["n_heads"],
            dropout=config["dropout"],
            dropout_att=config["dropout_att"],
        )

        self.conv = Conv(
            d_model=config["d_model"],
            kernel_size=config["conv_kernel_size"],
            pointwise_activation="glu_",
            # conv_norm_type
        )

        self.ffn2 = FFN(
            d_model=config["d_model"],
            d_ff=config["ff_expansion_factor"],
            dropout=config["dropout"],
        )

        self.norm = nn.LayerNorm(normalized_shape=config["d_model"])

    def forward(self, x):
        """
        Forward pass of the Conv module.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, time_steps, d_model)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, time_steps, d_model)`.
        """
        x = self.ffn1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        ffn_out = self.ffn2(x)
        x = x + ffn_out
        y = self.norm(x)
        return y


class ConformerEncoder(nn.Module):
    def __init__(self, config):
        super(ConformerEncoder, self).__init__()
        self.config = config

        self.spec_augment = SpecAugment(
            freq_masks=2, time_masks=5, freq_width=15, time_width=0.05
        )  # input [B, F, T] | output [B, F, T] # check param values from https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/conformer/conformer_ctc_bpe.yaml
        self.conv_subsampling = ConvSubsampling(  # input [B, F, T]
            input_dim=self.config["feat_in"],
            output_dim=self.config["d_model"],
            dropout=self.config["dropout_pre_encoder"],
        )
        self.layers = nn.ModuleList(
            [ConformerBlock(self.config) for _ in range(config["n_layers"])]
        )

    def forward(self, x, lengths=None):
        """
        Forward pass of the ConformerEncoder module.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, n_spectrogram, time_steps)`.
            lengths (torch.Tensor, optional): Tensor containing the lengths of each input in the batch.
                Shape: `(batch_size,)`. Default is `None`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, time_steps, d_model)`.
        """
        if lengths is not None:
            x = self.spec_augment(x, lengths)

        x, lengths = self.conv_subsampling(x, lengths)

        for layer in self.layers:
            x = layer(x)

        return x, lengths


class Conformer(nn.Module):
    def __init__(self, config, vocab_size):
        super(Conformer, self).__init__()
        self.config = config
        self.encoder = ConformerEncoder(config)
        self.decoder = nn.Linear(config["d_model"], vocab_size + 1)

    def forward(self, x, lengths=None):
        """
        Forward pass of the Conformer model.
        Args:
            x: Input tensor of shape (batch_size, time_steps, freq_dim).
            lengths: Tensor of input lengths for each spectrogram in the batch.
        Returns:
            logits: Output tensor of shape (batch_size, time_steps, vocab_size).
            lengths: Tensor of output lengths for each sequence in the batch.
        """
        x, lengths = self.encoder(x, lengths)
        logits = self.decoder(x)
        return logits, lengths

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path, weights_only=True))

    def infer(self, spectrogram, spectrogram_length, tokenizer, decoding_strategy="greedy", beam_width=5, kenlm_path=None):
        """
        Perform inference using the model and tokenizer.

        Args:
            spectrogram: Input spectrogram tensor of shape (batch_size, freq_dim, time_steps).
            spectrogram_length: Tensor of input lengths for each spectrogram in the batch.
            tokenizer: Instance of the Tokenizer class for encoding and decoding.
            decoding_strategy: Decoding strategy ("greedy" or "beam").
            beam_width: Width of the beam for beam search decoding.
            kenlm_path: Path to a KenLM language model (only used for beam search).

        Returns:
            List of decoded text sequences for each sample in the batch.
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(spectrogram, spectrogram_length)

        probs = torch.softmax(logits, dim=-1)

        if decoding_strategy == "beam":
            labels = [tokenizer.get_id_piece(i) for i in range(tokenizer.get_vocab_size())]

            decoder = build_ctcdecoder(labels, kenlm_model_path=kenlm_path)

            decoded_texts = [
                decoder.decode(probs[i].cpu().numpy(), beam_width=beam_width) for i in range(probs.size(0))
            ]
        elif decoding_strategy == "greedy":
            decoded_texts = []
            for i in range(probs.size(0)):
                pred_tokens = torch.argmax(probs[i], dim=-1).cpu().tolist()

                merged_tokens = []
                prev_token = None
                for token in pred_tokens:
                    if token != prev_token:  # Collapse repeats
                        merged_tokens.append(token)
                    prev_token = token

                final_tokens = [token for token in merged_tokens if token != tokenizer.blank_token]

                decoded_text = tokenizer.decode(final_tokens)
                decoded_texts.append(decoded_text)
        else:
            raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")

        return decoded_texts



def kaiming_init(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                init.zeros_(layer.bias)


def xavier_init(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            init.xavier_uniform_(layer.weight) 
            if layer.bias is not None:
                init.zeros_(layer.bias)


def init_weights(model, init_type='xavier'):
    for layer in model.modules():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            if init_type == 'xavier':
                init.xavier_uniform_(layer.weight)
            elif init_type == 'kaiming':
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                init.zeros_(layer.bias)