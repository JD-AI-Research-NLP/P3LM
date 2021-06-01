import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    MultiheadAttention,
    LayerNorm,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from .learned_positional_embedding import LearnedPositionalEmbedding
from .ngram_multihead_attention import NgramMultiheadAttention, ngram_attention_bias, makeOrder, makeOrder_ngram
import pdb

DEFAULT_MAX_SOURCE_POSITIONS = 512
DEFAULT_MAX_TARGET_POSITIONS = 512


@register_model('jdnet_mrc')
class JDNetMRCModel(FairseqEncoderDecoderModel):
    """
    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder
    The Transformer model provides the following named architectures and
    command-line arguments:
    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder, mlmDecoder):
        super().__init__(encoder, decoder)
        self.mlmDecoder = mlmDecoder
        self.train_mlm = args.train_mlm

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--ngram', type=int, metavar='N',
                            help='num of predicting grams')
        parser.add_argument('--num_buckets', type=int, metavar='N',
                            help='num of buckets for relative position')
        parser.add_argument('--relative_max_distance', type=int, metavar='N',
                            help='num of bucket for relative position')
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')

        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')

        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')

        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--load-from-pretrained-model', type=str, default=None,
                            help='Load from pretrained model')
        parser.add_argument('--load-from-pretrained-mlm-model', type=str, default=None,
                            help='Load from pretrained mlm model')
        parser.add_argument('--load-sep', action='store_true',
                            help='load pretrained [SEP] weight into [X_SEP]. ([SEP] used as eos in fine tuning)')
        parser.add_argument('--train-mlm', action='store_true',
                            help='pretrain mlm')
        # fmt: on

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
#             mlmDecoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim
            )
            mlmDecoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim
            )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = NgramTransformerDecoder(args, tgt_dict, decoder_embed_tokens)
#         mlmDecoder = MLMDecoder(args, tgt_dict, mlmDecoder_embed_tokens)

#         model = JDNetMRCModel(args, encoder, decoder, mlmDecoder)
        model = JDNetMRCModel(args, encoder, decoder, None)

#         pdb.set_trace()
        if args.load_from_pretrained_model is not None:
            print('Load Pretrained model from %s' % args.load_from_pretrained_model)
            states = torch.load(args.load_from_pretrained_model, map_location='cpu')
            if 'model' in states and 'args' in states:
                states = states['model']
            if args.load_sep:
                encoder_token_weight = states['encoder.embed_tokens.weight']
                decoder_token_weight = states['decoder.embed_tokens.weight']
                encoder_token_weight[2] = encoder_token_weight[102]
                decoder_token_weight[2] = decoder_token_weight[102]
                states['encoder.embed_tokens.weight'] = encoder_token_weight
                states['decoder.embed_tokens.weight'] = decoder_token_weight

            #----------wei zhen
            pretrained_ngram = states['decoder.ngram_input_embed.weight'].size(0)
            desired_ngram = model.decoder.ngram
            embed_dim = states['decoder.ngram_input_embed.weight'].size(1)
            if pretrained_ngram > desired_ngram:
                states['decoder.ngram_input_embed.weight'] = states['decoder.ngram_input_embed.weight'][:desired_ngram,:]
            elif pretrained_ngram < desired_ngram:
                num_grams_to_add = desired_ngram - pretrained_ngram
                new_grams_embed_to_add = torch.zeros(num_grams_to_add, embed_dim)
                nn.init.normal_(
                    new_grams_embed_to_add,
                    mean=0,
                    std=embed_dim ** -0.5
                )
                new_grams_embed_to_add = new_grams_embed_to_add.to(
                    dtype=states['decoder.ngram_input_embed.weight'].dtype,
                )
                states['decoder.ngram_input_embed.weight'] = torch.cat([
                    states['decoder.ngram_input_embed.weight'],
                    new_grams_embed_to_add]
                )

            loaded_dict_size = states['encoder.embed_tokens.weight'].size(0)
            num_langids_to_add = len(encoder.dictionary) - loaded_dict_size
            embed_dim = states['encoder.embed_tokens.weight'].size(1)

            if num_langids_to_add > 0:
                new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
                nn.init.normal_(
                    new_lang_embed_to_add,
                    mean=0,
                    std=embed_dim ** -0.5
                )
                new_lang_embed_to_add = new_lang_embed_to_add.to(
                    dtype=states['encoder.embed_tokens.weight'].dtype,
                )

                states['encoder.embed_tokens.weight'] = torch.cat([
                    states['encoder.embed_tokens.weight'],
                    new_lang_embed_to_add]
                )
                states['decoder.embed_tokens.weight'] = torch.cat([
                    states['decoder.embed_tokens.weight'],
                    new_lang_embed_to_add]
                )            
            #----------wei zhen
            
            for position_name, target_position_length in [('encoder.embed_positions.weight', model.encoder.embed_positions.weight.size(0)), \
                    ('decoder.embed_positions.weight', model.decoder.embed_positions.weight.size(0))]:
                if states[position_name].size(0) < target_position_length:
                    _index = torch.arange(states[position_name].size(1))
                    expend_position_states = states[position_name].clone()
                    while states[position_name].size(0) < target_position_length:
                        _index = torch.cat((_index[1:],_index[:1]), dim=0)
                        states[position_name] = torch.cat([states[position_name], expend_position_states[:,_index]], dim=0)
                if states[position_name].size(0) > target_position_length:
                    states[position_name] = states[position_name][:target_position_length]
            model.load_state_dict(states, strict=False)
            args.load_from_pretrained_model = None  # Clear this param

        #----------Junwei
#         pdb.set_trace()
        if args.load_from_pretrained_mlm_model is not None:
            print('Load Pretrained model from %s' % args.load_from_pretrained_mlm_model)
            states = torch.load(args.load_from_pretrained_mlm_model, map_location='cpu')
            if 'model' in states and 'args' in states:
                states = states['model']
            if args.load_sep:
                encoder_token_weight = states['encoder.embed_tokens.weight']
                decoder_token_weight = states['decoder.embed_tokens.weight']
                encoder_token_weight[2] = encoder_token_weight[102]
                decoder_token_weight[2] = decoder_token_weight[102]
                states['encoder.embed_tokens.weight'] = encoder_token_weight
                states['decoder.embed_tokens.weight'] = decoder_token_weight
            
            for position_name, target_position_length in [('encoder.embed_positions.weight', model.encoder.embed_positions.weight.size(0)), \
                    ('decoder.embed_positions.weight', model.decoder.embed_positions.weight.size(0))]:
                if states[position_name].size(0) < target_position_length:
                    _index = torch.arange(states[position_name].size(1))
                    expend_position_states = states[position_name].clone()
                    while states[position_name].size(0) < target_position_length:
                        _index = torch.cat((_index[1:],_index[:1]), dim=0)
                        states[position_name] = torch.cat([states[position_name], expend_position_states[:,_index]], dim=0)
                if states[position_name].size(0) > target_position_length:
                    states[position_name] = states[position_name][:target_position_length]
            model.load_state_dict(states, strict=False)
            args.load_from_pretrained_mlm_model = None  # Clear this param
        #----------Junwei
        
#         return JDNetMRCModel(args, encoder, decoder, mlmDecoder)        
        return JDNetMRCModel(args, encoder, decoder, None)

    def max_positions(self):
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def forward(self, src_tokens=None, src_lengths=None, prev_output_tokens=None, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.
        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::
            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        if 'EncodeOnly' in kwargs:
            encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
            return encoder_out
        elif 'ARLMOnly' in kwargs:
            encoder_out = kwargs['encode_out']
            decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
            return decoder_out
        elif 'AELMOnly' in kwargs:
            encoder_out = kwargs['encode_out']
            mlmDecoder_out = self.mlmDecoder(encoder_out=encoder_out, **kwargs)
            return mlmDecoder_out
        else:
            encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
            decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
#             if self.train_mlm:
#                 mlmDecoder_out = self.mlmDecoder(encoder_out=encoder_out, **kwargs)
#                 return mlmDecoder_out
#             else:
            return decoder_out


class TransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = 'relu',
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            export: bool = False,
    ) -> None:
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn


class NgramTransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            ngram=2,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = 'relu',
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            export: bool = False,

    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.ngram_self_attn = NgramMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            ngram=ngram
        )
        self.ngram = ngram

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.encoder_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            kdim=embedding_dim,
            vdim=embedding_dim,
            dropout=attention_dropout,
            encoder_decoder_attention=True,
        )
        #==================junwei===============
        self.num_heads = num_attention_heads
        self.head_dim = self.embedding_dim // self.num_heads
        self.onnx_trace = False
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)
        self.need_attn = True
        self.before_softmax = True
        #==================junwei===============
        self.encoder_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)
        

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_mask=None,
            incremental_state=None,
            prev_self_attn_state=None,
            prev_attn_state=None,
            self_attn_mask=None,
            ngram_mask_matrix=None,
            i_buckets_main_stream=None,
            i_bucket_relative_stream=None,
            real_positions=None
    ):
        # one main stream and ngram predicting streams
        residual = x

        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
#         pdb.set_trace()
        x, attn = self.ngram_self_attn(
            query=x,
            key=x,
            value=x,
            incremental_state=incremental_state,
            need_weights=False,
            self_attn_mask=self_attn_mask,
            ngram_mask_matrix=ngram_mask_matrix,
            i_buckets_main_stream=i_buckets_main_stream,
            i_bucket_relative_stream=i_bucket_relative_stream,
            real_positions=real_positions
        )
    
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        if prev_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.encoder_attn._set_input_buffer(incremental_state, saved_state)
#         x, attn = self.encoder_attn(
#             query=x,
#             key=encoder_out,
#             value=encoder_out,
#             key_padding_mask=encoder_mask,
#             incremental_state=incremental_state,
#             static_kv=True,
# #             need_weights=(not self.training and self.need_attn),
#             need_weights=self.need_attn,
#         )
        tgt_len, bsz, embed_dim = x.size()
        src_len, _ , _ = encoder_out.size()
        attn_weights, v = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_mask,
            incremental_state=incremental_state,
            static_kv=True,
#             need_weights=(not self.training and self.need_attn),
            need_weights=self.need_attn,
            before_softmax=self.before_softmax
        )
#         pdb.set_trace()
        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights_prob = attn_weights_float.type_as(attn_weights)
#         attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights_prob), p=self.attention_dropout, training=self.training)

        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if self.need_attn:
            attn_weights_prob = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            # average attention weights over heads
            attn_weights_prob = attn_weights_prob.mean(dim=0)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            attn_weights = attn_weights.mean(dim=0)
        else:
            attn_weights_prob = None
            attn_weights = None

        x = attn
    
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)

#         pdb.set_trace()
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
#         pdb.set_trace()
        return x, attn_weights, attn_weights_prob

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = True

class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = None #math.sqrt(embed_dim)
        self.embed_positions = LearnedPositionalEmbedding(
            args.max_source_positions + 1 + self.padding_idx, embed_dim, self.padding_idx,
        )

        self.layers = nn.ModuleList([])

        self.layers.extend([
            TransformerEncoderLayer(
                args.encoder_embed_dim,
                args.encoder_ffn_embed_dim,
                args.encoder_attention_heads,
                args.dropout,
                args.attention_dropout,
                args.activation_dropout,
                args.activation_fn,
            )
            for i in range(args.encoder_layers)
        ])

        self.emb_layer_norm = LayerNorm(embed_dim)

        self.apply(init_bert_params)

    def forward(self, src_tokens, src_lengths, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embed_tokens(src_tokens)
        # embed tokens and positions
        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            pos_emb, real_positions = self.embed_positions(src_tokens)
            x += pos_emb

        if self.emb_layer_norm:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if encoder_padding_mask is not None:
            x *= 1 - encoder_padding_mask.unsqueeze(-1).type_as(x)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            # x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask, real_positions=real_positions)
            x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask,)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class NgramTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))
        self.ngram = args.ngram
        self.num_buckets = args.num_buckets
        self.relative_max_distance = args.relative_max_distance

        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions
        self.embed_dim = embed_dim
        self.tgt_embed_tokens = embed_tokens
        self.embed_scale = None #math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.embed_positions = LearnedPositionalEmbedding(
            args.max_target_positions + 2 + self.padding_idx, embed_dim, self.padding_idx,
        )

        self.ngram_input_embed = Embedding(self.ngram, input_embed_dim, None)

        self.layers = nn.ModuleList([])

        self.layers.extend([
            NgramTransformerDecoderLayer(
                args.ngram,
                args.decoder_embed_dim,
                args.decoder_ffn_embed_dim,
                args.decoder_attention_heads,
                args.dropout,
                args.attention_dropout,
                args.activation_dropout,
                args.activation_fn,

            )
            for _ in range(args.decoder_layers)
        ])

#         if not self.share_input_output_embed:
#             self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.embed_dim))
#             nn.init.normal_(self.embed_out, mean=0, std=self.embed_dim ** -0.5)

        self.emb_layer_norm = LayerNorm(embed_dim)
        self.apply(init_bert_params)

    def forward(self,
                prev_output_tokens,
                encoder_out=None,
                incremental_state=None,
                **unused):
        # T
        T = prev_output_tokens.size(1)
        # x [B, (1+ngram)*T, C]
        x_list, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state, **unused)
        
#         {'attn_weights': attn_weights_list, 'attn_weights_prob': attn_weights_prob_list}
        if 'attn_weights' not in extra and 'attn_weights_prob' not in extra:
            x_predicted = x_list[1:]
            x_predicted = [self.output_layer(x, encoder_out=encoder_out) for x in x_predicted]
        elif 'attn_weights' in extra:
            x_predicted_1 = x_list[1:]
            x_predicted_1 = [self.output_layer(x, encoder_out=encoder_out)[:,:,:104] for x in x_predicted_1]
            x_predicted_1 = [pred.type(torch.float16) for pred in x_predicted_1]
            attns = [att.type(torch.float16) for att in extra['attn_weights'][1:]]
            extra['attn_weights'] = attns
            x_predicted = [torch.cat([others, attn], dim=-1) for others, attn in zip(x_predicted_1, attns)]
#             pdb.set_trace()
        elif 'attn_weights_prob' in extra:
            x_predicted_1 = x_list[1:]
            x_predicted_1 = [self.output_layer(x, encoder_out=encoder_out)[:,:,:104] for x in x_predicted_1]
            x_predicted_1 = [pred.type(torch.float16) for pred in x_predicted_1]
            attns = [att.type(torch.float16) for att in extra['attn_weights_prob'][1:]]
            extra['attn_weights_prob'] = attns
            x_predicted = [torch.cat([others, attn], dim=-1) for others, attn in zip(x_predicted_1, attns)]
#             pdb.set_trace()
            
        if incremental_state is not None:
            x_predicted = x_predicted[0]
            for k in extra:
                if extra[k] is not None:
                    extra[k] = extra[k][0]
        return x_predicted, extra

    def _relative_positions_bucket(self, relative_positions, bidirectional=False):
        num_buckets = self.num_buckets
        max_distance = self.relative_max_distance
        n = -relative_positions
        result = 0
        if bidirectional:
            num_buckets = num_buckets // 2
            result = result + torch.lt(n, torch.zeros_like(n)).int() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = torch.lt(n, max_exact)
        val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (
                num_buckets - max_exact)
        val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1))
        val_if_large = val_if_large.int()
        result = result + torch.where(is_small, n.int(), val_if_large)
        return result

    def cal_pretrain_relative_positions(self, real_positions):
#         pdb.set_trace()
        # main stream
        main_stream_relative_positions = real_positions.unsqueeze(1)
        # [B,T,T/S]
        main_stream_relative_positions = main_stream_relative_positions.repeat(1, real_positions.size(-1), 1)
        # [B,T,1]
        real_positions_main = real_positions.unsqueeze(-1)
        main_stream_relative_positions = main_stream_relative_positions - real_positions_main

        # predicting stream
        # input shift
        real_positions_shift_predicting_stream = real_positions - 1
        # [B,1, 2*T]
        predicting_stream_relative_positions = torch.cat((real_positions_shift_predicting_stream, real_positions),
                                                         dim=-1).unsqueeze(1)
        # [B,T, 2*T]
        predicting_stream_relative_positions = predicting_stream_relative_positions.repeat(1, real_positions.size(-1),
                                                                                           1)
        # [B,T, 1]
        real_positions_predicting_stream = real_positions.unsqueeze(-1)
        predicting_stream_relative_positions = predicting_stream_relative_positions - real_positions_predicting_stream
        i_buckets_main_stream = self._relative_positions_bucket(main_stream_relative_positions, bidirectional=False)
        i_bucket_relative_stream = self._relative_positions_bucket(predicting_stream_relative_positions,
                                                                   bidirectional=False)
        return i_buckets_main_stream, i_bucket_relative_stream

    def cal_finetune_relative_positions(self, real_positions):
        n_tokens = real_positions.size(-1)
        batch_size = real_positions.size(0)
        if not hasattr(self,
                       '_finetune_i_bucket_main_stream') or \
                self._finetune_i_bucket_main_stream is None \
                or self._finetune_i_bucket_main_stream.device != real_positions.device:
            fake_positions = torch.arange(1, self.max_target_positions + 1).repeat(1, 1)
            finetune_i_bucket_main_stream, finetune_i_bucket_predicting_stream = \
                self.cal_pretrain_relative_positions(fake_positions)
            self._finetune_i_bucket_main_stream = finetune_i_bucket_main_stream.to(real_positions.device)
            self._finetune_i_bucket_predicting_stream = finetune_i_bucket_predicting_stream.to(real_positions.device)
        finetune_i_bucket_main_stream = self._finetune_i_bucket_main_stream[:, :n_tokens, :n_tokens].repeat(batch_size,
                                                                                                            1, 1)
        finetune_i_bucket_predicting_stream = torch.cat([
            self._finetune_i_bucket_predicting_stream[:, :n_tokens, :n_tokens],
            self._finetune_i_bucket_predicting_stream[:, :n_tokens,
            self.max_target_positions:self.max_target_positions + n_tokens]
        ], 2).repeat(batch_size, 1, 1)
        return finetune_i_bucket_main_stream, finetune_i_bucket_predicting_stream

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        # embed positions
        # [bos, A, B, C, D, eos] with real positions [1,2,3,4,5,6](main stream), [2,3,4,5,6,7](predicting stream)
        # target [B,C,D] with prev [A,B,C] from [A,B,C,D] as pretraining span with real positions [2,3,4],
        # but target actually [3,4,5] for fine tune with another [bos].
        # thus [2,3,4] used for main stream shifted prev [A,B,C], [3,4,5] used for predicting [B,C,D]
#         pdb.set_trace()
        if 'positions' in unused:
            # pretrain procedure
            main_stream_pos_embed = self.embed_positions._forward(unused['positions'])
            real_positions = unused['positions']
            i_buckets_main_stream, i_bucket_relative_stream = \
                self.cal_pretrain_relative_positions(real_positions)
        else:
            # fine tune procedure
            main_stream_pos_embed, real_positions = self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            ) if self.embed_positions is not None else None
            if incremental_state is not None:
                i_buckets_main_stream, i_bucket_relative_stream = None, None
            else:
                i_buckets_main_stream, i_bucket_relative_stream = \
                    self.cal_finetune_relative_positions(real_positions)

        predicting_stream_pos_embed = self.embed_positions._forward(real_positions + 1)

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if main_stream_pos_embed is not None:
                main_stream_pos_embed = main_stream_pos_embed[:, -1:]

        #=================Junwei=========
        
#         x = self.tgt_embed_tokens(prev_output_tokens)
#         x = self.encoder_out['encode_out'](prev_output_tokens - 104)

#         bsz, length = prev_output_tokens.size()
#         part1 =self.tgt_embed_tokens.weight[:104].unsqueeze(0).repeat(bsz, 1, 1)
#         pdb.set_trace()
        p1 =self.tgt_embed_tokens.weight[:104]

        part2 = encoder_out['encoder_out'].transpose(0, 1)
        dec_emb = torch.cat([torch.cat((p1, p2),0).unsqueeze(0) for p2 in part2], 0)

#         dec_emb = torch.cat((part1, part2), 1)
#         pdb.set_trace()
        x = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(dec_emb, prev_output_tokens)], 0)
#         x = dec_emb.index_select(1, prev_output_tokens)
        
        # embed tokens and positions
        if self.embed_scale is not None:
            x *= self.embed_scale

        if main_stream_pos_embed is not None:
            x += main_stream_pos_embed

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]
        if main_stream_pos_embed is None:
            print('positions should be used to predict ngrams')
            raise Exception()

        if self.embed_scale is not None:
            ngram_input_embed = self.embed_scale * self.ngram_input_embed.weight
        else:
            ngram_input_embed = self.ngram_input_embed.weight

        if incremental_state is not None:
            B = x.size(1)
            ngram_masks = [
                (ngram_input_embed[ngram - 1] + predicting_stream_pos_embed).transpose(0, 1).repeat(1, B, 1)
                for ngram in range(self.ngram)]
        else:
            ngram_masks = [(ngram_input_embed[ngram - 1] + predicting_stream_pos_embed).transpose(0, 1) for
                           ngram in range(self.ngram)]



#         self_attn_mask = self.buffered_future_mask(x) if incremental_state is None else None
#         ngram_mask_matrix = self.buffered_future_mask_ngram(x) if incremental_state is None else None
        
        L2R = True
        if 'L2R' in unused:
            L2R = unused['L2R']
        sequence_order = real_positions - 1 #[batch_size, max_length]
        if incremental_state is None:
            bsz, max_length = sequence_order.size()
            lengths = torch.sum(torch.where(sequence_order.eq(self.padding_idx - 1), sequence_order.new_zeros(sequence_order.size()), sequence_order.new_ones(sequence_order.size())), 1).long()
            for b in range(bsz):
                n = lengths[b].item()
                if not L2R:
                    sequence_order[b] = torch.cat([torch.tensor([0]), torch.randperm(n-1)+1, torch.tensor([self.padding_idx - 1]*(max_length-n)).long()], 0)
                else:
                    sequence_order[b] = torch.cat([torch.arange(n), torch.tensor([self.padding_idx - 1]*(max_length-n)).long()], 0)
        
#         pdb.set_trace()
        l2rseq_max = self.get_l2rseq_max(x, self.max_target_positions)
        eye_matrix_max = self.get_eye_matrix_max(x, self.max_target_positions)

        self_attn_mask_plm = self.buffered_future_mask_plm(x, sequence_order, l2rseq_max, self.padding_idx - 1) if incremental_state is None else None     
        plm_ngram_mask_matrix = self.buffered_future_mask_ngram_plm(x, sequence_order,  l2rseq_max, eye_matrix_max, self.padding_idx - 1) if incremental_state is None else None
        
#         pdb.set_trace()

        # TODO in train [(1+ngram)*T, B, C], in inference [T+ngram, B, C]
        x = torch.cat([x] + ngram_masks, 0)

        if self.emb_layer_norm:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # decoder layers
        for layer in self.layers:
            x, attn_weights, attn_weights_prob = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
#                 self_attn_mask=self_attn_mask,
                self_attn_mask=self_attn_mask_plm,
#                 ngram_mask_matrix=ngram_mask_matrix,
                ngram_mask_matrix=plm_ngram_mask_matrix,
                i_buckets_main_stream=i_buckets_main_stream,
                i_bucket_relative_stream=i_bucket_relative_stream,
                real_positions=real_positions
            )
            inner_states.append(x)

        # TODO [(1+ngram)*T, B, C] -> [B, (1+ngram)*T, C]
        x_list = x.transpose(0, 1).chunk(1 + self.ngram, 1)
        if attn_weights is not None:
            attn_weights_list = attn_weights.chunk(1 + self.ngram, 1)
            attn_weights_prob_list = attn_weights_prob.chunk(1 + self.ngram, 1)
        else:
            attn_weights_list = None
            attn_weights_prob_list = None

#         pdb.set_trace()
        return x_list, {'attn_weights': attn_weights_list, 'attn_weights_prob': attn_weights_prob_list}

    def get_l2rseq_max(self, tensor, max_target_positions):
        if not hasattr(self, '_l2rseq_max') or self._l2rseq_max is None or self._l2rseq_max.device != tensor.device:
            self._l2rseq_max = torch.arange(max_target_positions).to(tensor.device)
        return self._l2rseq_max
    
    def get_eye_matrix_max(self, tensor, max_target_positions):
        if not hasattr(self, '_eye_matrix_max') or self._eye_matrix_max is None or self._eye_matrix_max.device != tensor.device:
            self._eye_matrix_max = torch.eye(max_target_positions).to(tensor.device)
        return self._eye_matrix_max
        
    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            if sample is not None:
                assert 'target' in sample
                target = sample['target']
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out
        '''
        logits_list = net_output[0]
        if log_probs:
            return [utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace) for logits in logits_list][0]
        else:
            return [utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace) for logits in logits_list][0]
        '''
        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
#         if self.share_input_output_embed:
#             pdb.set_trace()
#             return F.linear(features, self.tgt_embed_tokens.weight)
#         else:
#             return F.linear(features, torch.cat((self.tgt_embed_tokens[:104], )
#         pdb.set_trace()
        if 'encoder_out' in kwargs:
            p1 =self.tgt_embed_tokens.weight[:104]
            part2 = kwargs['encoder_out']['encoder_out'].transpose(0, 1)
            dec_emb = torch.cat([torch.cat((p1, p2),0).unsqueeze(0) for p2 in part2], 0).transpose(-1,-2)
#         return F.linear(features, encode_out)
        logits = torch.bmm(features, dec_emb)
        return logits


    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]
    
    def buffered_future_mask_plm(self, tensor, order, l2rseq_max, padding_idx):
        dim = tensor.size(0)
#         if not hasattr(self, '_future_mask_plm') or self._future_mask_plm is None or self._future_mask_plm.device != tensor.device or self._future_mask_plm.size(1) < dim:
#         pdb.set_trace()
        self._future_mask_plm, seqs = makeOrder(tensor, order, l2rseq_max, padding_idx)
        self._future_mask_plm = torch.cat([x.unsqueeze(0) for x in self._future_mask_plm], 0)
        return self._future_mask_plm[:,:dim, :dim]

    def buffered_future_mask_ngram(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_ngram_future_mask') or self._ngram_future_mask is None or self._ngram_future_mask.device != tensor.device:
            self._ngram_future_mask = ngram_attention_bias(self.max_target_positions, self.ngram).type(tensor.dtype).to(tensor.device)
        ngram_future_mask = torch.cat([self._ngram_future_mask[:, :dim, :dim],
                                       self._ngram_future_mask[:, :dim,
                                       self.max_target_positions: self.max_target_positions + dim]
                                       ], 2)
        return ngram_future_mask
    
    def buffered_future_mask_ngram_plm(self, tensor, order, l2rseq_max, eye_matrix_max, padding_idx): #order: batch * T
        dim = tensor.size(0)
#         if not hasattr(self, '_ngram_plm_future_mask') or self._ngram_future_mask_plm is None or self._ngram_future_mask_plm.device != tensor.device:
        self._ngram_future_mask_plm, ngram_seqs = makeOrder_ngram(self.ngram, tensor, order, l2rseq_max, eye_matrix_max, padding_idx)
#         ngram_future_mask_plm = torch.cat([self._ngram_future_mask_plm[:, :, :dim, :dim],
#                                        self._ngram_future_mask_plm[:, :, :dim, self.max_target_positions: self.max_target_positions + dim]
#                                        ], -1)
#         pdb.set_trace()
        return self._ngram_future_mask_plm


class MLMDecoder(FairseqDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_source_positions
        self.embed_dim = embed_dim
        self.embed_tokens = embed_tokens
        self.embed_scale = None #math.sqrt(embed_dim)  # todo: try with input_embed_dim

#         self.embed_positions = LearnedPositionalEmbedding(
#             args.max_target_positions + 2 + self.padding_idx, embed_dim, self.padding_idx,
#         )

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.embed_dim ** -0.5)

#         self.emb_layer_norm = LayerNorm(embed_dim)
        self.apply(init_bert_params)

    def forward(self, encoder_out=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(encoder_out=encoder_out, **kwargs)
        x = self.output_layer(x)[kwargs['masked_tokens']]
#         pdb.set_trace()
        return x, extra

    def extract_features(self, encoder_out=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        return encoder_out['encoder_out'].transpose(0, 1), None        

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())
    
    
def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


@register_model_architecture('jdnet_mrc', 'jdnet_mrc')
def base_architecture(args):
    args.ngram = getattr(args, 'ngram', 2)
    args.num_buckets = getattr(args, 'num_buckets', 32)
    args.relative_max_distance = getattr(args, 'relative_max_distance', 128)

    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)

    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.load_sep = getattr(args, 'load_sep', False)
    args.train_mlm = getattr(args, 'train_mlm', False)


@register_model_architecture('jdnet_mrc', 'jdnet_mrc_base')
def transformer_base(args):
    args.ngram = getattr(args, 'ngram', 2)
    args.num_buckets = getattr(args, 'num_buckets', 32)
    args.relative_max_distance = getattr(args, 'relative_max_distance', 128)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)

    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    base_architecture(args)


@register_model_architecture('jdnet_mrc', 'jdnet_mrc_middle')
def transformer_middle(args):
    args.ngram = getattr(args, 'ngram', 2)
    args.num_buckets = getattr(args, 'num_buckets', 32)
    args.relative_max_distance = getattr(args, 'relative_max_distance', 128)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    transformer_base(args)


@register_model_architecture('jdnet_mrc', 'jdnet_mrc_large')
def transformer_big(args):
    args.ngram = getattr(args, 'ngram', 2)
    args.num_buckets = getattr(args, 'num_buckets', 32)
    args.relative_max_distance = getattr(args, 'relative_max_distance', 128)

    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    transformer_middle(args)

@register_model_architecture('jdnet_mrc', 'jdnet_mrc_large_declayer1')
def transformer_big(args):
    args.ngram = getattr(args, 'ngram', 2)
    args.num_buckets = getattr(args, 'num_buckets', 32)
    args.relative_max_distance = getattr(args, 'relative_max_distance', 128)

    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    transformer_middle(args)

@register_model_architecture('jdnet_mrc', 'jdnet_mrc_large_declayer3')
def transformer_big(args):
    args.ngram = getattr(args, 'ngram', 2)
    args.num_buckets = getattr(args, 'num_buckets', 32)
    args.relative_max_distance = getattr(args, 'relative_max_distance', 128)

    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    transformer_middle(args)
    
@register_model_architecture('jdnet_mrc', 'jdnet_mrc_large_declayer6')
def transformer_big(args):
    args.ngram = getattr(args, 'ngram', 2)
    args.num_buckets = getattr(args, 'num_buckets', 32)
    args.relative_max_distance = getattr(args, 'relative_max_distance', 128)

    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    transformer_middle(args)