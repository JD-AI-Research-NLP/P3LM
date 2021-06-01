# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
import pdb

class ClassificationGenerator(object):
    def __init__(
        self,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        sampling=False,
        sampling_topk=-1,
        sampling_topp=-1.0,
        temperature=1.,
        diverse_beam_groups=-1,
        diverse_beam_strength=0.5,
        match_source_len=False,
        no_repeat_ngram_size=0,
    ):
        """Generates translations of a given source sentence.

        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            sampling (bool, optional): sample outputs instead of beam search
                (default: False)
            sampling_topk (int, optional): only sample among the top-k choices
                at each step (default: -1)
            sampling_topp (float, optional): only sample among the smallest set
                of words whose cumulative probability mass exceeds p
                at each step (default: -1.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            diverse_beam_groups/strength (float, optional): parameters for
                Diverse Beam Search sampling
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert sampling_topp < 0 or sampling, '--sampling-topp requires --sampling'
        assert temperature > 0, '--temperature must be greater than 0'

        if sampling:
            self.search = search.Sampling(tgt_dict, sampling_topk, sampling_topp)
        elif diverse_beam_groups > 0:
            self.search = search.DiverseBeamSearch(tgt_dict, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            self.search = search.LengthConstrainedBeamSearch(
                tgt_dict, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        else:
            self.search = search.BeamSearch(tgt_dict)

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        model = EnsembleModel(models)
        return self._generate(model, sample, **kwargs)

    def _get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes
    
    @torch.no_grad()
    def _generate(
        self,
        model,
        sample,
        prefix_tokens=None,
        bos_token=None,
        **kwargs
    ):
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but ClassificationGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        src_tokens = encoder_input['src_tokens']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )

        # compute the encoder output for each beam
        #pdb.set_trace()
        encoder_outs = model.forward_encoder(encoder_input)
        classification_out = model.forward_classifier(encoder_output=encoder_outs[0])
        logits = classification_out[0]['classify_logits']
                
        return logits

class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        if all(isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')
    
    #junwei
    def has_classifier(self):
        return hasattr(self.models[0], 'classifier')
    #junwei
    
    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.encoder(**encoder_input) for model in self.models]

    #junwei
    @torch.no_grad()
    def forward_classifier(self, encoder_output):
        if not self.has_classifier():
            return None
        return [model.classifier(encoder_out=encoder_output) for model in self.models]
    #junwei
    
    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1.):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )

        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(
                tokens,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens, encoder_out=encoder_out, incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)


class ClassificationGeneratorWithAlignment(ClassificationGenerator):

    def __init__(self, tgt_dict, left_pad_target=False, **kwargs):
        """Generates translations of a given source sentence.

        Produces alignments following "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        model = EnsembleModelWithAlignment(models)
        finalized = super()._generate(model, sample, **kwargs)

        src_tokens = sample['net_input']['src_tokens']
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        src_tokens, src_lengths, prev_output_tokens, tgt_tokens = \
            self._prepare_batch_for_alignment(sample, finalized)
        if any(getattr(m, 'full_context_alignment', False) for m in model.models):
            attn = model.forward_align(src_tokens, src_lengths, prev_output_tokens)
        else:
            attn = [
                finalized[i // beam_size][i % beam_size]['attention'].transpose(1, 0)
                for i in range(bsz * beam_size)
            ]

        # Process the attn matrix to extract hard alignments.
        for i in range(bsz * beam_size):
            alignment = utils.extract_hard_alignment(attn[i], src_tokens[i], tgt_tokens[i], self.pad, self.eos)
            finalized[i // beam_size][i % beam_size]['alignment'] = alignment
        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample['net_input']['src_tokens']
        bsz = src_tokens.shape[0]
        src_tokens = src_tokens[:, None, :].expand(-1, self.beam_size, -1).contiguous().view(bsz * self.beam_size, -1)
        src_lengths = sample['net_input']['src_lengths']
        src_lengths = src_lengths[:, None].expand(-1, self.beam_size).contiguous().view(bsz * self.beam_size)
        prev_output_tokens = data_utils.collate_tokens(
            [beam['tokens'] for example in hypothesis for beam in example],
            self.pad, self.eos, self.left_pad_target, move_eos_to_beginning=True,
        )
        tgt_tokens = data_utils.collate_tokens(
            [beam['tokens'] for example in hypothesis for beam in example],
            self.pad, self.eos, self.left_pad_target, move_eos_to_beginning=False,
        )
        return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


class EnsembleModelWithAlignment(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    def forward_align(self, src_tokens, src_lengths, prev_output_tokens):
        avg_attn = None
        for model in self.models:
            decoder_out = model(src_tokens, src_lengths, prev_output_tokens)
            attn = decoder_out[1]['attn']
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_attn.div_(len(self.models))
        return avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens,
                encoder_out=encoder_out,
                incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn
