# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion 
from torch.nn import CrossEntropyLoss

import pdb
    
@register_criterion('ngram_language_loss_L2R')
class NgramLmLossL2R(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        logits_list = model(**sample['net_input'], return_all_hiddens=False)[0]
        targets = model.get_targets(sample, [logits_list[0]])

#         pdb.set_trace()
        ngram = len(logits_list)
        # [B, ngram, T]
        expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
        for i in range(ngram):
            if i > 0 and self.disable_ngram_loss:
                break

            padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
            if 'target_idx' in sample:
                expend_targets[i,:,:] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
            else:
                expend_targets[i,:,:] = targets
        targets = expend_targets

        logits = torch.cat(logits_list, dim=0) #.view(ngram, *logits_list[0].size())

        lprobs = F.log_softmax(
                    logits.view(-1, logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                )

        loss = F.nll_loss(
               lprobs,
               targets.view(-1),
               reduction='sum',
               ignore_index=self.padding_idx,
               )

        if self.eps > 0.:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.eps / lprobs.size(-1)
            loss = (1. - self.eps) * loss + eps_i * smooth_loss

        sample_size = targets.ne(self.padding_idx).int().sum().item()
#         pdb.set_trace()

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output

@register_criterion('ngram_language_loss_PLM')
class NgramLmLossPLM(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        logits_list = model(**sample['net_input'], return_all_hiddens=False, L2R=False)[0]
        targets = model.get_targets(sample, [logits_list[0]])
#         pdb.set_trace()

        ngram = len(logits_list)
        # [B, ngram, T]
        expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
        for i in range(ngram):
            if i > 0 and self.disable_ngram_loss:
                break

            padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
            if 'target_idx' in sample:
                expend_targets[i,:,:] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
            else:
                expend_targets[i,:,:] = targets
        targets = expend_targets

        logits = torch.cat(logits_list, dim=0) #.view(ngram, *logits_list[0].size())

        lprobs = F.log_softmax(
                    logits.view(-1, logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                )

        loss = F.nll_loss(
               lprobs,
               targets.view(-1),
               reduction='sum',
               ignore_index=self.padding_idx,
               )

        if self.eps > 0.:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.eps / lprobs.size(-1)
            loss = (1. - self.eps) * loss + eps_i * smooth_loss

        sample_size = targets.ne(self.padding_idx).int().sum().item()

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output

@register_criterion('ngram_language_loss_MLM')
class NgramLmLossMLM(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        masked_tokens = sample['target_mlm'].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()
        
        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None
        
#         pdb.set_trace()
#         logits = model(**sample['net_input'], masked_tokens=masked_tokens)[0]
        encode_out = model(**sample['net_input'], EncodeOnly=True)
        logits = model(**sample['net_input'], encode_out=encode_out, AELMOnly=True, masked_tokens=masked_tokens)[0]

        targets = sample['target_mlm']

#         pdb.set_trace()
        if sample_size != 0:
            targets = targets[masked_tokens]

        lprobs = F.log_softmax(
            logits.view(-1, logits.size(-1)),
            dim=-1,
            dtype=torch.float32,
        )
        loss = F.nll_loss(
            lprobs,
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )
#         pdb.set_trace()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
    
@register_criterion('ngram_language_loss_L2RPLM')
class NgramLmLossL2RPLM(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
#         pdb.set_trace()

#         logits_list_L2R = model(**sample['net_input'], return_all_hiddens=False, L2R=True)[0]
#         logits_list_PLM = model(**sample['net_input'], return_all_hiddens=False, L2R=False)[0]
        encode_out = model(**sample['net_input'], EncodeOnly=True)
        logits_list_L2R = model(**sample['net_input'], encode_out=encode_out, ARLMOnly=True, L2R=True)[0]
        logits_list_PLM = model(**sample['net_input'], encode_out=encode_out, ARLMOnly=True, L2R=False)[0]

        targets = model.get_targets(sample, [logits_list_L2R[0]])

        ngram = len(logits_list_L2R)
        # [B, ngram, T]
        expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
        for i in range(ngram):
            if i > 0 and self.disable_ngram_loss:
                break

            padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
            if 'target_idx' in sample:
                expend_targets[i,:,:] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
            else:
                expend_targets[i,:,:] = targets
        targets = expend_targets

        logits_L2R = torch.cat(logits_list_L2R, dim=0) #.view(ngram, *logits_list[0].size())
        logits_PLM = torch.cat(logits_list_PLM, dim=0) #.view(ngram, *logits_list[0].size())

        lprobs_L2R = F.log_softmax(
                    logits_L2R.view(-1, logits_L2R.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                )
        lprobs_PLM = F.log_softmax(
                    logits_PLM.view(-1, logits_PLM.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                )

        loss_L2R = F.nll_loss(
               lprobs_L2R,
               targets.view(-1),
               reduction='sum',
               ignore_index=self.padding_idx,
               )

        loss_PLM = F.nll_loss(
               lprobs_PLM,
               targets.view(-1),
               reduction='sum',
               ignore_index=self.padding_idx,
               )

        if self.eps > 0.:
            smooth_loss = -lprobs_L2R.sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.eps / lprobs_L2R.size(-1)
            loss_L2R = (1. - self.eps) * loss_L2R + eps_i * smooth_loss

        if self.eps > 0.:
            smooth_loss = -lprobs_PLM.sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.eps / lprobs_PLM.size(-1)
            loss_PLM = (1. - self.eps) * loss_PLM + eps_i * smooth_loss

        loss = loss_L2R + loss_PLM
        sample_size_L2R = targets.ne(self.padding_idx).int().sum().item()
        sample_size_PLM = sample_size_L2R
        sample_size = targets.ne(self.padding_idx).int().sum().item() * 2 #L2R + PLM
#         pdb.set_trace()

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'loss_L2R': utils.item(loss_L2R.data) if reduce else loss_L2R.data,
            'loss_PLM': utils.item(loss_PLM.data) if reduce else loss_PLM.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'sample_size_L2R': sample_size_L2R,
            'sample_size_PLM': sample_size_PLM,
        }

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        loss_L2R = sum(log.get('loss_L2R', 0) for log in logging_outputs)
        loss_PLM = sum(log.get('loss_PLM', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        sample_size_L2R = sum(log.get('sample_size_L2R', 0) for log in logging_outputs)
        sample_size_PLM = sum(log.get('sample_size_PLM', 0) for log in logging_outputs)
        
        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ppl': math.pow(2, (loss / sample_size / math.log(2))),
            'loss_L2R': loss_L2R / sample_size_L2R / math.log(2),
            'loss_PLM': loss_PLM / sample_size_PLM / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output


#     def forward(self, model, sample, reduce=True):
#         """Compute the loss for the given sample.
#         Returns a tuple with three elements:
#         1) the loss
#         2) the sample size, which is used as the denominator for the gradient
#         3) logging outputs to display while training
#         """
#         # compute MLM loss
# #         pdb.set_trace()
        
# #         logits_list_L2R = model(**sample['net_input'], return_all_hiddens=False, L2R=True)[0]
# #         logits_list_PLM = model(**sample['net_input'], return_all_hiddens=False, L2R=False)[0]
#         encode_out = model(**sample['net_input'], EncodeOnly=True)
#         logits_list_L2R = model(**sample['net_input'], encode_out=encode_out, ARLMOnly=True, L2R=True)[0]
#         logits_list_PLM = model(**sample['net_input'], encode_out=encode_out, ARLMOnly=True, L2R=False)[0]
        
#         targets = model.get_targets(sample, [logits_list_L2R[0]])

#         ngram = len(logits_list_L2R)
#         # [B, ngram, T]
#         expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
#         for i in range(ngram):
#             if i > 0 and self.disable_ngram_loss:
#                 break

#             padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
#             if 'target_idx' in sample:
#                 expend_targets[i,:,:] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
#             else:
#                 expend_targets[i,:,:] = targets
#         targets = expend_targets

#         logits_L2R = torch.cat(logits_list_L2R, dim=0) #.view(ngram, *logits_list[0].size())
#         logits_PLM = torch.cat(logits_list_PLM, dim=0) #.view(ngram, *logits_list[0].size())

#         lprobs_L2R = F.log_softmax(
#                     logits_L2R.view(-1, logits_L2R.size(-1)),
#                     dim=-1,
#                     dtype=torch.float32,
#                 )
#         lprobs_PLM = F.log_softmax(
#                     logits_PLM.view(-1, logits_PLM.size(-1)),
#                     dim=-1,
#                     dtype=torch.float32,
#                 )

#         loss_L2R = F.nll_loss(
#                lprobs_L2R,
#                targets.view(-1),
#                reduction='sum',
#                ignore_index=self.padding_idx,
#                )
        
#         loss_PLM = F.nll_loss(
#                lprobs_PLM,
#                targets.view(-1),
#                reduction='sum',
#                ignore_index=self.padding_idx,
#                )

#         if self.eps > 0.:
#             smooth_loss = -lprobs_L2R.sum(dim=-1, keepdim=True)
#             non_pad_mask = targets.ne(self.padding_idx).view(-1)
#             smooth_loss = smooth_loss[non_pad_mask]
#             smooth_loss = smooth_loss.sum()

#             eps_i = self.eps / lprobs_L2R.size(-1)
#             loss_L2R = (1. - self.eps) * loss_L2R + eps_i * smooth_loss
        
#         if self.eps > 0.:
#             smooth_loss = -lprobs_PLM.sum(dim=-1, keepdim=True)
#             non_pad_mask = targets.ne(self.padding_idx).view(-1)
#             smooth_loss = smooth_loss[non_pad_mask]
#             smooth_loss = smooth_loss.sum()

#             eps_i = self.eps / lprobs_PLM.size(-1)
#             loss_PLM = (1. - self.eps) * loss_PLM + eps_i * smooth_loss

#         loss = loss_L2R + loss_PLM
#         sample_size = targets.ne(self.padding_idx).int().sum().item() * 2 #L2R + PLM
# #         pdb.set_trace()

#         logging_output = {
#             'loss': utils.item(loss.data) if reduce else loss.data,
#             'ntokens': sample['ntokens'],
#             'nsentences': sample['nsentences'],
#             'sample_size': sample_size,
#         }
#         return loss, sample_size, logging_output

#     @staticmethod
#     def aggregate_logging_outputs(logging_outputs):
#         """Aggregate logging outputs from data parallel training."""
#         loss = sum(log.get('loss', 0) for log in logging_outputs)
#         ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
#         nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
#         sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

#         agg_output = {
#             'loss': loss / sample_size / math.log(2),
#             'ntokens': ntokens,
#             'nsentences': nsentences,
#             'sample_size': sample_size,
#         }
#         return agg_output

@register_criterion('ngram_language_loss_L2RPLMMLM')
class NgramLmLossL2RPLMMLM(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
#         pdb.set_trace()
        masked_tokens_mlm = sample['target_mlm'].ne(self.padding_idx)
        
        sample_size_MLM = masked_tokens_mlm.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size_MLM == 0:
            masked_tokens_mlm = None
        
        encode_out = model(**sample['net_input'], EncodeOnly=True)
        logits_list_L2R = model(**sample['net_input'], encode_out=encode_out, ARLMOnly=True, L2R=True)[0]
        logits_list_PLM = model(**sample['net_input'], encode_out=encode_out, ARLMOnly=True, L2R=False)[0]
        logits_list_MLM = model(**sample['net_input'], encode_out=encode_out, AELMOnly=True, masked_tokens=masked_tokens_mlm)[0]

        targets = model.get_targets(sample, [logits_list_L2R[0]])
        targets_mlm = sample['target_mlm']
        if sample_size_MLM != 0:
            targets_mlm = targets_mlm[masked_tokens_mlm]

        ngram = len(logits_list_L2R)
        # [B, ngram, T]
        expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
        for i in range(ngram):
            if i > 0 and self.disable_ngram_loss:
                break

            padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
            if 'target_idx' in sample:
                expend_targets[i,:,:] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
            else:
                expend_targets[i,:,:] = targets
        targets = expend_targets

        logits_L2R = torch.cat(logits_list_L2R, dim=0) #.view(ngram, *logits_list[0].size())
        logits_PLM = torch.cat(logits_list_PLM, dim=0) #.view(ngram, *logits_list[0].size())

        lprobs_L2R = F.log_softmax(
                    logits_L2R.view(-1, logits_L2R.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                )
        lprobs_PLM = F.log_softmax(
                    logits_PLM.view(-1, logits_PLM.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                )
        lprobs_MLM = F.log_softmax(
                    logits_list_MLM.view(-1, logits_list_MLM.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                )
        
        loss_L2R = F.nll_loss(
               lprobs_L2R,
               targets.view(-1),
               reduction='sum',
               ignore_index=self.padding_idx,
               )
        
        loss_PLM = F.nll_loss(
               lprobs_PLM,
               targets.view(-1),
               reduction='sum',
               ignore_index=self.padding_idx,
               )
        loss_MLM = F.nll_loss(
               lprobs_MLM,
               targets_mlm.view(-1),
               reduction='sum',
               ignore_index=self.padding_idx,
               )

        if self.eps > 0.:
            smooth_loss = -lprobs_L2R.sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.eps / lprobs_L2R.size(-1)
            loss_L2R = (1. - self.eps) * loss_L2R + eps_i * smooth_loss
        
        if self.eps > 0.:
            smooth_loss = -lprobs_PLM.sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.eps / lprobs_PLM.size(-1)
            loss_PLM = (1. - self.eps) * loss_PLM + eps_i * smooth_loss

        loss = loss_L2R + loss_PLM + loss_MLM
        sample_size_L2R = targets.ne(self.padding_idx).int().sum().item()
        sample_size_PLM = sample_size_L2R
        sample_size = sample_size_L2R + sample_size_PLM +  sample_size_MLM #L2R + PLM + MLM

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'loss_L2R': utils.item(loss_L2R.data) if reduce else loss_L2R.data,
            'loss_PLM': utils.item(loss_PLM.data) if reduce else loss_PLM.data,
            'loss_MLM': utils.item(loss_MLM.data) if reduce else loss_MLM.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'sample_size_L2R': sample_size_L2R,
            'sample_size_PLM': sample_size_PLM,
            'sample_size_MLM': sample_size_MLM,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        loss_L2R = sum(log.get('loss_L2R', 0) for log in logging_outputs)
        loss_PLM = sum(log.get('loss_PLM', 0) for log in logging_outputs)
        loss_MLM = sum(log.get('loss_MLM', 0) for log in logging_outputs)

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        sample_size_L2R = sum(log.get('sample_size_L2R', 0) for log in logging_outputs)
        sample_size_PLM = sum(log.get('sample_size_PLM', 0) for log in logging_outputs)
        sample_size_MLM = sum(log.get('sample_size_MLM', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'loss_L2R': loss_L2R / sample_size_L2R / math.log(2),
            'loss_PLM': loss_PLM / sample_size_PLM / math.log(2),
            'loss_MLM': loss_MLM / sample_size_MLM / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'sample_size_MLM': sample_size_MLM,            
        }
        return agg_output
    
@register_criterion('ngram_language_loss_L2RPLM_v2')
class NgramLmLossL2RPLM_v2(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        import random
        rnd = random.randint(0,9)
        L2R = True
        if rnd < 5:
            L2R = True
        else:
            L2R = False
            
        logits_list = model(**sample['net_input'], return_all_hiddens=False, L2R=L2R)[0]
        targets = model.get_targets(sample, [logits_list[0]])
#         pdb.set_trace()

        ngram = len(logits_list)
        # [B, ngram, T]
        expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
        for i in range(ngram):
            if i > 0 and self.disable_ngram_loss:
                break

            padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
            if 'target_idx' in sample:
                expend_targets[i,:,:] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
            else:
                expend_targets[i,:,:] = targets
        targets = expend_targets

        logits = torch.cat(logits_list, dim=0) #.view(ngram, *logits_list[0].size())

        lprobs = F.log_softmax(
                    logits.view(-1, logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                )

        loss = F.nll_loss(
               lprobs,
               targets.view(-1),
               reduction='sum',
               ignore_index=self.padding_idx,
               )

        if self.eps > 0.:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.eps / lprobs.size(-1)
            loss = (1. - self.eps) * loss + eps_i * smooth_loss

        sample_size = targets.ne(self.padding_idx).int().sum().item()

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output

@register_criterion('ngram_language_loss_L2RPLM_adaptive')
class NgramLmLossL2RPLM_adaptive(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss
        self.plm_decay = args.plm_decay #Junwei
#         self.counter = args.start_update  #Junwei
        self.update_freq = args.update_freq
        self.counter = 0


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--plm-decay', default=0., type=float, metavar='D',
                            help='epsilon for plm decay, 0 means no plm decay')
#         parser.add_argument('--start-update', default=0., type=float, metavar='D',
#                             help='epsilon for plm decay, 0 means no plm decay')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        import random
        A=0
        B=1#小数的范围A ~ B
        a=random.uniform(A,B)
        C=4#随机数的精度round(数值，精度)
        rnd = round(a,C)
        
#         pdb.set_trace()
#         rnd = random.randint(0,9)
        number_update = int(self.counter/self.update_freq[0]) # One Update: num of minibatches (self.counter*self.update_freq)
        threshold = 1 - self.plm_decay * number_update
        
#         print('\nCount:' + str(self.counter) + '\n')
#         print('Number_update:' + str(number_update)+ '\n')
        if self.counter % self.update_freq[0] == 0:
            print('plm_decay=' + str(threshold))

        self.counter += 1
#         pdb.set_trace()
        L2R = True
        if rnd < threshold:
            L2R = False
        else:
            L2R = True
            
        logits_list = model(**sample['net_input'], return_all_hiddens=False, L2R=L2R)[0]
        targets = model.get_targets(sample, [logits_list[0]])
#         pdb.set_trace()

        ngram = len(logits_list)
        # [B, ngram, T]
        expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
        for i in range(ngram):
            if i > 0 and self.disable_ngram_loss:
                break

            padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
            if 'target_idx' in sample:
                expend_targets[i,:,:] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
            else:
                expend_targets[i,:,:] = targets
        targets = expend_targets

        logits = torch.cat(logits_list, dim=0) #.view(ngram, *logits_list[0].size())

        lprobs = F.log_softmax(
                    logits.view(-1, logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                )

        loss = F.nll_loss(
               lprobs,
               targets.view(-1),
               reduction='sum',
               ignore_index=self.padding_idx,
               )

        if self.eps > 0.:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.eps / lprobs.size(-1)
            loss = (1. - self.eps) * loss + eps_i * smooth_loss

        sample_size = targets.ne(self.padding_idx).int().sum().item()

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
    
@register_criterion('ngram_language_loss_L2RPLM_adaptive_v2')
class NgramLmLossL2RPLM_adaptive_v2(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss
        self.max_plm_update = args.max_plm_update #Junwei
        self.sigmoid_u = args.sigmoid_u #Junwei
        self.update_freq = args.update_freq
        
#         self.counter = args.save_interval_updates
#         pdb.set_trace()
        self.counter = 0


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--max-plm-update', default=0., type=float, metavar='D',
                            help='epsilon for plm decay, 0 means no plm decay')
        parser.add_argument('--sigmoid-u', default=0., type=float, metavar='D',
                            help='epsilon for plm decay, 0 means no plm decay')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
#         pdb.set_trace()
        import random
        A=0
        B=1#小数的范围A ~ B
        a=random.uniform(A,B)
        C=4#随机数的精度round(数值，精度)
        rnd = round(a,C)
        

#         rnd = random.randint(0,9)
        number_update = int(self.counter/self.update_freq[0]) # One Update: num of minibatches (self.counter*self.update_freq)
#         threshold = number_update / self.max_plm_update # Linear decay
#         threshold = 1.0 / (1.0 + math.pow(math.e, -(number_update - self.max_plm_update/2))) # Sigmoid Decay
        threshold = 1.0 / (1.0 + math.pow(self.sigmoid_u, -(number_update - self.max_plm_update/2))) # Sigmoid Decay
#         threshold = 1.0 / (1.0 + math.pow(1.01, -(number_update - self.max_plm_update/2))) # Sigmoid Decay
#         threshold = 1.0 / (1.0 + math.pow(1.001, -(number_update - self.max_plm_update/2))) # Sigmoid Decay




#         print('\nCount:' + str(self.counter) + '\n')
#         print('Number_update:' + str(number_update)+ '\n')
        if self.counter % self.update_freq[0] == 0:
            print('l2r_percentage=' + str(threshold))

        self.counter += 1
#         pdb.set_trace()
        L2R = True
        if rnd < threshold:
            L2R = True
        else:
            L2R = False
            
        logits_list = model(**sample['net_input'], return_all_hiddens=False, L2R=L2R)[0]
        targets = model.get_targets(sample, [logits_list[0]])
#         pdb.set_trace()

        ngram = len(logits_list)
        # [B, ngram, T]
        expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
        for i in range(ngram):
            if i > 0 and self.disable_ngram_loss:
                break

            padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
            if 'target_idx' in sample:
                expend_targets[i,:,:] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
            else:
                expend_targets[i,:,:] = targets
        targets = expend_targets

        logits = torch.cat(logits_list, dim=0) #.view(ngram, *logits_list[0].size())

        lprobs = F.log_softmax(
                    logits.view(-1, logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                )

        loss = F.nll_loss(
               lprobs,
               targets.view(-1),
               reduction='sum',
               ignore_index=self.padding_idx,
               )

        if self.eps > 0.:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.eps / lprobs.size(-1)
            loss = (1. - self.eps) * loss + eps_i * smooth_loss

        sample_size = targets.ne(self.padding_idx).int().sum().item()

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output


@register_criterion('ngram_language_loss_v2')
class NgramLmLossV2(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss
        self.damping_alpha = args.damping_alpha
        self.ngram = args.ngram
        if not self.disable_ngram_loss:
            self.loss_weights = [pow(self.damping_alpha, i) for i in range(self.ngram)]
            w_sum = sum(self.loss_weights)
            self.loss_weights = [w / w_sum for w in self.loss_weights]
        else:
            self.loss_weights = [1]

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        parser.add_argument('--damping_alpha', default=1.0, type=float, metavar='D',
                            help='the loss future i-th token will multiply a (damping_alpha)^(i-1) to be added. '
                                 'eg, 1.0 * loss1 + 0.5 * loss2 + 0.25 * loss3 ')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        logits_list = model(**sample['net_input'], return_all_hiddens=False)[0]
        #targets = model.get_targets(sample, [logits_list[0]])

        loss_sum = 0
        sample_size_sum = 0
        for i, logits in enumerate(logits_list):
            if i > 0 and self.disable_ngram_loss:
                break
            targets = model.get_targets(sample, [logits_list[0]])
            padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
            if 'target_idx' in sample:
                targets= torch.where(sample['target_idx'] >= i, targets, padding_targets)
            lprobs = F.log_softmax(
                logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            )

            loss = F.nll_loss(
                lprobs,
                targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )
            if self.eps > 0.:
                smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
                non_pad_mask = targets.ne(self.padding_idx).view(-1)
                smooth_loss = smooth_loss[non_pad_mask]
                smooth_loss = smooth_loss.sum()

                eps_i = self.eps / lprobs.size(-1)
                loss = (1. - self.eps) * loss + eps_i * smooth_loss
            loss_sum = loss_sum + loss * self.loss_weights[i]
            sample_size = targets.ne(self.padding_idx).int().sum().item()
            sample_size_sum = sample_size_sum + sample_size
        loss = loss_sum
        if not self.disable_ngram_loss:
            sample_size = sample_size_sum / self.ngram

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output



@register_criterion('ngram_language_loss_v3')
class NgramLmLossV3(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss
        self.damping_alpha = args.damping_alpha
        self.ngram = args.ngram
        if not self.disable_ngram_loss:
            self.loss_weights = []
            rest_weight = 1
            for i in range(self.ngram):
                if i != self.ngram - 1:
                    self.loss_weights.append(rest_weight * self.damping_alpha)
                    rest_weight = rest_weight - self.loss_weights[-1]
                else:
                    self.loss_weights.append(rest_weight)
        else:
            self.loss_weights = [1]


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        parser.add_argument('--damping_alpha', default=1.0, type=float, metavar='D',
                            help='the loss future i-th token will multiply a (damping_alpha)^(i-1) to be added. '
                                 'eg, 1.0 * loss1 + 0.5 * loss2 + 0.25 * loss3 ')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        logits_list = model(**sample['net_input'], return_all_hiddens=False)[0]
        #targets = model.get_targets(sample, [logits_list[0]])

        loss_sum = 0
        sample_size_sum = 0
        for i, logits in enumerate(logits_list):
            if i > 0 and self.disable_ngram_loss:
                break
            targets = model.get_targets(sample, [logits_list[0]])
            padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
            if 'target_idx' in sample:
                targets= torch.where(sample['target_idx'] >= i, targets, padding_targets)
            lprobs = F.log_softmax(
                logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            )

            loss = F.nll_loss(
                lprobs,
                targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )
            if self.eps > 0.:
                smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
                non_pad_mask = targets.ne(self.padding_idx).view(-1)
                smooth_loss = smooth_loss[non_pad_mask]
                smooth_loss = smooth_loss.sum()

                eps_i = self.eps / lprobs.size(-1)
                loss = (1. - self.eps) * loss + eps_i * smooth_loss
            loss_sum = loss_sum + loss * self.loss_weights[i]
            sample_size = targets.ne(self.padding_idx).int().sum().item()
            sample_size_sum = sample_size_sum + sample_size
        loss = loss_sum
        if not self.disable_ngram_loss:
            sample_size = sample_size_sum / self.ngram

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output

@register_criterion('ngram_language_loss_L2RPLM_MRC')
class NgramLmLossL2RPLM_MRC(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
#         pdb.set_trace()
        
#         logits_list_L2R = model(**sample['net_input'], return_all_hiddens=False, L2R=True)[0]
#         logits_list_PLM = model(**sample['net_input'], return_all_hiddens=False, L2R=False)[0]
        encode_out = model(**sample['net_input'], EncodeOnly=True)
        logits_list_L2R = model(**sample['net_input'], encode_out=encode_out, ARLMOnly=True, L2R=True)[0]
        logits_list_PLM = model(**sample['net_input'], encode_out=encode_out, ARLMOnly=True, L2R=False)[0]
        
        bsz, src_length = sample['net_input']['src_tokens'].size()
        
        targets = model.get_targets(sample, [logits_list_L2R[0]])
        _, tgt_len = targets.size()
        
        ngram = len(logits_list_L2R)
        src_mask = torch.cat([torch.ones(bsz, 104).bool().cuda(), sample['net_input']['src_tokens'].ne(self.padding_idx)], dim=-1).unsqueeze(1).repeat(1, tgt_len, 1).unsqueeze(0).repeat(ngram, 1, 1, 1)
        src_mask = src_mask.view(-1, src_mask.size(-1))
        # [B, ngram, T]
        expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
        for i in range(ngram):
            if i > 0 and self.disable_ngram_loss:
                break

            padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
            if 'target_idx' in sample:
                expend_targets[i,:,:] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
            else:
                expend_targets[i,:,:] = targets
        targets = expend_targets

        logits_L2R = torch.cat(logits_list_L2R, dim=0) #.view(ngram, *logits_list[0].size())
        logits_PLM = torch.cat(logits_list_PLM, dim=0) #.view(ngram, *logits_list[0].size())

        lprobs_L2R = F.log_softmax(
                    logits_L2R.view(-1, logits_L2R.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                )
        lprobs_PLM = F.log_softmax(
                    logits_PLM.view(-1, logits_PLM.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                )

        loss_L2R = F.nll_loss(
               lprobs_L2R,
               targets.view(-1),
               reduction='sum',
               ignore_index=self.padding_idx,
               )
        
        loss_PLM = F.nll_loss(
               lprobs_PLM,
               targets.view(-1),
               reduction='sum',
               ignore_index=self.padding_idx,
               )

        if self.eps > 0.:
#             pdb.set_trace()
            zerosnum = lprobs_L2R.new_zeros(lprobs_L2R.size())
            smooth_loss = torch.where(src_mask, -lprobs_L2R, zerosnum).sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.eps / lprobs_L2R.size(-1)
            loss_L2R = (1. - self.eps) * loss_L2R + eps_i * smooth_loss
        
        if self.eps > 0.:
            zerosnum = lprobs_PLM.new_zeros(lprobs_PLM.size())
            smooth_loss = torch.where(src_mask, -lprobs_PLM, zerosnum).sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.eps / lprobs_PLM.size(-1)
            loss_PLM = (1. - self.eps) * loss_PLM + eps_i * smooth_loss

        loss = loss_L2R + loss_PLM
        sample_size = targets.ne(self.padding_idx).int().sum().item() * 2 #L2R + PLM
#         pdb.set_trace()

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
    
@register_criterion('NgramLmLoss_Classification')
class NgramLmLoss_Classification(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss
        self.loss_fct = CrossEntropyLoss()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
#         pdb.set_trace()

        encode_out = model(**sample['net_input'], EncodeOnly=True)
        classification_out = model(**sample['net_input'], encode_out=encode_out, EncClassificationOnly=True)

        logits = classification_out['classify_logits']
        
        target_position = sample['target']
        position_prediction = target_position[:, 0] - 104
#         position_prediction = target_position[:, 0]

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, position_prediction)
        
        sample_size = 1
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output

@register_criterion('NgramLmLoss_Classification_EncDec')
class NgramLmLoss_Classification_EncDec(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss
        self.loss_fct = CrossEntropyLoss()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
#         pdb.set_trace()

#         pdb.set_trace()
        encode_out = model(**sample['net_input'], EncodeOnly=True)
        decode_out, extra = model(**sample['net_input'], encode_out=encode_out, DecodeOnly=True)
#         decode_out, extra = model(**sample['net_input'], encode_out=encode_out, DecodeOnly=True, ALL=True)

#         pdb.set_trace()
        classification_out = model(**sample['net_input'], decode_out=decode_out, DecClassificationOnly=True)

        logits = classification_out['classify_logits']
        
        target_position = sample['target']
#         pdb.set_trace()
        position_prediction = target_position[:, 0] - 104
#         position_prediction = target_position[:, 0]


        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, position_prediction)
        
        sample_size = 1
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
    
@register_criterion('ngram_language_loss_enc_MRC')
class ngram_language_loss_enc_MRC(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss
        self.loss_fct = CrossEntropyLoss()
#         self.dense = nn.Linear(1024, 2)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
#         pdb.set_trace()
       
        
        encode_out = model(**sample['net_input'], EncodeOnly=True)
        classification_out = model(**sample['net_input'], encode_out=encode_out, EncClassificationOnly=True)

#         answerable_logit = encode_out['answerable_logit']
        start_logits, end_logits = classification_out['start_end_logits']
        start_logits = start_logits.squeeze(2)
        end_logits = end_logits.squeeze(2)
                
        start_end_position = sample['target']
#         pdb.set_trace()
        start_position_prediction = start_end_position[:, 0] - 104
        end_position_prediction = start_end_position[:, 1] - 104
#         ignored_index = start_logits.size(1)
#         start_position_prediction.clamp_(0, ignored_index)
#         end_position_prediction.clamp_(0, ignored_index)
#         loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        loss_fct = CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_position_prediction)
        end_loss = loss_fct(end_logits, end_position_prediction)
        
#         answerable_target = start_end_position[:, 2] - 616
#         answerable_loss = loss_fct(answerable_logit, answerable_target)
#         pdb.set_trace()
        loss = (start_loss + end_loss)/2 
        
        sample_size = 1
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
    
@register_criterion('ngram_language_loss_enc_MRC_withNoAnswerLoss')
class ngram_language_loss_enc_MRC_withNoAnswerLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss
        self.loss_fct = CrossEntropyLoss()
#         self.dense = nn.Linear(1024, 2)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
#         pdb.set_trace()
       
        
        encode_out = model(**sample['net_input'], EncodeOnly=True)
        answerable_logit = encode_out['answerable_logit']
        start_logits, end_logits = encode_out['start_end_logits']
        start_logits = start_logits.squeeze(2)
        end_logits = end_logits.squeeze(2)
                
        start_end_position = sample['target']
        start_position_prediction = start_end_position[:, 0] - 104
        end_position_prediction = start_end_position[:, 1] - 104
        ignored_index = start_logits.size(1)
        start_position_prediction.clamp_(0, ignored_index)
        end_position_prediction.clamp_(0, ignored_index)
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_position_prediction)
        end_loss = loss_fct(end_logits, end_position_prediction)
        
        answerable_target = start_end_position[:, 2] - 616
        answerable_loss = loss_fct(answerable_logit, answerable_target)
        
        loss = (start_loss + end_loss + answerable_loss)/3 
        
        sample_size = 1
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output