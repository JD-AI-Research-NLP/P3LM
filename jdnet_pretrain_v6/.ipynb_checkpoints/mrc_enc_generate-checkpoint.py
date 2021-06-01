#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
import pdb
import collections

import json

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary
#     pdb.set_trace()
    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    
    with progress_bar.build_progress_bar(args, itr) as t:
#         w1 = open('./squadmrcv4/res','w')
#         w2 = open('./squadmrcv4/gt','w')
#         w3 = open('./squadmrcv4/score','w')
#         w4 = open('./squadmrcv4/ans_score','w')
#         w5 = open('./squadmrcv4/has_ans_or_not','w')
#         r1 = open('./squadmrcv4/map_id','r').read().splitlines()
        wps_meter = TimeMeter()
        prelim_predictions = []
        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
          "PrelimPrediction",
          ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])
        tot = -1
#         r2 = open('./squadmrcv4/ex_id_fea','r').read().splitlines()
        r2 = open(args.example_id_dir).read().splitlines()
#         r3 = open('./squadmrcv4/truth_id','r').read().splitlines()
        preds = []
        example_index_to_features = {}
        feat_len = {}

        for sample in t:  
            src_len = sample['net_input']['src_lengths']
            ids = sample['id']
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()

            hypos = task.inference_step(generator, models, sample, prefix_tokens)
#             pdb.set_trace()
#             for i in range(hypos[2].size()[0]):
            for i in range(hypos[0].size()[0]):
                tot+=1
                id_fea = ids[i].item()
                ex_id = r2[id_fea]
#                 id = r2[tot]
#                 preds.append((hypos[0][i, :], hypos[1][i, :], hypos[2][i, :], src_len[i], ids[i]))
                preds.append((hypos[0][i, :], hypos[1][i, :], src_len[i], ids[i]))

                if id not in example_index_to_features:
                    example_index_to_features[ex_id] = []
                example_index_to_features[ex_id].append(len(preds)-1)  

        uniq_ex_id = list(set(r2))
        for example_index in uniq_ex_id:
            pred_ids = example_index_to_features[example_index]
            
            prelim_predictions = []
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min mull score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0
            for index in pred_ids:
                hypos =  preds[index]                     
                start_logits = hypos[0]
                end_logits = hypos[1]
#                 token_len = hypos[3]
#                 id_ = hypos[4]
                token_len = hypos[2]
                id_ = hypos[3]
        
                start_indexes = _get_best_indexes(start_logits, 20)
                end_indexes = _get_best_indexes(end_logits, 20)
        
                feature_null_score = hypos[0][0] + hypos[1][0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = id_
                    null_start_logit = hypos[0][0]
                    null_end_logit = hypos[1][0]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                      # We could hypothetically create invalid predictions, e.g., predict
                      # that the start of the span is in the question. We throw out all
                      # invalid predictions.
                        if start_index >= token_len:
                            continue
                        if end_index >= token_len:
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > 30:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                              feature_index=id_,
                              start_index=start_index,
                              end_index=end_index,
                              start_logit=start_logits[start_index],
                              end_logit=end_logits[end_index]))
#                 pdb.set_trace()      
            if len(prelim_predictions)>0:
                best_non_null = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)[0]
            else:
                best_non_null = _PrelimPrediction(
                                  feature_index=min_null_feature_index,
                                  start_index=0,
                                  end_index=0,
                                  start_logit=0,
                                  end_logit=0)
            prelim_predictions.append(
              _PrelimPrediction(
              feature_index=min_null_feature_index,
              start_index=0,
              end_index=0,
              start_logit=null_start_logit,
              end_logit=null_end_logit))
                                              
                                              
            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True)
                                                           
            score_diff = score_null - best_non_null.start_logit - (best_non_null.end_logit)
               
            tmp = ""
            if score_diff > -1:
#                 w1.write(str(min_null_feature_index)+ '\t' + "-1"+ '\t' + "-1" + '\n')
                print(str(min_null_feature_index)+ '\t' + "-1"+ '\t' + "-1")
            else:
                print(str(best_non_null.feature_index.data)+ '\t' + str(best_non_null.start_index) + '\t' + str(best_non_null.end_index))
#                 w1.write(str(best_non_null.feature_index)+ '\t' + str(best_non_null.start_index) + '\t' + str(best_non_null.end_index) + '\n')
#             w1.flush()

def cli_main():
    parser = options.get_generation_parser()
    #Junwei added
    parser.add_argument("--example-id-dir", metavar="DIR", default="tokenized/id",
                       help="destination dir")
    #Junwei added
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
