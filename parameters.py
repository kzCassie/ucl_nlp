## Class struct to store parameters

# For explantion of each param, see init_arg_parser() below:
from common.utils import init_arg_parser
from os.path import basename

#############################################
# Train param, will overwrite default param
seed = 0
# train_file = "data/conala/mined_10.bin"
# dev_file = "data/conala/mined_10.bin"
train_file = "data/conala/train.gold.full.bin"
dev_file = "data/conala/dev.bin"
vocab = "data/conala/vocab.src_freq3.code_freq3.mined_10.bin"
dropout = 0.3
hidden_size = 256
embed_size = 128
action_embed_size = 128
field_embed_size = 64
type_embed_size = 64
lr = 0.001
lr_decay = 0.5
batch_size = 64
max_epoch = 1
beam_size = 15
lstm='lstm'
lr_decay_after_epoch = 15

model_name = f"conala.{lstm}.hidden{hidden_size}.embed{embed_size}.action{action_embed_size}" \
           f".field{field_embed_size}.type{type_embed_size}.dr{dropout}.lr{lr}" \
           f".lr_de{lr_decay}.lr_da{lr_decay_after_epoch}.beam{beam_size}" \
           f".{basename(vocab)}).{basename(train_file)}).glorot" \
           f".par_state.seed{seed}"

train_param = {'seed': seed,
               'mode': 'train',
               'batch_size': batch_size,
               # 'evaluator': "conala_evaluator",
               'asdl_file': "asdl/lang/py3/py3_asdl.simplified.txt",
               'transition_system': 'python3',
               'train_file': train_file,
               'dev_file': dev_file,
               'vocab': vocab,
               'lstm': lstm,
               'no_parent_field_type_embed': True,
               'no_parent_production_embed': True,
               'hidden_size': hidden_size,
               'embed_size': embed_size,
               'action_embed_size': action_embed_size,
               'field_embed_size': field_embed_size,
               'type_embed_size': type_embed_size,
               'dropout': dropout,
               'patience': 5,
               'max_num_trial': 5,
               'glorot_init': True,
               'lr': lr,
               'lr_decay': lr_decay,
               'lr_decay_after_epoch': lr_decay_after_epoch,
               'max_epoch': max_epoch,
               'beam_size': beam_size,
               'log_every': 50,
               'model_name': model_name,
               'save_to':  f"saved_models/conala/{model_name}"}


#######################
# default param
general_config = {'seed': 0,
                 'cuda': False,
                 'lang': 'python3',
                 'asdl_file':None,
                 'mode':None}

modularized_config = {'parser': 'default_parser',
                      'transition_system': 'python3',
                      'evaluator': 'default_evaluator'}

model_config = {'lstm': 'lstm'}

embedding_sizes={'embed_size': 128,
                 'action_embed_size': 128,
                 'field_embed_size': 64,
                 'type_embed_size': 64}

hidden_size = {'hidden_size': 256,
               'ptrnet_hidden_dim': 32,
               'att_vec_size': 256}

readout_layer = {'no_query_vec_to_action_map': False,
                 'readout': 'identity',
                 'query_vec_to_action_diff_map': False}

supervised_attention = {'sup_attention': False}

parent_information = {'no_parent_production_embed': False,
                      'no_parent_field_embed': False,
                      'no_parent_field_type_embed': False,
                      'no_parent_state': False,
                      'no_input_feed': False,
                      'no_copy': False}

training_param = {'vocab': None,
                  'glove_embed_path': None,
                  'train_file': None,
                  'dev_file': None,
                  'pretrain': None,
                  'batch_size': 10,
                  'dropout': 0,
                  'word_dropout': 0,
                  'decoder_word_dropout': 0.3,
                  'primitive_token_label_smoothing': 0.0,
                  'src_token_label_smoothing': 0.0,
                  'negative_sample_type': 'best'}

training_schedule = {'valid_metric': 'acc',
                     'valid_every_epoch': 1,
                     'log_every': 10,
                     'save_to': 'model',
                     'save_all_models': False,
                     'patience': 5,
                     'max_num_trial': 10,
                     'uniform_init': None,
                     'glorot_init': False,
                     'clip_grad': 5.0,
                     'max_epoch': -1,
                     'optimizer': 'Adam',
                     'lr': 0.001,
                     'lr_decay': 0.5,
                     'lr_decay_after_epoch': 0,
                     'decay_lr_every_epoch': 'store_true',
                     'reset_optimizer': 'store_true',
                     'verbose': 'store_true',
                     'eval_top_pred_only': 'store_true'}

decoding_val_test = {'load_model': None,
                     'beam_size': 5,
                     'decode_max_time_step': 100,
                     'sample_size': 5,
                     'test_file': None,
                     'save_decode_to': None}

self_training = {'load_decode_results': None,
                 'unsup_loss_weight': 1.0,
                 'unlabeled_file': None}


class Parameters(object):
    def __init__(self, input_param_dict):
        default_dict = {**general_config, **modularized_config, **model_config,
            **embedding_sizes, **hidden_size, **readout_layer,
            **supervised_attention,**parent_information, **training_param,
            **training_schedule, **decoding_val_test, **self_training}

        for key in default_dict:
            setattr(self, key, default_dict[key])

        for key in input_param_dict:
            setattr(self, key, input_param_dict[key])


if __name__== '__main__':
    from exp import *

    args = Parameters(train_param)
    if args.mode=='train':
        train(args)
        #TODO: redirect log "2>&1 | tee logs/conala/{model_name}.log"
    elif args.mode=='test':
        test(args)
    else:
        raise RuntimeError('unknown mode')
