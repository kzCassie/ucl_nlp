# coding=utf-8
from __future__ import print_function

import os
from six.moves import xrange as range
import math
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from asdl.hypothesis import Hypothesis, GenTokenAction
from asdl.transition_system import ApplyRuleAction, ReduceAction, Action
from common.registerable import Registrable
from components.decode_hypothesis import DecodeHypothesis
from components.action_info import ActionInfo
from components.dataset import Batch
from common.utils import update_args, init_arg_parser
from model import nn_utils
from model.attention_util import AttentionUtil
from model.nn_utils import LabelSmoothing, generate_square_subsequent_mask, length_array_to_mask_tensor
from model.pointer_net import PointerNet
from model.pos_enc import PositionalEncoding


@Registrable.register('transformer_parser')
class TransformerParser(nn.Module):
    """Implementation of a semantic parser

    The parser translates a natural language utterance into an AST defined under
    the ASDL specification, using the transition system described in https://arxiv.org/abs/1810.02720
    """
    def __init__(self, args, vocab, transition_system):
        super(TransformerParser, self).__init__()

        self.args = args
        self.vocab = vocab

        self.transition_system = transition_system
        self.grammar = self.transition_system.grammar

        # Embedding layers

        # source token embedding
        self.src_embed = nn.Embedding(len(vocab.source), args.embed_size)

        # embedding table of ASDL production rules (constructors), one for each ApplyConstructor action,
        # the last entry is the embedding for Reduce action
        self.production_embed = nn.Embedding(len(transition_system.grammar) + 1, args.action_embed_size)

        # embedding table for target primitive tokens
        self.primitive_embed = nn.Embedding(len(vocab.primitive), args.action_embed_size)

        # embedding table for ASDL fields in constructors
        self.field_embed = nn.Embedding(len(transition_system.grammar.fields), args.field_embed_size)

        # embedding table for ASDL types
        self.type_embed = nn.Embedding(len(transition_system.grammar.types), args.type_embed_size)

        nn.init.xavier_normal_(self.src_embed.weight.data)
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.primitive_embed.weight.data)
        nn.init.xavier_normal_(self.field_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)

        # decoder input dimension
        input_dim = args.action_embed_size  # previous action
        # frontier info
        input_dim += args.action_embed_size * (not args.no_parent_production_embed)
        input_dim += args.field_embed_size * (not args.no_parent_field_embed)
        input_dim += args.type_embed_size * (not args.no_parent_field_type_embed)
        self.input_dim = input_dim

        #### Transformer ####
        # Transformer Encoder
        transformer_encoder_layer = nn.TransformerEncoderLayer(args.hidden_size, nhead=args.enc_nhead)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=args.enc_nlayer)
        self.src_pos_encoder = PositionalEncoding(args.hidden_size, dropout=0.1)

        # Transformer Decoder
        transformer_decoder_layer = nn.TransformerDecoderLayer(args.hidden_size, nhead=args.dec_nhead)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=args.dec_nlayer)
        self.tgt_pos_encoder = PositionalEncoding(args.hidden_size, dropout=0.1)

        # Transformer decoder must accepts vectors of the same hidden_size as the encoder.
        self.src_enc_linear = nn.Linear(args.embed_size, args.hidden_size)
        self.tgt_dec_linear = nn.Linear(self.input_dim, args.hidden_size)
        #####################

        if args.no_copy is False:
            # pointer net for copying tokens from source side
            self.src_pointer_net = PointerNet(query_vec_size=args.hidden_size, src_encoding_size=args.hidden_size)

            # given the decoder's hidden state, predict whether to copy or generate a target primitive token
            # output: [p(gen(token)) | s_t, p(copy(token)) | s_t]

            self.primitive_predictor = nn.Linear(args.hidden_size, 2)

        if args.primitive_token_label_smoothing:
            self.label_smoothing = LabelSmoothing(args.primitive_token_label_smoothing, len(self.vocab.primitive), ignore_indices=[0, 1, 2])

        # bias for predicting ApplyConstructor and GenToken actions
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(transition_system.grammar) + 1).zero_())
        self.tgt_token_readout_b = nn.Parameter(torch.FloatTensor(len(vocab.primitive)).zero_())

        if args.no_query_vec_to_action_map:
            # if there is no additional linear layer between the attentional vector (i.e., the query vector)
            # and the final softmax layer over target actions, we use the attentional vector to compute action
            # probabilities

            assert args.att_vec_size == args.action_embed_size
            self.production_readout = lambda q: F.linear(q, self.production_embed.weight, self.production_readout_b)
            self.tgt_token_readout = lambda q: F.linear(q, self.primitive_embed.weight, self.tgt_token_readout_b)
        else:
            # by default, we feed the attentional vector (i.e., the query vector) into a linear layer without bias, and
            # compute action probabilities by dot-producting the resulting vector and (GenToken, ApplyConstructor) action embeddings
            # i.e., p(action) = query_vec^T \cdot W \cdot embedding

            self.query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.embed_size, bias=args.readout == 'non_linear')
            if args.query_vec_to_action_diff_map:
                # use different linear transformations for GenToken and ApplyConstructor actions
                self.query_vec_to_primitive_embed = nn.Linear(args.att_vec_size, args.embed_size, bias=args.readout == 'non_linear')
            else:
                self.query_vec_to_primitive_embed = self.query_vec_to_action_embed

            self.read_out_act = torch.tanh if args.readout == 'non_linear' else nn_utils.identity

            self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                         self.production_embed.weight, self.production_readout_b)
            self.tgt_token_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_primitive_embed(q)),
                                                        self.primitive_embed.weight, self.tgt_token_readout_b)

        # dropout layer
        self.dropout = nn.Dropout(args.dropout)

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

    def encode(self, src_sents_var, src_sents_len):
        """Encode the input natural language utterance

        Args:
            src_sents_var: a variable of shape (src_sent_len, batch_size), representing word ids of the input
            src_sents_len: a list of lengths of input source sentences, sorted by descending order

        Returns:
            src_encodings: source encodings of shape (src_sent_len, batch_size, hidden_size)
        """
        args = self.args

        # (src_sent_len, batch_size, embed_size)
        # apply word dropout
        if self.training and args.word_dropout:
            mask = Variable(self.new_tensor(src_sents_var.size()).fill_(1. - args.word_dropout).bernoulli().long())
            src_sents_var = src_sents_var * mask + (1 - mask) * self.vocab.source.unk_id

        # (src_sent_len, batch_size, hidden_size)
        src_enc_vec = torch.tanh(self.src_enc_linear(self.src_embed(src_sents_var)))
        # (src_sent_len, batch_size, hidden_size)
        src = self.src_pos_encoder(src_enc_vec * math.sqrt(args.embed_size))
        # (src_sent_len,src_sent_len)
        src_mask = generate_square_subsequent_mask(src.shape[0])
        # (batch_size, src_sent_len)
        src_key_padding_mask = length_array_to_mask_tensor(src_sents_len, args.cuda)

        # (src_sent_len, batch_size, 128????) TODO: size of encoder output
        src_encodings = self.transformer_encoder(src, src_mask, src_key_padding_mask)

        # TODO: shape assertion
        src_sent_len, batch_size = src_sents_var.shape
        assert(src_enc_vec.shape == (src_sent_len, batch_size, args.hidden_size))
        assert(src.shape == (src_sent_len, batch_size, args.hidden_size))
        assert(src_mask.shape == (src_sent_len, src_sent_len))
        assert(src_key_padding_mask.shape == (batch_size, src_sent_len))
        assert(src_encodings.shape == (src_sent_len, batch_size, args.hidden_size))

        return src_encodings, src_key_padding_mask

    def decode(self, batch, src_encodings, src_key_padding_mask):
        """Given a batch of examples and their encodings of input utterances,
        compute query vectors at each decoding time step, which are used to compute
        action probabilities

        Args:
            batch: a `Batch` object storing input examples
            src_encodings: variable of shape (src_sent_len, batch_size, hidden_size), encodings of source utterances
            src_key_padding_mask: to be used as the memory_key_padding_mask for the attention decoder

        Returns:
            Query vectors, a variable of shape (tgt_action_len, batch_size, hidden_size)
        """
        batch_size = len(batch)
        tgt_action_len = batch.max_action_num
        args = self.args

        # # (batch_size, query_len, 128)
        # src_encodings_att_linear = self.att_src_linear(src_encodings)
        # TODO:additional linear layer to remain parallel to LSTM model?

        # Input of the decoder is composite of diff parts.
        # Each action recursively depends on the previous action
        # Compared with the original tranX, parts that are specific to RNN are removed
        xs = []
        history_states = []
        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())
        for t in range(tgt_action_len):
            # the input to the decoder LSTM is a concatenation of multiple signals
            # [
            #   embedding of previous action -> `a_tm1_embed`,
            #   embedding of the current frontier (parent) constructor (rule) -> `parent_production_embed`,
            #   embedding of the frontier (parent) field -> `parent_field_embed`,
            #   embedding of the ASDL type of the frontier field -> `parent_field_type_embed`,
            # ]

            if t == 0:
                # no previous action
                x = Variable(self.new_tensor(batch_size, self.input_dim).zero_(), requires_grad=False)

                # initialize using the root type embedding
                if args.no_parent_field_type_embed is False:
                    offset = args.action_embed_size  # prev_action
                    offset += args.action_embed_size * (not args.no_parent_production_embed)
                    offset += args.field_embed_size * (not args.no_parent_field_embed)

                    x[:, offset: offset + args.type_embed_size] = self.type_embed(Variable(self.new_long_tensor(
                        [self.grammar.type2id[self.grammar.root_type] for e in batch.examples])))
            else:
                a_tm1_embeds = []
                for example in batch.examples:
                    # action t - 1
                    if t < len(example.tgt_actions):
                        a_tm1 = example.tgt_actions[t - 1]
                        if isinstance(a_tm1.action, ApplyRuleAction):
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.action.production]]
                        elif isinstance(a_tm1.action, ReduceAction):
                            a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                        else:
                            a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[a_tm1.action.token]]
                    else:
                        a_tm1_embed = zero_action_embed

                    a_tm1_embeds.append(a_tm1_embed)

                # (batch_size, embed_size)
                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]
                if args.no_parent_production_embed is False:
                    parent_production_embed = self.production_embed(batch.get_frontier_prod_idx(t))
                    inputs.append(parent_production_embed)
                if args.no_parent_field_embed is False:
                    parent_field_embed = self.field_embed(batch.get_frontier_field_idx(t))
                    inputs.append(parent_field_embed)
                if args.no_parent_field_type_embed is False:
                    parent_field_type_embed = self.type_embed(batch.get_frontier_field_type_idx(t))
                    inputs.append(parent_field_type_embed)

                # (batch_size, input_dim)
                x = torch.cat(inputs, dim=-1)
                assert(x.shape==(batch_size, self.input_dim))
            xs.append(x)

        # decoder_inputs: (tgt_action_len, batch_size, input_dim)
        decoder_inputs = torch.stack(xs, dim=0)
        assert decoder_inputs.shape == (batch.max_action_num, batch_size, self.input_dim)

        # Transformer decoder
        # (tgt_action_len, batch_size, hidden_size)
        tgt_dec_vec = torch.tanh(self.tgt_dec_linear(decoder_inputs))
        # (tgt_action_len, batch_size, hidden_size)
        tgt = self.tgt_pos_encoder(tgt_dec_vec * math.sqrt(self.input_dim))
        # (tgt_action_len, tgt_action_len)
        tgt_mask = generate_square_subsequent_mask(tgt_action_len)
        memory_mask = None
        # (batch_size, tgt_action_len)
        tgt_key_padding_mask = batch.tgt_action_mask
        # (batch_size, src_sent_len)
        memory_key_padding_mask = src_key_padding_mask.clone()

        # TODO: shape assertion
        assert(tgt_dec_vec.shape==(tgt_action_len, batch_size, args.hidden_size))
        assert(tgt.shape==(tgt_action_len, batch_size, args.hidden_size))
        assert(tgt_mask.shape==(tgt_action_len, tgt_action_len))
        assert(tgt_key_padding_mask.shape==(batch_size, tgt_action_len))

        att_vecs = self.transformer_decoder(tgt, src_encodings, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask)
        assert(att_vecs.shape==(tgt_action_len, batch_size, args.hidden_size))

        return att_vecs

    def step(self, x, src_encodings, src_key_padding_mask, hyp_len):
        """
        At each step during inference time, x contains embeddings of tentative hypothesis. We need to mask
        appropriately and pass the entire x into the transformer decoder to get the updated att_vec for each
        hypothesis.

        Args:
            x: tgt inputs of shape (t, hyp_num, input_dim), t is the max hypothesis length at step t during inference.
            src_encodings: variable of shape (src_sent_len, batch_size, hidden_size), encodings of source utterances.
            src_key_padding_mask: to be used as the memory_key_padding_mask for the attention decoder.
            hyp_len: in-progress hypothesis length np.array of shape (hyp_num,). All values = t.

        Returns:
            att_t: output of the transformer decoder for the t-th step of the shape (t, hidden_size).
        """
        batch_size = x.shape[1]
        tgt_action_len = x.shape[0]
        args = self.args

        # Transformer decoder
        # (tgt_action_len, batch_size, hidden_size)
        tgt_dec_vec = torch.tanh(self.tgt_dec_linear(x))
        # (tgt_action_len, batch_size, hidden_size)
        tgt = self.tgt_pos_encoder(tgt_dec_vec * math.sqrt(self.input_dim))
        # (tgt_action_len, tgt_action_len)
        tgt_mask = generate_square_subsequent_mask(tgt_action_len)
        memory_mask = None
        # (batch_size, tgt_action_len)
        tgt_key_padding_mask = length_array_to_mask_tensor(hyp_len, args.cuda)
        # (batch_size, src_sent_len)
        memory_key_padding_mask = src_key_padding_mask.clone()

        # TODO: shape assertion
        assert(tgt_dec_vec.shape==(tgt_action_len, batch_size, args.hidden_size))
        assert(tgt.shape==(tgt_action_len, batch_size, args.hidden_size))
        assert(tgt_mask.shape==(tgt_action_len, tgt_action_len))
        assert(tgt_key_padding_mask.shape==(batch_size, tgt_action_len))

        att_vecs = self.transformer_decoder(tgt, src_encodings, tgt_mask, memory_mask,
                                            tgt_key_padding_mask, memory_key_padding_mask)
        assert(att_vecs.shape==(tgt_action_len, batch_size, args.hidden_size))

        return att_vecs[-1, :, :]

    def score(self, examples, return_encode_state=False):
        """Training time
        Given a list of examples, compute the log-likelihood of generating the target AST

        Args:
            examples: a batch of examples
            return_encode_state: return encoding states of input utterances
        output: score for each training example: Variable(batch_size)
        """
        batch = Batch(examples, self.grammar, self.vocab, copy=self.args.no_copy is False, cuda=self.args.cuda)

        # src_encodings: (src_sent_len, batch_size, hidden_size)
        src_encodings, src_key_padding_mask = self.encode(batch.src_sents_var, batch.src_sents_len)

        # query vectors are sufficient statistics used to compute action probabilities
        # query_vectors: (tgt_action_len, batch_size, hidden_size)
        query_vectors = self.decode(batch, src_encodings, src_key_padding_mask)

        # src_encodings: (batch_size, src_sent_len, hidden_size)
        src_encodings = src_encodings.permute(1, 0, 2)

        # ApplyRule (i.e., ApplyConstructor) action probabilities
        # (tgt_action_len, batch_size, grammar_size)
        apply_rule_prob = F.softmax(self.production_readout(query_vectors), dim=-1)

        # probabilities of target (gold-standard) ApplyRule actions
        # (tgt_action_len, batch_size)
        tgt_apply_rule_prob = torch.gather(apply_rule_prob, dim=2,
                                           index=batch.apply_rule_idx_matrix.unsqueeze(2)).squeeze(2)

        #### compute generation and copying probabilities

        # (tgt_action_len, batch_size, primitive_vocab_size)
        gen_from_vocab_prob = F.softmax(self.tgt_token_readout(query_vectors), dim=-1)

        # (tgt_action_len, batch_size)
        tgt_primitive_gen_from_vocab_prob = torch.gather(gen_from_vocab_prob, dim=2,
                                                         index=batch.primitive_idx_matrix.unsqueeze(2)).squeeze(2)

        if self.args.no_copy:
            # mask positions in action_prob that are not used

            if self.training and self.args.primitive_token_label_smoothing:
                # (tgt_action_len, batch_size)
                # this is actually the negative KL divergence size we will flip the sign later
                # tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
                #     gen_from_vocab_prob.view(-1, gen_from_vocab_prob.size(-1)).log(),
                #     batch.primitive_idx_matrix.view(-1)).view(-1, len(batch))

                tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
                    gen_from_vocab_prob.log(),
                    batch.primitive_idx_matrix)
            else:
                tgt_primitive_gen_from_vocab_log_prob = tgt_primitive_gen_from_vocab_prob.log()

            # (tgt_action_len, batch_size)
            action_prob = tgt_apply_rule_prob.log() * batch.apply_rule_mask + \
                          tgt_primitive_gen_from_vocab_log_prob * batch.gen_token_mask
        else:
            # binary gating probabilities between generating or copying a primitive token
            # (tgt_action_len, batch_size, 2)
            primitive_predictor = F.softmax(self.primitive_predictor(query_vectors), dim=-1)

            # pointer network copying scores over source tokens
            # (tgt_action_len, batch_size, src_sent_len)
            primitive_copy_prob = self.src_pointer_net(src_encodings, batch.src_token_mask, query_vectors)

            # marginalize over the copy probabilities of tokens that are same
            # (tgt_action_len, batch_size)
            tgt_primitive_copy_prob = torch.sum(primitive_copy_prob * batch.primitive_copy_token_idx_mask, dim=-1)

            # mask positions in action_prob that are not used
            # (tgt_action_len, batch_size)
            action_mask_pad = torch.eq(batch.apply_rule_mask + batch.gen_token_mask + batch.primitive_copy_mask, 0.)
            action_mask = 1. - action_mask_pad.float()

            # (tgt_action_len, batch_size)
            action_prob = tgt_apply_rule_prob * batch.apply_rule_mask + \
                          primitive_predictor[:, :, 0] * tgt_primitive_gen_from_vocab_prob * batch.gen_token_mask + \
                          primitive_predictor[:, :, 1] * tgt_primitive_copy_prob * batch.primitive_copy_mask

            # avoid nan in log
            action_prob.data.masked_fill_(action_mask_pad.data, 1.e-7)

            action_prob = action_prob.log() * action_mask

        scores = torch.sum(action_prob, dim=0)

        returns = [scores]
        return returns

    def parse(self, src_sent, context=None, beam_size=5, debug=False):
        """Test time
        Perform beam search to infer the target AST given a source utterance

        Args:
            src_sent: list of source utterance tokens
            context: other context used for prediction
            beam_size: beam size

        Returns:
            A list of `DecodeHypothesis`, each representing an AST
        """
        args = self.args
        primitive_vocab = self.vocab.primitive
        T = torch.cuda if args.cuda else torch

        src_sent_var = nn_utils.to_input_variable([src_sent], self.vocab.source, cuda=args.cuda, training=False)

        # Variable(src_sent_len, 1, hidden_size)
        src_encodings, src_key_padding_mask = self.encode(src_sent_var, [len(src_sent)])

        # decoding
        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())

        with torch.no_grad():
            hyp_scores = Variable(self.new_tensor([0.]))

        # For computing copy probabilities, we marginalize over tokens with the same surface form
        # `aggregated_primitive_tokens` stores the position of occurrence of each source token
        aggregated_primitive_tokens = OrderedDict()
        for token_pos, token in enumerate(src_sent):
            aggregated_primitive_tokens.setdefault(token, []).append(token_pos)

        t = 0
        hypotheses = [DecodeHypothesis(rec_embed=True)]
        completed_hypotheses = []
        xs = []

        while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
            hyp_num = len(hypotheses)

            # (src_sent_len, hyp_num, hidden_size)
            exp_src_encodings = src_encodings.expand(src_encodings.size(0), hyp_num, src_encodings.size(2))

            if t == 0:
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.input_dim).zero_())
                if args.no_parent_field_type_embed is False:
                    offset = args.action_embed_size  # prev_action
                    offset += args.action_embed_size * (not args.no_parent_production_embed)
                    offset += args.field_embed_size * (not args.no_parent_field_embed)

                    x[0, offset: offset + args.type_embed_size] = \
                        self.type_embed.weight[self.grammar.type2id[self.grammar.root_type]]

                # store history embedding in hyp
                assert len(hypotheses) == 1
                for hyp in hypotheses:
                    hyp.action_embed.append(x)
            else:
                actions_tm1 = [hyp.actions[-1] for hyp in hypotheses]

                a_tm1_embeds = []
                for a_tm1 in actions_tm1:
                    if a_tm1:
                        if isinstance(a_tm1, ApplyRuleAction):
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.production]]
                        elif isinstance(a_tm1, ReduceAction):
                            a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                        else:
                            a_tm1_embed = self.primitive_embed.weight[self.vocab.primitive[a_tm1.token]]

                        a_tm1_embeds.append(a_tm1_embed)
                    else:
                        a_tm1_embeds.append(zero_action_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]
                if args.no_parent_production_embed is False:
                    # frontier production
                    frontier_prods = [hyp.frontier_node.production for hyp in hypotheses]
                    frontier_prod_embeds = self.production_embed(Variable(self.new_long_tensor(
                        [self.grammar.prod2id[prod] for prod in frontier_prods])))
                    inputs.append(frontier_prod_embeds)
                if args.no_parent_field_embed is False:
                    # frontier field
                    frontier_fields = [hyp.frontier_field.field for hyp in hypotheses]
                    frontier_field_embeds = self.field_embed(Variable(self.new_long_tensor([
                        self.grammar.field2id[field] for field in frontier_fields])))

                    inputs.append(frontier_field_embeds)
                if args.no_parent_field_type_embed is False:
                    # frontier field type
                    frontier_field_types = [hyp.frontier_field.type for hyp in hypotheses]
                    frontier_field_type_embeds = self.type_embed(Variable(self.new_long_tensor([
                        self.grammar.type2id[type] for type in frontier_field_types])))
                    inputs.append(frontier_field_type_embeds)

                x = torch.cat(inputs, dim=-1)

                # store history embedding in hyp
                for i, hyp in enumerate(hypotheses):
                    hyp.action_embed.append(x[i, :])

            decoder_inputs = []
            for hyp in hypotheses:
                decoder_inputs.append(hyp.get_hist_action_embeddings())
            decoder_inputs = torch.cat(decoder_inputs, dim=1)

            print(f"decoder_inputs:{decoder_inputs.shape}")
            assert decoder_inputs.shape == (t+1, hyp_num, self.input_dim)

            hyp_len = (np.ones(hyp_num) * (t+1)).astype('int')      #TODO: is it correct for completed actions?
            # (t, hidden_size)
            att_t = self.step(decoder_inputs, exp_src_encodings, src_key_padding_mask, hyp_len)
            assert att_t.shape == (t+1, args.hidden_size)

            # Variable(batch_size, grammar_size)
            # apply_rule_log_prob = torch.log(F.softmax(self.production_readout(att_t), dim=-1))
            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

            # Variable(batch_size, primitive_vocab_size)
            gen_from_vocab_prob = F.softmax(self.tgt_token_readout(att_t), dim=-1)

            if args.no_copy:
                primitive_prob = gen_from_vocab_prob
            else:
                # Variable(batch_size, src_sent_len)
                primitive_copy_prob = self.src_pointer_net(src_encodings, None, att_t.unsqueeze(0)).squeeze(0)

                # Variable(batch_size, 2)
                primitive_predictor_prob = F.softmax(self.primitive_predictor(att_t), dim=-1)

                # Variable(batch_size, primitive_vocab_size)
                primitive_prob = primitive_predictor_prob[:, 0].unsqueeze(1) * gen_from_vocab_prob

                # if src_unk_pos_list:
                #     primitive_prob[:, primitive_vocab.unk_id] = 1.e-10

            gentoken_prev_hyp_ids = []
            gentoken_new_hyp_unks = []
            applyrule_new_hyp_scores = []
            applyrule_new_hyp_prod_ids = []
            applyrule_prev_hyp_ids = []

            for hyp_id, hyp in enumerate(hypotheses):
                # generate new continuations
                action_types = self.transition_system.get_valid_continuation_types(hyp)

                for action_type in action_types:
                    if action_type == ApplyRuleAction:
                        productions = self.transition_system.get_valid_continuating_productions(hyp)
                        for production in productions:
                            prod_id = self.grammar.prod2id[production]
                            prod_score = apply_rule_log_prob[hyp_id, prod_id].data.item()
                            new_hyp_score = hyp.score + prod_score

                            applyrule_new_hyp_scores.append(new_hyp_score)
                            applyrule_new_hyp_prod_ids.append(prod_id)
                            applyrule_prev_hyp_ids.append(hyp_id)
                    elif action_type == ReduceAction:
                        action_score = apply_rule_log_prob[hyp_id, len(self.grammar)].data.item()
                        new_hyp_score = hyp.score + action_score

                        applyrule_new_hyp_scores.append(new_hyp_score)
                        applyrule_new_hyp_prod_ids.append(len(self.grammar))
                        applyrule_prev_hyp_ids.append(hyp_id)
                    else:
                        # GenToken action
                        gentoken_prev_hyp_ids.append(hyp_id)
                        hyp_copy_info = dict()  # of (token_pos, copy_prob)
                        hyp_unk_copy_info = []

                        if args.no_copy is False:
                            for token, token_pos_list in aggregated_primitive_tokens.items():
                                sum_copy_prob = torch.gather(primitive_copy_prob[hyp_id], 0, Variable(T.LongTensor(token_pos_list))).sum()
                                gated_copy_prob = primitive_predictor_prob[hyp_id, 1] * sum_copy_prob

                                if token in primitive_vocab:
                                    token_id = primitive_vocab[token]
                                    primitive_prob[hyp_id, token_id] = primitive_prob[hyp_id, token_id] + gated_copy_prob

                                    hyp_copy_info[token] = (token_pos_list, gated_copy_prob.data.item())
                                else:
                                    hyp_unk_copy_info.append({'token': token, 'token_pos_list': token_pos_list,
                                                              'copy_prob': gated_copy_prob.data.item()})

                        if args.no_copy is False and len(hyp_unk_copy_info) > 0:
                            unk_i = np.array([x['copy_prob'] for x in hyp_unk_copy_info]).argmax()
                            token = hyp_unk_copy_info[unk_i]['token']
                            primitive_prob[hyp_id, primitive_vocab.unk_id] = hyp_unk_copy_info[unk_i]['copy_prob']
                            gentoken_new_hyp_unks.append(token)

                            hyp_copy_info[token] = (hyp_unk_copy_info[unk_i]['token_pos_list'], hyp_unk_copy_info[unk_i]['copy_prob'])

            new_hyp_scores = None
            if applyrule_new_hyp_scores:
                new_hyp_scores = Variable(self.new_tensor(applyrule_new_hyp_scores))
            if gentoken_prev_hyp_ids:
                primitive_log_prob = torch.log(primitive_prob)
                gen_token_new_hyp_scores = (hyp_scores[gentoken_prev_hyp_ids].unsqueeze(1) + primitive_log_prob[gentoken_prev_hyp_ids, :]).view(-1)

                if new_hyp_scores is None: new_hyp_scores = gen_token_new_hyp_scores
                else: new_hyp_scores = torch.cat([new_hyp_scores, gen_token_new_hyp_scores])
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores,
                                                             k=min(new_hyp_scores.size()[0], beam_size - len(completed_hypotheses)))

            live_hyp_ids = []
            new_hypotheses = []
            for new_hyp_score, new_hyp_pos in zip(top_new_hyp_scores.data.cpu(), top_new_hyp_pos.data.cpu()):
                action_info = ActionInfo()
                if new_hyp_pos < len(applyrule_new_hyp_scores):
                    # it's an ApplyRule or Reduce action
                    prev_hyp_id = applyrule_prev_hyp_ids[new_hyp_pos]
                    prev_hyp = hypotheses[prev_hyp_id]

                    prod_id = applyrule_new_hyp_prod_ids[new_hyp_pos]
                    # ApplyRule action
                    if prod_id < len(self.grammar):
                        production = self.grammar.id2prod[prod_id]
                        action = ApplyRuleAction(production)
                    # Reduce action
                    else:
                        action = ReduceAction()
                else:
                    # it's a GenToken action
                    token_id = (new_hyp_pos - len(applyrule_new_hyp_scores)) % primitive_prob.size(1)

                    k = (new_hyp_pos - len(applyrule_new_hyp_scores)) // primitive_prob.size(1)
                    # try:
                    # copy_info = gentoken_copy_infos[k]
                    prev_hyp_id = gentoken_prev_hyp_ids[k]
                    prev_hyp = hypotheses[prev_hyp_id]
                    # except:
                    #     print('k=%d' % k, file=sys.stderr)
                    #     print('primitive_prob.size(1)=%d' % primitive_prob.size(1), file=sys.stderr)
                    #     print('len copy_info=%d' % len(gentoken_copy_infos), file=sys.stderr)
                    #     print('prev_hyp_id=%s' % ', '.join(str(i) for i in gentoken_prev_hyp_ids), file=sys.stderr)
                    #     print('len applyrule_new_hyp_scores=%d' % len(applyrule_new_hyp_scores), file=sys.stderr)
                    #     print('len gentoken_prev_hyp_ids=%d' % len(gentoken_prev_hyp_ids), file=sys.stderr)
                    #     print('top_new_hyp_pos=%s' % top_new_hyp_pos, file=sys.stderr)
                    #     print('applyrule_new_hyp_scores=%s' % applyrule_new_hyp_scores, file=sys.stderr)
                    #     print('new_hyp_scores=%s' % new_hyp_scores, file=sys.stderr)
                    #     print('top_new_hyp_scores=%s' % top_new_hyp_scores, file=sys.stderr)
                    #
                    #     torch.save((applyrule_new_hyp_scores, primitive_prob), 'data.bin')
                    #
                    #     # exit(-1)
                    #     raise ValueError()

                    if token_id == primitive_vocab.unk_id:
                        if gentoken_new_hyp_unks:
                            token = gentoken_new_hyp_unks[k]
                        else:
                            token = primitive_vocab.id2word(primitive_vocab.unk_id)
                    else:
                        token = primitive_vocab.id2word(token_id.item())

                    action = GenTokenAction(token)

                    if token in aggregated_primitive_tokens:
                        action_info.copy_from_src = True
                        action_info.src_token_position = aggregated_primitive_tokens[token]

                    if debug:
                        action_info.gen_copy_switch = 'n/a' if args.no_copy else primitive_predictor_prob[prev_hyp_id, :].log().cpu().data.numpy()
                        action_info.in_vocab = token in primitive_vocab
                        action_info.gen_token_prob = gen_from_vocab_prob[prev_hyp_id, token_id].log().cpu().data.item() \
                            if token in primitive_vocab else 'n/a'
                        action_info.copy_token_prob = torch.gather(primitive_copy_prob[prev_hyp_id],
                                                                   0,
                                                                   Variable(T.LongTensor(action_info.src_token_position))).sum().log().cpu().data.item() \
                            if args.no_copy is False and action_info.copy_from_src else 'n/a'

                action_info.action = action
                action_info.t = t
                if t > 0:
                    action_info.parent_t = prev_hyp.frontier_node.created_time
                    action_info.frontier_prod = prev_hyp.frontier_node.production
                    action_info.frontier_field = prev_hyp.frontier_field.field

                if debug:
                    action_info.action_prob = new_hyp_score - prev_hyp.score

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score

                if new_hyp.completed:
                    # add length normalization
                    new_hyp.score /= (t+1)
                    completed_hypotheses.append(new_hyp)
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                hypotheses = new_hypotheses
                hyp_scores = Variable(self.new_tensor([hyp.score for hyp in hypotheses]))
                t += 1
            else:
                break

        completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'transition_system': self.transition_system,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

    @classmethod
    def load(cls, model_path, cuda=False):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        transition_system = params['transition_system']
        saved_args = params['args']
        # update saved args
        update_args(saved_args, init_arg_parser())
        saved_state = params['state_dict']
        saved_args.cuda = cuda

        parser = cls(saved_args, vocab, transition_system)

        parser.load_state_dict(saved_state)

        if cuda: parser = parser.cuda()
        parser.eval()

        return parser
