# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 08:09:40 2016

@author: aeloyq
"""
# -*- coding: utf-8 -*-
import logging
import pprint
import configurations
import numpy
from collections import Counter
from theano import tensor
from toolz import merge

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta,
                               CompositeRule)
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.select import Selector

from machine_translation.checkpoint import CheckpointNMT, LoadNMT
from machine_translation.model import BidirectionalEncoder, Decoder
from machine_translation.sampling import BleuValidator, Sampler
from stream import get_test_stream, get_tr_stream

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)


def main(config):

    # Create Theano variables
    logger.info('Creating theano variables')
    source_sentence = tensor.lmatrix('source')
    source_sentence_mask = tensor.matrix('source_mask')
    target_sentence = tensor.lmatrix('target')
    target_sentence_mask = tensor.matrix('target_mask')
    sampling_input = tensor.lmatrix('input')

    # Construct model
    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(
        config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])
    decoder = Decoder(
        config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
        config['enc_nhids'] * 2)
    cost = decoder.cost(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, target_sentence, target_sentence_mask)

    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    # Initialize model
    logger.info('Initializing model')
    encoder.weights_init = decoder.weights_init = IsotropicGaussian(
        config['weight_scale'])
    encoder.biases_init = decoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    decoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    decoder.transition.weights_init = Orthogonal()
    encoder.initialize()
    decoder.initialize()

    # Set up training model
    logger.info("Building model")
    training_model = Model(cost)

    # Reload model if necessary
    if config['reload']:
        LoadNMT(config['saveto'])

    # Set up beam search and sampling computation graphs if necessary
    if config['bleu_script'] is not None:
        logger.info("Building sampling model")
        sampling_representation = encoder.apply(
            sampling_input, tensor.ones(sampling_input.shape))
        generated = decoder.generate(sampling_input, sampling_representation)
        search_model = Model(generated)
        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))  # generated[1] is next_outputs'''

     
    # Add sampling
    logger.info("Building sampler")
    global samplers_ob
    samplers_ob=Sampler(model=search_model, data_stream=input_sentence_mask,
                hook_samples=config['hook_samples'],
                every_n_batches=config['sampling_freq'],
                src_vocab_size=config['src_vocab_size'])
    # Initialize main loop
    logger.info("Initializing main loop")

'''import cPickle
def get_true_length(self, seq, vocab):
        try:
            return seq.tolist().index(vocab['</S>']) + 1
        except ValueError:
            return len(seq)
def idx_to_word(self, seq, ivocab):
        return " ".join([ivocab.get(idx, "<UNK>") for idx in seq])
def dotests(model,input_sentence_mask):
    configuration = configurations.get_config_en2zh()

    src_vocab = cPickle.load(open(configuration['src_vocab'],'rb'))
    trg_vocab = cPickle.load(open(configuration['trg_vocab'],'rb'))
    src_ivocab = {v: k for k, v in src_vocab.items()}
    trg_ivocab = {v: k for k, v in trg_vocab.items()}
    src_vocab_size = len(src_vocab)

    input_ = input_sentence_mask

    # Sample
    print()
    input_length = get_true_length(input_, src_vocab)
    sampling_fn = model.get_theano_function()
    inp = input_
    _1, outputs, _2, _3, costs = (sampling_fn(inp[None, :]))
    outputs = outputs.flatten()
    costs = costs.T

    sample_length = get_true_length(outputs, trg_vocab)

    print("Input : ", idx_to_word(input_[:input_length],
                                        src_ivocab))
    print("Sample: ", idx_to_word(outputs[:sample_length],
                                        trg_ivocab))
    print("Sample cost: ", costs[:sample_length].sum())
    print()'''
# Get configurations for model
configuration = configurations.get_config_en2zh()
logger.info("Model options:\n{}".format(pprint.pformat(configuration)))
# Get data streams and call main
input_sentence='hello world' #raw_input('Please enter a sentence:\r\n')
write_sentence=open('./data/test.en','w')
write_sentence.write(input_sentence)
write_sentence.close()
#import os
#tokenizer_file = os.path.join('./data', 'tokenizer.perl')
#tokenize_text_files('./data/test.en', tokenizer_file)
input_sentence_mask=get_test_stream(test_set='./data/test.en', src_vocab=configuration['src_vocab'], trg_vocab=configuration['trg_vocab'])
main(configuration)
print samplers_ob,samplers_ob.__class__
samplers_ob.dotests(input_sentence_mask)#input_sentence_mask)
print 'done'
