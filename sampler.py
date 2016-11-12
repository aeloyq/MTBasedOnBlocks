# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 08:09:40 2016

@author: aeloyq
"""
# -*- coding: utf-8 -*-
import logging
import pprint
import configurations
from theano import tensor
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model

from machine_translation.checkpoint import LoadNMT
from machine_translation.model import BidirectionalEncoder, Decoder
from machine_translation.sampling import Sampler
from stream import get_test_stream


logger = logging.getLogger(__name__)


def main(config):
    print('working on it ...')
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
    # Extensions
    extensions = []
    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto']))

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
    main_loop = MainLoop(
        model=training_model,
        algorithm=None,
        data_stream=None,
        extensions=extensions
    )
                
    for extension in main_loop.extensions:
        extension.main_loop = main_loop
    main_loop._run_extensions('before_training')
    


# Get configurations for model
configuration = configurations.get_config_en2zh()
logger.info("Model options:\n{}".format(pprint.pformat(configuration)))
# Get data streams and call main
input_sentence='hello world' #raw_input('Please enter a sentence:\r\n')
write_sentence=open('./data/test.en','w')
write_sentence.write(input_sentence)
write_sentence.close()
input_sentence_mask=get_test_stream(test_set='./data/test.en', src_vocab=configuration['src_vocab'], trg_vocab=configuration['trg_vocab'])
print('loading main function')
main(configuration)
print('loaded main function')
samplers_ob.dotests(input_sentence_mask)
print 'done'
