#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#  Copyright 2015 Pascual Martinez-Gomez
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import print_function

import copy
import copyreg
import argparse
import codecs
import logging
from lxml import etree
from multiprocessing import Pool
from multiprocessing import Lock
import os
import sys
import textwrap

from nltk.sem.logic import LogicalExpressionException

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
from lxml import etree

def element_unpickler(data):
    return etree.fromstring(data)

def element_pickler(element):
    data = etree.tostring(element)
    return element_unpickler, (data,)

copyreg.pickle(etree._Element, element_pickler, element_unpickler)

def elementtree_unpickler(data):
    data = StringIO(data)
    return etree.parse(data)

def elementtree_pickler(tree):
    data = StringIO()
    tree.write(data)
    return elementtree_unpickler, (data.getvalue(),)

copyreg.pickle(etree._ElementTree, elementtree_pickler, elementtree_unpickler)

#  Added to be called like a library function.
if __name__ != '__main__':
    sys.path.append(os.path.dirname(__file__))

from ccg2lambda_tools import assign_semantics_to_ccg
from semantic_index import SemanticIndex

SEMANTIC_INDEX=None
ARGS=None
SENTENCES=None
kMaxTasksPerChild=None

def main(args = None):
    global SEMANTIC_INDEX
    global ARGS
    global SENTENCES
    DESCRIPTION=textwrap.dedent("""\
            categories_template.yaml should contain the semantic templates
              in YAML format.
            parsed_sentence.xml contains the CCG-parsed sentences.
            If --arbi-types is specified, then the arbitrary specification of
              types is enabled, thus using the argument as the field of the semantic
              template that is used. E.g, by specifying "--arbi-types coq_type"
              and a semantic template:
            - semantics: \P x.P(x)
              category: NP
              coq_type: Animal
            The type "Animal" will be used for this expression. Otherwise,
            types of the sem/logic module of NLTK are used.
      """)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=DESCRIPTION)
    parser.add_argument("ccg")
    parser.add_argument("templates")
    parser.add_argument("sem")
    parser.add_argument("--arbi-types", action="store_true", default=False)
    parser.add_argument("--gold_trees", action="store_true", default=True)
    parser.add_argument("--nbest", nargs='?', type=int, default="0")
    parser.add_argument("--ncores", nargs='?', type=int, default="3",
        help="Number of cores for multiprocessing.")
    ARGS = parser.parse_args()
      
    if not os.path.exists(ARGS.templates):
        print('File does not exist: {0}'.format(ARGS.templates))
        sys.exit(1)
    if not os.path.exists(ARGS.ccg):
        print('File does not exist: {0}'.format(ARGS.ccg))
        sys.exit(1)
    
    logging.basicConfig(level=logging.WARNING)

    SEMANTIC_INDEX = SemanticIndex(ARGS.templates)

    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(ARGS.ccg, parser)

    SENTENCES = root.findall('.//sentence')
    # print('Found {0} sentences'.format(len(SENTENCES)))
    # from pudb import set_trace; set_trace()
    sentence_inds = range(len(SENTENCES))
    sem_nodes_lists = semantic_parse_sentences(sentence_inds, ARGS.ncores)
    assert len(sem_nodes_lists) == len(SENTENCES), \
        'Element mismatch: {0} vs {1}'.format(len(sem_nodes_lists), len(SENTENCES))
    logging.info('Adding XML semantic nodes to sentences...')
    for sentence, sem_nodes in zip(SENTENCES, sem_nodes_lists):
        sentence.extend(sem_nodes)
    logging.info('Finished adding XML semantic nodes to sentences.')

    root_xml_str = serialize_tree(root)
    with codecs.open(ARGS.sem, 'wb') as fout:
        fout.write(root_xml_str)

def semantic_parse_sentences(sentence_inds, ncores=1):
    if ncores <= 1:
        sem_nodes_lists = semantic_parse_sentences_seq(sentence_inds)
    else:
        sem_nodes_lists = semantic_parse_sentences_par(sentence_inds, ncores)
    sem_nodes_lists = [
        [etree.fromstring(s) for s in sem_nodes] for sem_nodes in sem_nodes_lists]
    return sem_nodes_lists

def semantic_parse_sentences_par(sentence_inds, ncores=3):
    params_list = []
    for sentence_ind in sentence_inds:
        semantic_index_copy = copy.deepcopy(SEMANTIC_INDEX)
        args_copy = copy.deepcopy(ARGS)
        sentences_copy = copy.deepcopy(SENTENCES)
        params_list.append([semantic_index_copy, args_copy, sentences_copy, sentence_ind])
    pool = Pool(processes=ncores, maxtasksperchild=kMaxTasksPerChild)
    sem_nodes = pool.map(semantic_parse_sentence, params_list)
    pool.close()
    pool.join()
    return sem_nodes

def semantic_parse_sentences_seq(sentence_inds):
    sem_nodes = []
    for sentence_ind in sentence_inds:
        semantic_index_copy = copy.deepcopy(SEMANTIC_INDEX)
        args_copy = copy.deepcopy(ARGS)
        sentences_copy = copy.deepcopy(SENTENCES)
        sem_node = semantic_parse_sentence([semantic_index_copy, args_copy, sentences_copy, sentence_ind])
        sem_nodes.append(sem_node)
    return sem_nodes

def semantic_parse_sentence(params):
    """
    `sentence` is an lxml tree with tokens and ccg nodes.
    It returns an lxml semantics node.
    """
    semantic_index_, args_, sentences_, sentence_ind = params
    sentence = sentences_[sentence_ind]
    sem_nodes = []
    # TODO: try to prevent semantic parsing for fragmented CCG trees.
    # Otherwise, produce fragmented semantics.
    if args_.gold_trees:
        # In xpath, elements are 1-indexed.
        # However, gold_tree annotations assumed zero-index.
        # This line fixes it.
        tree_indices = [int(sentence.get('gold_tree', '0')) + 1]
    if args_.nbest != 1:
        tree_indices = get_tree_indices(sentence, args_.nbest)
    for tree_index in tree_indices: 
        sem_node = etree.Element('semantics')
        try:
            sem_tree = assign_semantics_to_ccg(
                sentence, semantic_index_, tree_index)
            filter_attributes(sem_tree)
            sem_node.extend(sem_tree.xpath('.//descendant-or-self::span'))
            sem_node.set('status', 'success')
            sem_node.set('ccg_id',
                sentence.xpath('./ccg[{0}]/@id'.format(tree_index))[0])
            sem_node.set('root',
                sentence.xpath('./ccg[{0}]/@root'.format(tree_index))[0])
            # print('.', end='', file=sys.stdout)
            sys.stdout.flush()
        except Exception as e:
            sem_node.set('status', 'failed')
            # from pudb import set_trace; set_trace()
            sentence_surf = ' '.join(sentence.xpath('tokens/token/@surf'))
            logging.error('An error occurred: {0}\nSentence: {1}\nTree XML:\n{2}'.format(
                e, sentence_surf,
                etree.tostring(sentence, encoding='utf-8', pretty_print=True).decode('utf-8')))
            # print('x', end='', file=sys.stdout)
            sys.stdout.flush()
        sem_nodes.append(sem_node)
    return [etree.tostring(sem_node) for sem_node in sem_nodes]

def get_tree_indices(sentence, nbest):
    num_ccg_trees = int(sentence.xpath('count(./ccg)'))
    if nbest < 1:
        nbest = num_ccg_trees
    return list(range(1, min(nbest, num_ccg_trees) + 1))

keep_attributes = set(['id', 'child', 'sem', 'type'])
def filter_attributes(tree):
    if 'coq_type' in tree.attrib and 'child' not in tree.attrib:
        sem_type = \
            tree.attrib['coq_type'].lstrip('["Parameter ').rstrip('."]')
        if sem_type:
            tree.attrib['type'] = sem_type
    attrib_to_delete = [a for a in tree.attrib.keys() if a not in keep_attributes]
    for a in attrib_to_delete:
        del tree.attrib[a]
    for child in tree:
        filter_attributes(child)
    return

def serialize_tree(tree):
    tree_str = etree.tostring(
        tree, xml_declaration=True, encoding='utf-8', pretty_print=True)
    return tree_str

#  Added to be called like a library function.
def semparse(ccg_derivation: etree.ElementTree, templates: str):
    """ Wrapper function for main functionality of semparse.py """

    global SEMANTIC_INDEX
    global SENTENCES
    global ARGS

    class args():
        def __init__(self):
            self.ncores = 1
            self.gold_trees = True
            self.nbest = 0

    ARGS = args()
    SEMANTIC_INDEX = SemanticIndex(templates)
    SENTENCES = ccg_derivation.findall('.//sentence')

    sentence_inds = range(len(SENTENCES))
    sem_nodes_lists = semantic_parse_sentences(sentence_inds)
    assert len(sem_nodes_lists) == len(SENTENCES), \
        'Element mismatch: {0} vs {1}'.format(len(sem_nodes_lists), len(SENTENCES))

    for sentence, sem_nodes in zip(SENTENCES, sem_nodes_lists):
        sentence.extend(sem_nodes)

    return ccg_derivation

if __name__ == '__main__':
    main()
