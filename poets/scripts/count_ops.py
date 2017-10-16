#!/usr/bin/python
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sys
import tensorflow as tf

def load_graph(file_name):
  with open(file_name,'rb') as f:
    content = f.read()
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(content)
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')
  return graph

def count_ops(file_name, op_name = None):
  graph = load_graph(file_name)

  if op_name is None:
    return len(graph.get_operations())
  else:
    return sum(1 for op in graph.get_operations() 
               if op.name == op_name)

if __name__ == "__main__":
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  print(count_ops(*sys.argv[1:]))

