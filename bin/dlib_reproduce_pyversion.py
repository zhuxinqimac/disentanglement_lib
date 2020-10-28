# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pipeline to reproduce fixed models and evaluation protocols.

This is the main pipeline for the reasoning step in the paper:
Are Disentangled Representations Helpful for Abstract Visual Reasoning?
Sjoerd van Steenkiste, Francesco Locatello, Juergen Schmidhuber, Olivier Bachem.
NeurIPS, 2019.
https://arxiv.org/abs/1905.12506
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from disentanglement_lib.config import reproduce
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.visualize import visualize_model
import argparse
import numpy as np
import tensorflow.compat.v1 as tf

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
  parser = argparse.ArgumentParser(description='Project description.')
  parser.add_argument('--study', help='Name of the study.', type=str, default='unsupervised_study_v1')
  parser.add_argument('--output_directory', help='Output directory of experiments.', type=str, default=None)
  parser.add_argument('--model_dir', help='Directory to take trained model from.', type=str, default=None)
  parser.add_argument('--model_num', help='Integer with model number to train.', type=int, default=None)
  parser.add_argument('--only_print', help='Whether to only print the hyperparameter settings.', type=_str_to_bool, default=False)
  parser.add_argument('--overwrite', help='Whether to overwrite output directory.', type=_str_to_bool, default=False)
  args = parser.parse_args()
  # logging.set_verbosity('error')
  # logging.set_stderrthreshold('error')
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

  # Obtain the study to reproduce.
  study = reproduce.STUDIES[args.study]

  # Print the hyperparameter settings.
  if args.model_dir is None:
    study.print_model_config(args.model_num)
  else:
    print("Model directory (skipped training):")
    print("--")
    print(args.model_dir)
  print()
  study.print_postprocess_config()
  print()
  study.print_eval_config()
  if args.only_print:
    return

  # Set correct output directory.
  if args.output_directory is None:
    if args.model_dir is None:
      output_directory = os.path.join("output", "{study}", "{model_num}")
    else:
      output_directory = "output"
  else:
    output_directory = args.output_directory

  # Insert model number and study name into path if necessary.
  output_directory = output_directory.format(model_num=str(args.model_num),
                                             study=str(args.study))

  # Model training (if model directory is not provided).
  if args.model_dir is None:
    model_bindings, model_config_file = study.get_model_config(args.model_num)
    print("Training model...")
    model_dir = os.path.join(output_directory, "model")
    model_bindings = [
        "model.name = '{}'".format(os.path.basename(model_config_file)).replace(
            ".gin", ""),
        "model.model_num = {}".format(args.model_num),
    ] + model_bindings
    train.train_with_gin(model_dir, args.overwrite, [model_config_file],
                         model_bindings)
  else:
    print("Skipped training...")
    model_dir = args.model_dir

  # We visualize reconstructions, samples and latent space traversals.
  visualize_dir = os.path.join(output_directory, "visualizations")
  visualize_model.visualize(model_dir, visualize_dir, args.overwrite)

  # We fix the random seed for the postprocessing and evaluation steps (each
  # config gets a different but reproducible seed derived from a master seed of
  # 0). The model seed was set via the gin bindings and configs of the study.
  random_state = np.random.RandomState(0)

  # We extract the different representations and save them to disk.
  postprocess_config_files = sorted(study.get_postprocess_config_files())
  for config in postprocess_config_files:
    post_name = os.path.basename(config).replace(".gin", "")
    print("Extracting representation %s..." % post_name)
    post_dir = os.path.join(output_directory, "postprocessed", post_name)
    postprocess_bindings = [
        "postprocess.random_seed = {}".format(random_state.randint(2**32)),
        "postprocess.name = '{}'".format(post_name)
    ]
    postprocess.postprocess_with_gin(model_dir, post_dir, args.overwrite,
                                     [config], postprocess_bindings)

  # Iterate through the disentanglement metrics.
  eval_configs = sorted(study.get_eval_config_files())
  blacklist = ['downstream_task_logistic_regression.gin']
  for config in postprocess_config_files:
    post_name = os.path.basename(config).replace(".gin", "")
    post_dir = os.path.join(output_directory, "postprocessed",
                            post_name)
    # Now, we compute all the specified scores.
    for gin_eval_config in eval_configs:
      if os.path.basename(gin_eval_config) not in blacklist:
        metric_name = os.path.basename(gin_eval_config).replace(".gin", "")
        print("Computing metric '%s' on '%s'..." % (metric_name, post_name))
        metric_dir = os.path.join(output_directory, "metrics", post_name,
                                  metric_name)
        eval_bindings = [
            "evaluation.random_seed = {}".format(random_state.randint(2**32)),
            "evaluation.name = '{}'".format(metric_name)
        ]
        evaluate.evaluate_with_gin(post_dir, metric_dir, args.overwrite,
                                   [gin_eval_config], eval_bindings)


if __name__ == "__main__":
    main()
