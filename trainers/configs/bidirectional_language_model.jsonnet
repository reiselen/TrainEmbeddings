// From https://github.com/allenai/allennlp-models/blob/main/training_config/lm/bidirectional_language_model.jsonnet
local NUM_GPUS = 3;
local NUM_GRAD_ACC = 4;
local BATCH_SIZE = 32;

local BASE_READER = {
        "type": "allennlp_models.lm.dataset_readers.simple_language_modeling.SimpleLanguageModelingDatasetReader",
        "tokenizer": {
	        // The 1 Billion Word Language Model Benchmark dataset is
	        // pre-tokenized. (Also, if you're running against a untokenized
	        // dataset be aware that there are serialization issues with Spacy.
	        // These come into play in the multiprocess case.)
          "type": "just_spaces"
        },
        "token_indexers": {
          "tokens": {
            "type": "single_id"
          },
          "token_characters": {
            "type": "elmo_characters"
          }
        },
        "max_sequence_length": 400,
        "start_tokens": ["<S>"],
        "end_tokens": ["</S>"],
};

local BASE_LOADER = {
  "max_instances_in_memory": BATCH_SIZE * 100,
  "batch_sampler": {
    "type": "bucket",
    "batch_size": BATCH_SIZE,
  }
};

{
  "dataset_reader": {
    "type": "sharded",
    "base_reader": BASE_READER,
  },
  // Note: We don't set a validation_data_path because the softmax is only
  // sampled during training. Not sampling on GPUs results in a certain OOM
  // given our large vocabulary. We'll need to evaluate against the test set
  // (when we'll want a full softmax) with the CPU.
  "train_data_path": std.extVar("BIDIRECTIONAL_LM_TRAIN_PATH"),
  //"validation_data_path": std.extVar("BIDIRECTIONAL_LM_VALIDATION_PATH"),
  "vocabulary": {
      // Use a prespecified vocabulary for efficiency.
      //"type": "from_files",
      //"directory": std.extVar("BIDIRECTIONAL_LM_VOCAB_PATH"),
      // Plausible config for generating the vocabulary.
      "tokens_to_add": {
           "tokens": ["<S>", "</S>"],
           "token_characters": ["<>/S"]
      },
      "min_count": {"tokens": 2}
  },
  "model": {
    "type": "language_model",
    "bidirectional": true,
    "num_samples": 8192,
    # Sparse embeddings don't work with DistributedDataParallel.
    "sparse_embeddings": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "empty"
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "num_embeddings": 262,
                // Same as the Transformer ELMo in Calypso. Matt reports that
                // this matches the original LSTM ELMo as well.
                "embedding_dim": 16
            },
            "encoder": {
                "type": "cnn-highway",
                "activation": "relu",
                "embedding_dim": 16,
                "filters": [
                    [1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 1024]],
                "num_highway": 2,
                "projection_dim": 512,
                "projection_location": "after_highway",
                "do_layer_norm": true
            }
        }
      }
    },
    // TODO(brendanr): Consider the following.
    // remove_bos_eos: true,
    // Applies to the contextualized embeddings.
    "dropout": 0.1,
    "contextualizer": {
        "type": "bidirectional_language_model_transformer",
        "input_dim": 512,
        "hidden_dim": 2048,
        "num_layers": 6,
        "dropout": 0.1,
        "input_dropout": 0.1
    }
  },
  "data_loader": BASE_LOADER,
  "distributed": {
    "cuda_devices": if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
  },
  "trainer": {
    "num_epochs": 40,
    // If training with CPU, add back
    // "cuda_device": -1, 
    //"checkpointer": {
      // Number of intermediate models to keep
    //  "num_serialized_models_to_keep": 1 // optionally keep one for each epoch (num_epochs)
    //},
    "optimizer": {
      // The gradient accumulators in Adam for the running stdev and mean for
      // words not used in the sampled softmax would be decayed to zero with the
      // standard "adam" optimizer.
      "type": "dense_sparse_adam"
    },
    // TODO(brendanr): Needed with transformer too?
    // "grad_norm": 10.0,
    "learning_rate_scheduler": {
      "type": "noam",
      // See https://github.com/allenai/calypso/blob/master/calypso/train.py#L401
      "model_size": 512,
      // See https://github.com/allenai/calypso/blob/master/bin/train_transformer_lm1b.py#L51.
      // Adjusted based on our sample size relative to Calypso's.
      "warmup_steps": 6000
    },
    "num_gradient_accumulation_steps": NUM_GRAD_ACC,
    // Set True when training on CUDA
    "use_amp": true
  }
}
