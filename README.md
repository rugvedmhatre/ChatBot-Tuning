# ChatBot-Tuning
This repository contains code for a ChatBot trained using parameters obtained from a Weights &amp; Biases (W&amp;B) parameter sweep. The model is then profiled using PyTorch Profiler to analyze performance bottlenecks. Additionally, an optimized TorchScript version is created for efficient deployment.

---

We train a simple chatbot using movie scripts from the [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) based on the [PyTorch Chatbot Tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html).

This tutorial allows us to train recurrent sequence-to-sequence model,

- Handle loading and pre-processing of the Cornell Movie-Dialogs Corpus dataset
- Implement a sequence-to-sequence model with [Luong attention mechanism](https://arxiv.org/abs/1508.04025)
- Jointly train encoder and decoder models using mini-batches
- Implement greedy-search decoding module
- Interact with the trained chatbot

We use Weights &amp; Biases (W&amp;B) to run a [hyperparameter sweep](https://www.youtube.com/watch?v=9zrmUIlScdY).

We create a sweep configuration using the using the W&B Random Search strategy for the following hyperparameters:

![sweep configuration](https://github.com/rugvedmhatre/ChatBot-Tuning/blob/main/images/sweep_config.png?raw=true)

### Results 

Weights &amp; Biases Project - [https://wandb.ai/nyu-hpml/hpml-chatbot/workspace?nw=nwuserrrm9598](https://wandb.ai/nyu-hpml/hpml-chatbot/workspace?nw=nwuserrrm9598)

![sweep loss](https://github.com/rugvedmhatre/ChatBot-Tuning/blob/main/images/sweep_loss.png?raw=true)

The above figure shows the W&amp;B Hyper-parameter Sweep Output - Losses for 10 runs with lowest losses.

![sweep loss all runs](https://github.com/rugvedmhatre/ChatBot-Tuning/blob/main/images/sweep_all_loss.png?raw=true)

The above figure shows the W&amp;B Hyper-parameter Sweep Output - Losses for all runs.

![parameter evaluation](https://github.com/rugvedmhatre/ChatBot-Tuning/blob/main/images/parameter_evaluation.png?raw=true)

The above figure shows W&amp;B Hyper-parameter Evaluation. The best configuration is the one with the lowest loss. The lowest loss is 2.01315 for the configuration:

- `clip` = 50
- `decoder_learning_ratio` = 3
- `learning_rate` = 0.0005
- `optimizer`= `adam`
- `teacher_forcing_ratio` = 1

![parameter importance](https://github.com/rugvedmhatre/ChatBot-Tuning/blob/main/images/parameter_importance.png?raw=true)

The parameter importance chart (above figure) highlights the importance of each hyper-parameter with the loss of the model. From the chart, we see that `clip` has the highest importance, and it also has strong negative correlation, so if we choose a high value for `clip`, we are able the achieve a low loss. Secondly, `teacher_forcing_ratio` is the next important hyper-parameter, and it has a mildly negative correlation, which results in low loss when the the teacher forcing ratio is high. Learning rate, `decoder_learning_ratio` and the choice of optimizer have comparatively low importance than the previous two hyper-parameters. This can also be actually analyzed better by running more runs or running an exhaustive hyper-parameter sweep like a Grid Search.

---

We then use the Pytorch profiler to measure the time and memory consumption of the
selected model’s operators. We also use the Pytotch profiler to examine the sequence of profiled operators and CUDA kernels in Chrome trace viewer (`chrome://tracing`). ([PyTorch Profiler Example](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html))

### Results

PyTorch Time and Memory Profiler outputs are printed in the Python Notebook. In summary, CPU
execution time is 39.582 ms, GPU execution time is 4.342 ms for model inference for the model with the best configuration.

![chrome trace view](https://github.com/rugvedmhatre/ChatBot-Tuning/blob/main/images/chrome_tracing.png?raw=true)

The above figure shows Chrome Trace View of the PyTorch Profiler.

---

We convert the trained sequence-to-sequence model to TorchScript using the TorchScript API. This module has two core modalities for converting an eager-mode model to a TorchScript graph
representation: tracing and scripting. ([PyTorch Torchscript Tutorial](https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html))

### Differences between tracing and scripting and how they are used in TorchScript

Tracing works well for straightforward modules and functions with static control flow, such as standard convolutional neural networks. It records the operations executed during a forward pass with a given example input, creating a computation graph. However, if the function contains data-dependent control flow—such as if statements or loops that vary based on input values—tracing will only capture the operations along the specific path taken by that example input. This means the control flow itself is not recorded, which can lead to incorrect behavior for different inputs.

To handle modules and functions with dynamic control flow, TorchScript provides a scripting mechanism. Scripting compiles the entire model, including any branching logic, loops, or conditions, ensuring it works correctly for all input scenarios. While tracing is faster to generate and efficient for models with fixed operations, scripting is essential for more complex architectures where behavior changes based on input data, such as transformers.

### Changes needed in the ChatBot model to allow for scripting

To support scripting, several changes are required in the `GreedySearchDecoder` class. First, we need to add `decoder_n_layers` as a constructor argument since the encoder and decoder models will be instances of `TracedModule`, which does not allow access to attributes like `decoder.n_layers`. Passing this value during module construction ensures it is available when needed. Additionally, because scripting restricts the use of global variables within the `forward` method, all relevant values (such as `device` and `SOS_token`) must be stored as attributes in the constructor. These attributes should be added to a special list called `__constants__` to treat them as literal values when constructing the computation graph. Furthermore, TorchScript assumes all function parameters are tensors by default, so it is necessary to use type annotations for any non-tensor arguments. Another important change involves the initialization of `decoder_input`. In the original implementation, we used `torch.LongTensor([[SOS_token]])`, but scripting prohibits such literal initialization. Instead, we use a function like `torch.ones` to create the tensor and multiply it by the
stored `self._SOS_token` value to replicate the original behavior.

### Comparison of evaluation latency of TorchScripted model and regular PyTorch model on CPU and GPU

![latency](https://github.com/rugvedmhatre/ChatBot-Tuning/blob/main/images/latency.png?raw=true)

The table above shows the average PyTorch latency and average TorchScript latency for a batch of 64 data points (run for 100 iterations). The TorchScript model is clearly more optimized than the PyTorch one.