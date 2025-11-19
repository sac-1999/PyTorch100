🧠 100 PyTorch Practice Tasks
🔹 1. Fundamentals & Tensor Operations (10)
Implement matrix multiplication manually using PyTorch tensors (no torch.mm).

Write a function to normalize tensors (min-max scaling).

Create a custom tensor broadcasting example.

Implement element-wise operations without built-in functions.

Compute gradients manually using autograd.

Write a script to compute cosine similarity between two tensors.

Implement tensor reshaping and flattening operations.

Build a one-hot encoding function using tensors.

Implement tensor slicing for 3D tensors.

Write a script to measure tensor operation speed on CPU vs GPU.

🔹 2. Custom Autograd & Backpropagation (10)
Build a custom autograd function for ReLU.

Implement backpropagation for linear regression without nn.Module.

Write a custom loss function (Huber loss).

Create a custom optimizer (SGD with momentum).

Implement gradient clipping manually.

Build a custom autograd function for softmax.

Implement cross-entropy loss manually.

Write a script to visualize gradient flow in a network.

Implement weight decay manually.

Compare training with and without gradient accumulation.

🔹 3. Neural Network Basics (10)
Build a feedforward neural network from scratch using nn.Module.

Train a network on MNIST digits without torchvision.

Implement dropout manually.

Compare training with and without batch normalization.

Visualize weight updates during training.

Implement early stopping manually.

Build a multilayer perceptron for tabular data.

Train a model with different activation functions (ReLU, Tanh, Sigmoid).

Implement learning rate scheduling manually.

Write a script to log training metrics with TensorBoard.

🔹 4. CNNs (Computer Vision) (10)
Implement a CNN for CIFAR-10 classification.

Write a custom convolution layer (no nn.Conv2d).

Build a ResNet-like skip connection manually.

Train a CNN with data augmentation.

Implement Grad-CAM visualization.

Build a CNN for fashion-MNIST dataset.

Implement depthwise separable convolution.

Compare CNN performance with and without pooling.

Implement dilated convolution manually.

Train a CNN with transfer learning from pretrained models.

🔹 5. RNNs & Sequence Models (10)
Build a character-level RNN for text generation.

Implement LSTM manually (no nn.LSTM).

Train a sentiment analysis model on IMDB dataset.

Build a sequence-to-sequence model for translation.

Implement attention mechanism for RNNs.

Train a model for next-word prediction.

Implement GRU manually.

Build a time-series forecasting model.

Train a model for named entity recognition.

Implement beam search for sequence generation.

🔹 6. Transformers (10)
Implement scaled dot-product attention from scratch.

Build a mini Transformer encoder for text classification.

Train a Transformer on news headlines.

Implement positional encoding manually.

Compare Transformer vs LSTM on sequence tasks.

Build a Transformer decoder for text generation.

Implement multi-head attention manually.

Train a Transformer for machine translation.

Implement masked language modeling (like BERT).

Build a Vision Transformer (ViT) for CIFAR-100.

🔹 7. GANs (Generative Models) (10)
Implement a vanilla GAN for MNIST.

Build a DCGAN for CIFAR-10.

Train a conditional GAN (cGAN).

Implement Wasserstein GAN with gradient penalty.

Build a CycleGAN for image-to-image translation.

Implement Pix2Pix GAN.

Train a GAN for super-resolution.

Build a StyleGAN-like architecture.

Implement progressive growing GAN.

Compare GAN vs VAE for image generation.

🔹 8. Advanced Topics (10)
Implement mixed precision training with torch.cuda.amp.

Train a model with distributed data parallel (DDP).

Write a script for model quantization.

Implement pruning on a CNN.

Build a reinforcement learning agent with policy gradients.

Implement actor-critic reinforcement learning.

Train a model with curriculum learning.

Implement knowledge distillation between models.

Build a meta-learning model (MAML).

Implement contrastive learning (SimCLR).

🔹 9. Deployment & Optimization (10)
Export a PyTorch model to ONNX format.

Optimize inference with TorchScript.

Deploy a PyTorch model with FastAPI.

Implement model checkpointing and resuming training.

Write a script to monitor GPU usage during training.

Build a REST API for serving PyTorch models.

Deploy a model with Flask.

Implement quantization-aware training.

Optimize inference with TensorRT.

Deploy a PyTorch model on mobile.

🔹 10. Research-Level Challenges (10)
Train a Vision Transformer (ViT) on CIFAR-100.

Build a BERT-like model for masked language modeling.

Implement diffusion models for image generation.

Reproduce ResNet results using PyTorch.

Reproduce GPT-2 results using PyTorch.

Implement reinforcement learning with PPO.

Train a graph neural network (GNN).

Implement Deep Q-Learning.

Build a neural style transfer model.

Implement self-supervised learning with BYOL.

✅ Summary
That’s 100 PyTorch practice tasks, covering:

Fundamentals

Autograd

Neural networks

CNNs

RNNs

Transformers

GANs

Advanced topics

Deployment

Research-level challenges



1. Low-level tensors and autograd
Core

Description: Understand how tensors store data (contiguous vs non-contiguous), how strides and views affect memory layout, and how broadcasting rules apply. Learn gradient accumulation and how hooks can intercept gradients.

Deliverables:

Implement a Tensor class with stride-based slicing.

Write a gradient hook that logs and modifies gradients during backprop.

Demonstrate broadcasting by implementing elementwise ops manually.

Deep

Description: Build custom autograd functions with torch.autograd.Function. Explore higher-order gradients, vector-Jacobian products (vjp), Jacobian-vector products (jvp), and graph checkpointing for memory efficiency.

Deliverables:

Implement a custom backward pass for a new activation function.

Build a mini autograd engine supporting vjp/jvp.

Add checkpointing to reduce memory usage in a deep network.

2. Numerical optimization
Core

Description: Learn how optimizers update parameters. Implement SGD, Momentum, Adam, and RMSProp from scratch. Understand stability tricks like epsilon and bias correction.

Deliverables:

Train a CNN on CIFAR-10 using your own Adam implementation.

Compare convergence curves of SGD vs Adam.

Deep

Description: Master mixed precision training (AMP), dynamic loss scaling to prevent underflow, and gradient clipping/normalization to stabilize training.

Deliverables:

Implement AMP manually with autocast rules.

Add dynamic loss scaling to prevent NaNs.

Train a Transformer with gradient clipping and show improved stability.

3. GPU and kernels
Core

Description: Learn CUDA streams, events, pinned memory, async transfers, and occupancy basics to maximize GPU utilization.

Deliverables:

Build a dataloader with pinned memory and async transfers.

Benchmark GPU utilization with and without streams.

Deep

Description: Write custom C++/CUDA extensions, optimize with tiling, shared memory, warp-level operations, and kernel fusion.

Deliverables:

Implement a fused LayerNorm + Dropout CUDA kernel.

Optimize a reduction kernel with shared memory.

Benchmark against PyTorch’s native ops.

4. Performance engineering
Core

Description: Profile models using torch.profiler and NVTX. Learn microbenchmarking and avoid Python overhead by vectorizing operations.

Deliverables:

Profile ResNet-18 training and identify top 3 bottlenecks.

Replace Python loops with vectorized tensor ops.

Deep

Description: Understand PyTorch’s memory allocator, use CUDA graphs for inference, fuse operators, and experiment with Triton or custom codegen.

Deliverables:

Implement inference with CUDA graphs and measure latency reduction.

Write a Triton kernel for matrix multiplication.

Compare allocator fragmentation with different batch sizes.

5. Distributed systems
Core

Description: Learn DistributedDataParallel (DDP), gradient buckets, allreduce, and communication overlap for multi-GPU training.

Deliverables:

Train ResNet-50 on 4 GPUs with DDP.

Tune bucket sizes for optimal communication overlap.

Deep

Description: Implement ZeRO-style optimizer sharding, pipeline parallelism, tensor parallelism, and checkpointing for large models.

Deliverables:

Train a 300M parameter model with optimizer sharding.

Implement pipeline parallelism for a Transformer.

Add activation checkpointing to reduce memory usage.

6. Data, I/O, and reproducibility
Core

Description: Build high-throughput data pipelines with prefetching and pinned buffers to keep GPUs busy.

Deliverables:

Implement a dataloader that achieves ≥90% GPU utilization.

Compare throughput with and without prefetching.

Deep

Description: Ensure determinism across runs, track variance in experiments, and build scaffolding for reproducible ML research.

Deliverables:

Run experiments with fixed seeds and verify reproducibility.

Build a variance tracker that logs accuracy fluctuations across runs.

Create a reproducibility checklist for large-scale experiments.

🎯 How to use this map
Treat each Core as a foundation project.

Treat each Deep as a capstone project with benchmarks.

Document every deliverable with code, benchmarks, and lessons learned.
