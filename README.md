# Awesome-Omnimodal-Embeddings

## Table of Contents

1. [Text Embedding](#1-text-embedding)

   1.1 [Embedding Models](#11-embedding-models)

   - [Word Vectors](#word-vectors)
   - [BERT-Based Embedding](#bert-based-embedding)
   - [LLM-Based Embedding](#llm-based-embedding)
   - [Instruction-Following Embedding](#instruction-following-embedding)
   - [Training Methods](#training-methods)
   - [Training-Free Methods](#training-free-methods)

   <!-- prettier ignore -->

   1.2 [Embedding Benchmarks](#12-embedding-benchmarks)

   1.3 [Embedding Analysis](#13-embedding-analysis)

   - [Interpretability](#interpretability)
   - [Attack](#attack)
   - [Bias](#bias)
   - [Cross-Lingual Alignment](#cross-lingual-alignment)
   - [Others](#others)

   <!-- prettier ignore -->

   1.4 [Embedding Applications](#14-embedding-applications)

   - [Domain-Specific Embedding](#domain-specific-embedding)
   - [Retrieval](#retrieval)
   - [Reranking](#reranking)
   - [Classification & Clustering & STS](#classification--clustering--sts)
   - [Sentiment Analysis](#sentiment-analysis)
   - [Discourse Analysis & Topic Modeling](#discourse-analysis--topic-modeling)
   - [LLM Self-Evaluation](#llm-self-evaluation)
   - [Regression](#regression)

2. [Code Embedding](#2-code-embedding)

   2.1 [Models](#21-models)

   2.2 [Datasets & Benchmarks](#22-datasets--benchmarks)

   2.3 [Applications](#23-applications)

   - [Retrieval](#retrieval-1)
   - [Reranking](#reranking-1)
   - [Others](#others-1)

3. [Vision Embedding](#3-vision-embedding)

   3.1 [Models](#31-models)

   - [General Representation Models](#general-representation-models)
   - [3D Representation](#3d-representation)
   - [Video Representation](#video-representation)
   - [Vision-Language Model](#vision-language-model)
   - [Domain-Specific Model](#domain-specific-model)

   <!-- prettier ignore -->

   3.2 [Training Methods](#32-training-methods)

   - [Contrastive & Self-Supervised Training](#contrastive--self-supervised-training)
   - [Disentanglement & Causality & Invariance](#disentanglement--causality--invariance)
   - [Robustness, Privacy, Federated & Efficient Training](#robustness-privacy-federated--efficient-training)
   - [Semi, Weak, Few-Shot Supervision](#semi-weak-few-shot-supervision)
   - [Multimodal Training](#multimodal-training)

   <!-- prettier ignore -->

   3.3 [Datasets & Benchmarks](#33-datasets--benchmarks)

   3.4 [Embedding Analysis](#34-embedding-analysis)

   - [Representation Properties & Mechanisms](#representation-properties--mechanisms)
   - [Robustness & Human Alignment](#robustness--human-alignment)
   - [Interpretability](#interpretability-1)

   <!-- prettier ignore -->

   3.5 [Embedding Applications](#35-embedding-applications)

   - [Segmentation & Detection](#segmentation--detection)
   - [Retrieval & Reranking](#retrieval--reranking)
   - [Identification & Surveillance](#identification--surveillance)
   - [Autonomous Driving](#autonomous-driving)
   - [Enhancement & Generation & Editing](#enhancement--generation--editing)
   - [Others](#others-2)

4. [Audio Embedding](#4-audio-embedding)

   4.1 [Models](#41-models)

   - [Speaker Embedding](#speaker-embedding)
   - [General Speech Representation](#general-speech-representation)
   - [Speech Content, Phoneme & Articulatory Representation](#speech-content-phoneme--articulatory-representation)
   - [Emotion, Paralinguistic & Prosody Embedding](#emotion-paralinguistic--prosody-embedding)
   - [Multilingual Embedding](#multilingual-embedding)
   - [Multimodal](#multimodal)

   <!-- prettier ignore -->

   4.2 [Training Methods](#42-training-methods)

   - [Disentanglement & Decoupling](#disentanglement--decoupling)
   - [Contrastive, Generative & Multi-Objective Learning](#contrastive-generative--multi-objective-learning)
   - [Distillation, Pruning & Efficiency-Oriented Training](#distillation-pruning--efficiency-oriented-training)
   - [Multilingual Training](#multilingual-training)
   - [Privacy-Preserving Representation Learning](#privacy-preserving-representation-learning)
   - [Robustness-Oriented Training](#robustness-oriented-training)
   - [Other Methods](#other-methods)

   <!-- prettier ignore -->

   4.3 [Embedding Analysis](#43-embedding-analysis)

   - [Representation Properties](#representation-properties)
   - [Speaker Characteristics](#speaker-characteristics)
   - [Benchmarking & Evaluating Extractors](#benchmarking--evaluating-extractors)
   - [Interpretability](#interpretability-2)
   - [Cross-Domain Generalization](#cross-domain-generalization)

   <!-- prettier ignore -->

   4.4 [Benchmarks & Toolkits](#44-benchmarks--toolkits)

   4.5 [Embedding Applications](#45-embedding-applications)

   - [Speaker-Related Application](#speaker-related-application)
   - [Speech Recognition & Transcription](#speech-recognition--transcription)
   - [Speech Emotion, Paralinguistic, Health & Cognitive Applications](#speech-emotion-paralinguistic-health--cognitive-applications)
   - [Text-To-Speech](#text-to-speech)
   - [Spoofing & Security](#spoofing--security)
   - [Others](#others-3)

5. [Graph Embedding](#5-graph-embedding)

   5.1 [Node Embedding](#51-node-embedding)

   5.2 [Graph Embedding](#52-graph-embedding)

   5.3 [Edge Embedding](#53-edge-embedding)

   5.4 [Knowledge Graph Embedding](#54-knowledge-graph-embedding)

6. [Time Series Embedding](#6-time-series-embedding)

   6.1 [Foundation Models](#61-foundation-models)

   6.2 [Model Architecture & Training Methods](#62-model-architecture--training-methods)

   6.3 [Temporal Knowledge Graph](#63-temporal-knowledge-graph)

   6.4 [Temporal Network](#64-temporal-networks)

## 1. Text Embedding

### 1.1 Embedding Models

#### Word Vectors

- "Efficient Estimation of Word Representations in Vector Space" [2013-01] [[paper](https://arxiv.org/abs/1301.3781)]

- "Distributed Representations of Words and Phrases and their Compositionality" [2013-10] [NeurIPS 2013] [[paper](https://arxiv.org/abs/1310.4546)]

- "GloVe: Global Vectors for Word Representation" [2014-10] [EMNLP 2014] [[paper](https://aclanthology.org/D14-1162/)]

- "Axis Tour: Word Tour Determines the Order of Axes in ICA-transformed Embeddings" [2024-01] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2401.06112)]

- "The Shape of Word Embeddings: Quantifying Non-Isometry with Topological Data Analysis" [2024-04] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2404.00500)]

- "Statistical Uncertainty in Word Embeddings: GloVe-V" [2024-06] [EMNLP 2024] [[paper](https://arxiv.org/abs/2406.12165)]

- "Exploring Intra and Inter-language Consistency in Embeddings with ICA" [2024-06] [EMNLP 2024] [[paper](https://arxiv.org/abs/2406.12474)]

- "GrEmLIn: A Repository of Green Baseline Embeddings for 87 Low-Resource Languages Injected with Multilingual Graph Knowledge" [2024-09] [NAACL 2025 Findings] [[paper](https://arxiv.org/abs/2409.18193)]

- "Understanding Higher-Order Correlations Among Semantic Components in Embeddings" [2024-09] [EMNLP 2024] [[paper](https://arxiv.org/abs/2409.19919)]

#### BERT-Based Embedding

- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" [2018-10] [NAACL 2019] [[paper](https://arxiv.org/abs/1810.04805)]

- "Cross-lingual Language Model Pretraining" [2019-01] [NeurIPS 2019] [[paper](https://arxiv.org/abs/1901.07291)]

- "ERNIE: Enhanced Representation through Knowledge Integration" [2019-04] [[paper](https://arxiv.org/abs/1904.09223)]

- "Unified Language Model Pre-training for Natural Language Understanding and Generation" [2019-05] [NeurIPS 2019] [[paper](https://arxiv.org/abs/1905.03197)]

- "SpanBERT: Improving Pre-training by Representing and Predicting Spans" [2019-07] [TACL 2020] [[paper](https://arxiv.org/abs/1907.10529)]

- "RoBERTa: A Robustly Optimized BERT Pretraining Approach" [2019-07] [[paper](https://arxiv.org/abs/1907.11692)]

- "ERNIE 2.0: A Continual Pre-training Framework for Language Understanding" [2019-07] [AAAI 2020] [[paper](https://arxiv.org/abs/1907.12412)]

- "StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding" [2019-08] [ICLR 2020] [[paper](https://arxiv.org/abs/1908.04577)]

- "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" [2019-08] [EMNLP 2019] [[paper](https://arxiv.org/abs/1908.10084)]

- "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" [2019-09] [ICLR 2020] [[paper](https://arxiv.org/abs/1909.11942)]

- "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter" [2019-10] [[paper](https://arxiv.org/abs/1910.01108)]

- "Unsupervised Cross-lingual Representation Learning at Scale" [2019-11] [ACL 2020] [[paper](https://arxiv.org/abs/1911.02116)]

- "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers" [2020-02] [NeurIPS 2020] [[paper](https://arxiv.org/abs/2002.10957)]

- "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators" [2020-03] [ICLR 2020] [[paper](https://arxiv.org/abs/2003.10555)]

- "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT" [2020-04] [SIGIR 2020] [[paper](https://arxiv.org/abs/2004.12832)]

- "DeBERTa: Decoding-enhanced BERT with Disentangled Attention" [2020-06] [ICLR 2021] [[paper](https://arxiv.org/abs/2006.03654)]

- "MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers" [2020-12] [ACL 2021 Findings] [[paper](https://arxiv.org/abs/2012.15828)]

- "CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation" [2021-03] [TACL 2022] [[paper](https://arxiv.org/abs/2103.06874)]

- "SimCSE: Simple Contrastive Learning of Sentence Embeddings" [2021-04] [EMNLP 2021] [[paper](https://arxiv.org/abs/2104.08821)]

- "Larger-Scale Transformers for Multilingual Masked Language Modeling" [2021-05] [[paper](https://arxiv.org/abs/2105.00572)]

- "XLM-E: Cross-lingual Language Model Pre-training via ELECTRA" [2021-06] [ACL 2022] [[paper](https://arxiv.org/abs/2106.16138)]

- "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing" [2021-11] [ICLR 2023] [[paper](https://arxiv.org/abs/2111.09543)]

- "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction" [2021-12] [NAACL 2022] [[paper](https://arxiv.org/abs/2112.01488)]

- "BiTimeBERT: Extending Pre-Trained Language Representations with Bi-Temporal Information" [2022-04] [SIGIR 2023] [[paper](https://arxiv.org/abs/2204.13032)]

- "SimLM: Pre-training with Representation Bottleneck for Dense Passage Retrieval" [2022-07] [ACL 2023] [[paper](https://arxiv.org/abs/2207.02578)]

- "Beyond English-Centric Bitexts for Better Multilingual Language Representation Learning" [2022-10] [ACL 2023] [[paper](https://arxiv.org/abs/2210.14867)]

- "Text Embeddings by Weakly-Supervised Contrastive Pre-training" [2022-12] [[paper](https://arxiv.org/abs/2212.03533)]

- "Representation Deficiency in Masked Language Modeling" [2023-02] [ICLR 2024] [[paper](https://arxiv.org/abs/2302.02060)]

- "Dual-Alignment Pre-training for Cross-lingual Sentence Embedding" [2023-05] [ACL 2023] [[paper](https://arxiv.org/abs/2305.09148)]

- "Glot500: Scaling Multilingual Corpora and Language Models to 500 Languages" [2023-05] [ACL 2023] [[paper](https://arxiv.org/abs/2305.12182)]

- "Learning Multilingual Sentence Representations with Cross-lingual Consistency Regularization" [2023-06] [EMNLP 2023 Industry] [[paper](https://arxiv.org/abs/2306.06919)]

- "Towards General Text Embeddings with Multi-stage Contrastive Learning" [2023-08] [[paper](https://arxiv.org/abs/2308.03281)]

- "Augmenting Transformers with Recursively Composed Multi-grained Representations" [2023-09] [ICLR 2024] [[paper](https://arxiv.org/abs/2309.16319)]

- "EELBERT: Tiny Models through Dynamic Embeddings" [2023-10] [EMNLP 2023 Industry] [[paper](https://arxiv.org/abs/2310.20144)]

- "Learning Mutually Informed Representations for Characters and Subwords" [2023-11] [NAACL 2024 Findings] [[paper](https://arxiv.org/abs/2311.07853)]

- "BERT Has More to Offer: BERT Layers Combination Yields Better Sentence Embeddings" [2023-12] [EMNLP 2023 Findings] [[paper](https://aclanthology.org/2023.findings-emnlp.1030/)]

- "Nomic Embed: Training a Reproducible Long Context Text Embedder" [2024-02] [TMLR 2025] [[paper](https://arxiv.org/abs/2402.01613)]

- "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation" [2024-02] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2402.03216)]

- "Multilingual E5 Text Embeddings: A Technical Report" [2024-02] [[paper](https://arxiv.org/abs/2402.05672)]

- "Are ELECTRA’s Sentence Embeddings Beyond Repair? The Case of Semantic Textual Similarity" [2024-02] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2402.13130)]

- "NextLevelBERT: Masked Language Modeling with Higher-Level Representations for Long Documents" [2024-02] [ACL 2024] [[paper](https://arxiv.org/abs/2402.17682)]

- "Arctic-Embed: Scalable, Efficient, and Accurate Text Embedding Models" [2024-05] [[paper](https://arxiv.org/abs/2405.05374)]

- "Subword Attention and Post-Processing for Rare and Unknown Contextualized Embeddings" [2024-06] [NAACL 2024 Findings] [[paper](https://aclanthology.org/2024.findings-naacl.88/)]

- "mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval" [2024-07] [EMNLP 2024 Industry] [[paper](https://arxiv.org/abs/2407.19669)]

- "MEXMA: Token-level objectives improve sentence representations" [2024-09] [ACL 2025] [[paper](https://arxiv.org/abs/2409.12737)]

- "Arctic-Embed 2.0: Multilingual Retrieval Without Compromise" [2024-12] [[paper](https://arxiv.org/abs/2412.04506)]

- "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference" [2024-12] [ACL 2025] [[paper](https://arxiv.org/abs/2412.13663)]

- "NeoBERT: A Next-Generation BERT" [2025-02] [TMLR 2025] [[paper](https://arxiv.org/abs/2502.19587)]

- "Beyond instruction-conditioning, MoTE: Mixture of Task Experts for Multi-task Embedding Models" [2025-07] [ACL 2025 Findings] [[paper](https://arxiv.org/abs/2506.17781)]

- "mmBERT: A Modern Multilingual Encoder with Annealed Language Learning" [2025-09] [[paper](https://arxiv.org/abs/2509.06888)]

#### LLM-Based Embedding

- "Text and Code Embeddings by Contrastive Pre-Training" [2022-01] [[paper](https://arxiv.org/abs/2201.10005)]

- "SGPT: GPT Sentence Embeddings for Semantic Search" [2022-02] [[paper](https://arxiv.org/abs/2202.08904)]

- "Scaling Sentence Embeddings with Large Language Models" [2023-07] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2307.16645)]

- "Nugget: Neural Agglomerative Embeddings of Text" [2023-10] [ICML 2023] [[paper](https://arxiv.org/abs/2310.01732)]

- "BeLLM: Backward Dependency Enhanced Large Language Model for Sentence Embeddings" [2023-11] [NAACL 2024] [[paper](https://arxiv.org/abs/2311.05296)]

- "Improving Text Embeddings with Large Language Models" [2024-01] [ACL 2024] [[paper](https://arxiv.org/abs/2401.00368)]

- "Generative Representational Instruction Tuning" [2024-02] [ICLR 2025] [[paper](https://arxiv.org/abs/2402.09906)]

- "Gecko: Versatile Text Embeddings Distilled from Large Language Models" [2024-03] [[paper](https://arxiv.org/abs/2403.20327)]

- "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders" [2024-04] [[paper](https://arxiv.org/abs/2404.05961)]

- "NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models" [2024-05] [ICLR 2025] [[paper](https://arxiv.org/abs/2405.17428)]

- "Llama2Vec: Unsupervised Adaptation of Large Language Models for Dense Retrieval" [2024-08] [ACL 2024] [[paper](https://aclanthology.org/2024.acl-long.191/)]

- "Making Text Embedders Few-Shot Learners" [2024-09] [ICLR 2025] [[paper](https://arxiv.org/abs/2409.15700)]

- "Linq-Embed-Mistral Technical Report" [2024-12] [[paper](https://arxiv.org/abs/2412.03223)]

- "LUSIFER: Language Universal Space Integration for Enhanced Multilingual Embeddings with Large Language Models" [2025-01] [SIGIR 2025] [[paper](https://arxiv.org/abs/2501.00874)]

- "Enhancing Lexicon-Based Text Embeddings with Large Language Models" [2025-01] [ACL 2025] [[paper](https://arxiv.org/abs/2501.09749)]

- "Cramming 1568 Tokens into a Single Vector and Back Again: Exploring the Limits of Embedding Space Capacity" [2025-02] [ACL 2025] [[paper](https://arxiv.org/abs/2502.13063)]

- "Gemini Embedding: Generalizable Embeddings from Gemini" [2025-03] [[paper](https://arxiv.org/abs/2503.07891)]

- "CSE-SFP: Enabling Unsupervised Sentence Representation Learning via a Single Forward Pass" [2025-05] [SIGIR 2025] [[paper](https://arxiv.org/abs/2505.00389)]

- "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models" [2025-06] [[paper](https://arxiv.org/abs/2506.05176)]

- "LGAI-EMBEDDING-Preview Technical Report" [2025-06] [[paper](https://arxiv.org/abs/2506.07438)]

- "Learning to Look at the Other Side: A Semantic Probing Study of Word Embeddings in LLMs with Enabled Bidirectional Attention" [2025-07] [ACL 2025] [[paper](https://arxiv.org/abs/2510.01652)]

- "Training LLMs to be Better Text Embedders through Bidirectional Reconstruction" [2025-09] [EMNLP 2025] [[paper](https://arxiv.org/abs/2509.03020)]

- "Conan-Embedding-v2: Training an LLM from Scratch for Text Embeddings" [2025-09] [[paper](https://arxiv.org/abs/2509.12892)]

- "EmbeddingGemma: Powerful and Lightweight Text Representations" [2025-09] [[paper](https://arxiv.org/abs/2509.20354)]

- "F2LLM Technical Report: Matching SOTA Embedding Performance with 6 Million Open-Source Data" [2025-10] [[paper](https://arxiv.org/abs/2510.02294)]

#### Instruction-Following Embedding

- "One Embedder, Any Task: Instruction-Finetuned Text Embeddings" [2022-12] [ACL 2023 Findings] [[paper](https://arxiv.org/abs/2212.09741)]

- "DATA-CUBE: Data Curriculum for Instruction-based Sentence Representation Learning" [2024-01] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2401.03563)]

- "Answer is All You Need: Instruction-following Text Embedding via Answering the Question" [2024-02] [ACL 2024] [[paper](https://arxiv.org/abs/2402.09642)]

- "Hyper-CL: Conditioning Sentence Representations with Hypernetworks" [2024-03] [ACL 2024] [[paper](https://arxiv.org/abs/2403.09490)]

- "Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models" [2024-09] [ICLR 2025] [[paper](https://arxiv.org/abs/2409.11136)]

- "Varying Sentence Representations via Condition-Specified Routers" [2024-11] [EMNLP 2024] [[paper](https://aclanthology.org/2024.emnlp-main.963/)]

#### Training Methods

- "Ranking-Enhanced Unsupervised Sentence Representation Learning" [2022-09] [ACL 2023] [[paper](https://arxiv.org/abs/2209.04333)]

- "Language Agnostic Multilingual Information Retrieval with Contrastive Learning" [2022-10] [ACL 2023 Findings] [[paper](https://arxiv.org/abs/2210.06633)]

- "miCSE: Mutual Information Contrastive Learning for Low-shot Sentence Embeddings" [2022-11] [ACL 2023] [[paper](https://arxiv.org/abs/2211.04928)]

- "Improving Contrastive Learning of Sentence Embeddings from AI Feedback" [2023-05] [ACL 2023 Findings] [[paper](https://arxiv.org/abs/2305.01918)]

- "StrAE: Autoencoding for Pre-Trained Embeddings using Explicit Structure" [2023-05] [EMNLP 2023] [[paper](https://arxiv.org/abs/2305.05588)]

- "Alleviating Over-smoothing for Unsupervised Sentence Representation" [2023-05] [ACL 2023] [[paper](https://arxiv.org/abs/2305.06154)]

- "Distilling Semantic Concept Embeddings from Contrastively Fine-Tuned Language Models" [2023-05] [SIGIR 2023] [[paper](https://arxiv.org/abs/2305.09785)]

- "SimCSE++: Improving Contrastive Learning for Sentence Embeddings from Two Perspectives" [2023-05] [EMNLP 2023] [[paper](https://arxiv.org/abs/2305.13192)]

- "Bridging Continuous and Discrete Spaces: Interpretable Sentence Representation Learning via Compositional Operations" [2023-05] [EMNLP 2023] [[paper](https://arxiv.org/abs/2305.14599)]

- "Contrastive Learning of Sentence Embeddings from Scratch" [2023-05] [EMNLP 2023] [[paper](https://arxiv.org/abs/2305.15077)]

- "Efficient Document Embeddings via Self-Contrastive Bregman Divergence Learning" [2023-05] [ACL 2023 Findings] [[paper](https://arxiv.org/abs/2305.16031)]

- "RankCSE: Unsupervised Sentence Representations Learning via Learning to Rank" [2023-05] [ACL 2023] [[paper](https://arxiv.org/abs/2305.16726)]

- "WhitenedCSE: Whitening-based Contrastive Learning of Sentence Embeddings" [2023-05] [ACL 2023] [[paper](https://arxiv.org/abs/2305.17746)]

- "On The Inadequacy of Optimizing Alignment and Uniformity in Contrastive Learning of Sentence Representations" [2023-05] [ICLR 2023] [[paper](https://openreview.net/forum?id=MxvHVNukama)]

- "Composition-contrastive Learning for Sentence Embeddings" [2023-07] [ACL 2023] [[paper](https://arxiv.org/abs/2307.07380)]

- "Robustness-Aware Word Embedding Improves Certified Robustness to Adversarial Word Substitutions" [2023-07] [ACL 2023 Findings] [[paper](https://aclanthology.org/2023.findings-acl.42/)]

- "AoE: Angle-optimized Embeddings for Semantic Textual Similarity" [2023-09] [ACL 2024] [[paper](https://arxiv.org/abs/2309.12871)]

- "Improving Contrastive Learning of Sentence Embeddings with Focal InfoNCE" [2023-10] [EMNLP 2023 Findings] [[paper](https://arxiv.org/abs/2310.06918)]

- "HiCL: Hierarchical Contrastive Learning of Unsupervised Sentence Embeddings" [2023-10] [EMNLP 2023 Findings] [[paper](https://arxiv.org/abs/2310.09720)]

- "Large Language Models can Contrastively Refine their Generation for Better Sentence Representation Learning" [2023-10] [NAACL 2024] [[paper](https://arxiv.org/abs/2310.10962)]

- "DistillCSE: Distilled Contrastive Learning for Sentence Embeddings" [2023-10] [EMNLP 2023 Findings] [[paper](https://arxiv.org/abs/2310.13499)]

- "On the Dimensionality of Sentence Embeddings" [2023-10] [EMNLP 2023 Findings] [[paper](https://arxiv.org/abs/2310.15285)]

- "EMMA-X: An EM-like Multilingual Pre-training Algorithm for Cross-lingual Representation Learning" [2023-10] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2310.17233)]

- "Non-contrastive sentence representations via self-supervision" [2023-10] [NAACL 2024 Findings] [[paper](https://arxiv.org/abs/2310.17690)]

- "AdaSent: Efficient Domain-Adapted Sentence Embeddings for Few-Shot Classification" [2023-11] [EMNLP 2023] [[paper](https://arxiv.org/abs/2311.00408)]

- "Sub-Sentence Encoder: Contrastive Learning of Propositional Semantic Representations" [2023-11] [NAACL 2024] [[paper](https://arxiv.org/abs/2311.04335)]

- "Text Representation Distillation via Information Bottleneck Principle" [2023-11] [EMNLP 2023] [[paper](https://arxiv.org/abs/2311.05472)]

- "Sparsity-Preserving Differentially Private Training of Large Embedding Models" [2023-11] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2311.08357)]

- "RobustEmbed: Robust Sentence Embeddings Using Self-Supervised Contrastive Pre-Training" [2023-12] [EMNLP 2023 Findings] [[paper](https://aclanthology.org/2023.findings-emnlp.305/)]

- "DocSplit: Simple Contrastive Pretraining for Large Document Embeddings" [2023-12] [EMNLP 2023 Findings] [[paper](https://aclanthology.org/2023.findings-emnlp.945/)]

- "OssCSE: Overcoming Surface Structure Bias in Contrastive Learning for Unsupervised Sentence Embedding" [2023-12] [EMNLP 2023] [[paper](https://aclanthology.org/2023.emnlp-main.448/)]

- "Landmark Embedding: A Chunking-Free Embedding Method For Retrieval Augmented Long-Context Large Language Models" [2024-02] [ACL 2024] [[paper](https://arxiv.org/abs/2402.11573)]

- "ESE: Espresso Sentence Embeddings" [2024-02] [ICLR 2025] [[paper](https://arxiv.org/abs/2402.14776)]

- "Towards Better Understanding of Contrastive Sentence Representation Learning: A Unified Paradigm for Gradient" [2024-02] [ACL 2024] [[paper](https://arxiv.org/abs/2402.18281)]

- "RobustSentEmbed: Robust Sentence Embeddings Using Adversarial Self-Supervised Contrastive Learning" [2024-03] [NAACL 2024 Findings] [[paper](https://arxiv.org/abs/2403.11082)]

- "Space Decomposition for Sentence Embedding" [2024-06] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2406.03125)]

- "Repurposing Language Models into Embedding Models: Finding the Compute-Optimal Recipe" [2024-06] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2406.04165)]

- "Representation Learning with Conditional Information Flow Maximization" [2024-06] [ACL 2024] [[paper](https://arxiv.org/abs/2406.05510)]

- "SKICSE: Sentence Knowable Information Prompted by LLMs Improves Contrastive Sentence Embeddings" [2024-06] [NAACL 2024] [[paper](https://aclanthology.org/2024.naacl-short.13/)]

- "Banyan: Improved Representation Learning with Explicit Structure" [2024-07] [ICML 2025] [[paper](https://arxiv.org/abs/2407.17771)]

- "Matryoshka-Adaptor: Unsupervised and Supervised Tuning for Smaller Embedding Dimensions" [2024-07] [EMNLP 2024] [[paper](https://arxiv.org/abs/2407.20243)]

- "VAEGPT-Sim: Improving Sentence Representation with Limited Corpus Using Gradually-Denoising VAE" [2024-08] [ACL 2024 Findings] [[paper](https://aclanthology.org/2024.findings-acl.513/)]

- "Enhancing Unsupervised Sentence Embeddings via Knowledge-Driven Data Augmentation and Gaussian-Decayed Contrastive Learning" [2024-09] [ACL 2025] [[paper](https://arxiv.org/abs/2409.12887)]

- "Contextual Document Embeddings" [2024-10] [ICLR 2025] [[paper](https://arxiv.org/abs/2410.02525)]

- "Little Giants: Synthesizing High-Quality Embedding Data at Scale" [2024-10] [NAACL 2025] [[paper](https://arxiv.org/abs/2410.18634)]

- "A Simple Angle-based Approach for Contrastive Learning of Unsupervised Sentence Representation" [2024-11] [EMNLP 2024 Findings] [[paper](https://aclanthology.org/2024.findings-emnlp.318/)]

- "MAGNET: Augmenting Generative Decoders with Representation Learning and Infilling Capabilities" [2025-01] [ACL 2025] [[paper](https://arxiv.org/abs/2501.08648)]

- "Better Embeddings with Coupled Adam" [2025-02] [ACL 2025] [[paper](https://arxiv.org/abs/2502.08441)]

- "Refining Sentence Embedding Model through Ranking Sentences Generation with Large Language Models" [2025-02] [ACL 2025 Findings] [[paper](https://arxiv.org/abs/2502.13656)]

- "Multi-Sense Embeddings for Language Models and Knowledge Distillation" [2025-04] [ACL 2025 Findings] [[paper](https://arxiv.org/abs/2504.06036)]

- "Effective post-training embedding compression via temperature control in contrastive training" [2025-04] [ICLR 2025] [[paper](https://openreview.net/pdf?id=szRmEM8Kx5)]

- "Don’t Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation" [2025-05] [ACL 2025] [[paper](https://arxiv.org/abs/2505.24754)]

- "Adapting General-Purpose Embedding Models to Private Datasets Using Keyword-based Retrieval" [2025-05] [ACL 2025 Findings] [[paper](https://arxiv.org/abs/2506.00363)]

- "On the Relation Between Fine-Tuning, Topological Properties, and Task Performance in Sense-Enhanced Embeddings" [2025-07] [ACL 2025] [[paper](https://aclanthology.org/2025.acl-long.1151/)]

- "Cheap Character Noise for OCR-Robust Multilingual Embeddings" [2025-07] [ACL 2025 Findings] [[paper](https://aclanthology.org/2025.findings-acl.609/)]

- "Negative Matters: Multi-Granularity Hard-Negative Synthesis and Anchor-Token-Aware Pooling for Enhanced Text Embeddings" [2025-07] [ACL 2025] [[paper](https://arxiv.org/abs/2509.00842)]

- "Understanding the Influence of Synthetic Data for Text Embedders" [2025-07] [ACL 2025 Findings] [[paper](https://arxiv.org/abs/2509.06184)]

#### Training-Free Methods

- "Ditto: A Simple and Efficient Approach to Improve Sentence Embeddings" [2023-05] [EMNLP 2023] [[paper](https://arxiv.org/abs/2305.10786)]

- "EmbedTextNet: Dimension Reduction with Weighted Reconstruction and Correlation Losses for Efficient Text Embedding" [2023-07] [ACL 2023 Findings] [[paper](https://aclanthology.org/2023.findings-acl.625/)]

- "Repetition Improves Language Model Embeddings" [2024-02] [ICLR 2025] [[paper](https://arxiv.org/abs/2402.15449)]

- "Meta-Task Prompting Elicits Embeddings from Large Language Models" [2024-02] [ACL 2024] [[paper](https://arxiv.org/abs/2402.18458)]

- "PromptReps: Prompting Large Language Models to Generate Dense and Sparse Representations for Zero-Shot Document Retrieval" [2024-04] [EMNLP 2024] [[paper](https://arxiv.org/abs/2404.18424)]

- "Crafting Interpretable Embeddings for Language Neuroscience by Asking LLMs Questions" [2024-05] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2405.16714)]

- "Semantic Compression for Word and Sentence Embeddings using Discrete Wavelet Transform" [2024-07] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2508.00220)]

- "A General Framework for Producing Interpretable Semantic Text Embeddings" [2024-10] [ICLR 2025] [[paper](https://arxiv.org/abs/2410.03435)]

- "Your Mixture-of-Experts LLM Is Secretly an Embedding Model for Free" [2024-10] [ICLR 2025] [[paper](https://arxiv.org/abs/2410.10814)]

- "GenEOL: Harnessing the Generative Power of LLMs for Training-Free Sentence Embeddings" [2024-10] [NAACL 2025 Findings] [[paper](https://arxiv.org/abs/2410.14635)]

- "Length-Induced Embedding Collapse in PLM-based Models" [2024-10] [ACL 2025] [[paper](https://arxiv.org/abs/2410.24200)]

- "Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs" [2024-12] [ACL 2025] [[paper](https://arxiv.org/abs/2412.11556)]

- "LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations" [2025-05] [ACL 2025 Findings] [[paper](https://arxiv.org/abs/2505.10354)]

- "Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering" [2025-05] [ACL 2025] [[paper](https://arxiv.org/abs/2505.12831)]

### 1.2 Embedding Benchmarks

- "SciRepEval: A Multi-Format Benchmark for Scientific Document Representations" [2022-11] [EMNLP 2023] [[paper](https://arxiv.org/abs/2211.13308)]

- "Moving Beyond Downstream Task Accuracy for Information Retrieval Benchmarking" [2023-07] [ACL 2023 Findings] [[paper](https://aclanthology.org/2023.findings-acl.738/)]

- "C-Pack: Packed Resources For General Chinese Embeddings" [2023-09] [SIGIR 2024] [[paper](https://arxiv.org/abs/2309.07597)]

- "How Well Do Text Embedding Models Understand Syntax?" [2023-11] [EMNLP 2023 Findings] [[paper](https://arxiv.org/abs/2311.07996)]

- "FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions" [2024-03] [NAACL 2025] [[paper](https://arxiv.org/abs/2403.15246)]

- "LongEmbed: Extending Embedding Models for Long Context Retrieval" [2024-04] [EMNLP 2024] [[paper](https://arxiv.org/abs/2404.12096)]

- "Cocktail: A Comprehensive Information Retrieval Benchmark with LLM-Generated Documents Integration" [2024-05] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2405.16546)]

- "The Scandinavian Embedding Benchmarks: Comprehensive Assessment of Multilingual and Monolingual Text Embedding" [2024-06] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2406.02396)]

- "ClimRetrieve: A Benchmarking Dataset for Information Retrieval from Corporate Climate Disclosures" [2024-06] [EMNLP 2024] [[paper](https://arxiv.org/abs/2406.09818)]

- "The Russian-focused embedders’ exploration: ruMTEB benchmark and Russian embedding model design" [2024-08] [NAACL 2025] [[paper](https://arxiv.org/abs/2408.12503)]

- "Swan and ArabicMTEB: Dialect-Aware, Arabic-Centric, Cross-Lingual, and Cross-Cultural Embedding Models and Benchmarks" [2024-11] [NAACL 2025 Findings] [[paper](https://arxiv.org/abs/2411.01192)]

- "ALIGN-SIM: A Task-Free Test Bed for Evaluating and Interpreting Sentence Embeddings through Semantic Similarity Alignment" [2024-11] [EMNLP 2024 Findings] [[paper](https://aclanthology.org/2024.findings-emnlp.436/)]

- "AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark" [2024-12] [ACL 2025] [[paper](https://arxiv.org/abs/2412.13102)]

- "MMTEB: Massive Multilingual Text Embedding Benchmark" [2025-02] [ICLR 2025] [[paper](https://arxiv.org/abs/2502.13595)]

- "IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval" [2025-03] [NAACL 2025] [[paper](https://arxiv.org/abs/2503.04644)]

- "On Linear Representations and Pretraining Data Frequency in Language Models" [2025-04] [ICLR 2025] [[paper](https://arxiv.org/abs/2504.12459)]

- "MedEureka: A Medical Domain Benchmark for Multi-Granularity and Multi-Data-Type Embedding-Based Retrieval" [2025-04] [NAACL 2025 Findings] [[paper](https://aclanthology.org/2025.findings-naacl.154/)]

- "Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval" [2025-05] [ACL 2025 Findings] [[paper](https://arxiv.org/abs/2505.19356)]

### 1.3 Embedding Analysis

### Interpretability

- "Interpreting Embedding Spaces by Conceptualization" [2022-08] [EMNLP 2023] [[paper](https://arxiv.org/abs/2209.00445)]

- "Norm of Word Embedding Encodes Information Gain" [2022-12] [EMNLP 2023] [[paper](https://arxiv.org/abs/2212.09663)]

- "Discovering Universal Geometry in Embeddings with ICA" [2023-05] [EMNLP 2023] [[paper](https://arxiv.org/abs/2305.13175)]

- "Estimating class separability of text embeddings with persistent homology" [2023-05] [TMLR 2024] [[paper](https://arxiv.org/abs/2305.15016)]

- "A Method for Studying Semantic Construal in Grammatical Constructions with Interpretable Contextual Embedding Spaces" [2023-07] [ACL 2023] [[paper](https://aclanthology.org/2023.acl-long.14/)]

- "Demystifying Embedding Spaces using Large Language Models" [2023-10] [ICLR 2024] [[paper](https://arxiv.org/abs/2310.04475)]

- "The Linear Representation Hypothesis and the Geometry of Large Language Models" [2023-11] [ICML 2024] [[paper](https://arxiv.org/abs/2311.03658)]

- "Is Probing All You Need? Indicator Tasks as an Alternative to Probing Embedding Spaces" [2023-12] [EMNLP 2023 Findings] [[paper](https://aclanthology.org/2023.findings-emnlp.348/)]

- "On the Origins of Linear Representations in Large Language Models" [2024-03] [ICML 2024] [[paper](https://arxiv.org/abs/2403.03867)]

- "Adjusting Interpretable Dimensions in Embedding Space with Human Judgments" [2024-04] [NAACL 2024] [[paper](https://arxiv.org/abs/2404.02619)]

- "A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens" [2024-06] [ACL 2025] [[paper](https://arxiv.org/abs/2406.17378)]

- "Representational Isomorphism and Alignment of Multilingual Large Language Models" [2024-11] [EMNLP 2024 Findings] [[paper](https://aclanthology.org/2024.findings-emnlp.823/)]

- "Redundancy, Isotropy, and Intrinsic Dimensionality of Prompt-based Text Embeddings" [2025-06] [ACL 2025 Findings] [[paper](https://arxiv.org/abs/2506.01435)]

### Attack

- "Sentence Embedding Leaks More Information than You Expect: Generative Embedding Inversion Attack to Recover the Whole Sentence" [2023-05] [ACL 2023 Findings] [[paper](https://arxiv.org/abs/2305.03010)]

- "Text Embedding Inversion Security for Multilingual Language Models" [2024-01] [ACL 2024] [[paper](https://arxiv.org/abs/2401.12192)]

- "WARDEN: Multi-Directional Backdoor Watermarks for Embedding-as-a-Service Copyright Protection" [2024-03] [ACL 2024] [[paper](https://arxiv.org/abs/2403.01472)]

- "Can’t Hide Behind the API: Stealing Black-Box Commercial Embedding Models" [2024-06] [NAACL 2025 Findings] [[paper](https://arxiv.org/abs/2406.09355)]

- "Transferable Embedding Inversion Attack: Uncovering Privacy Risks in Text Embeddings without Model Queries" [2024-06] [ACL 2024] [[paper](https://arxiv.org/abs/2406.10280)]

- "WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service with Linear Transformation Watermarks" [2024-09] [ACL 2025] [[paper](https://arxiv.org/abs/2409.04459)]

- "An Inversion Attack Against Obfuscated Embedding Matrix in Language Model Inference" [2024-11] [EMNLP 2024] [[paper](https://aclanthology.org/2024.emnlp-main.126/)]

- "GuardEmb: Dynamic Watermark for Safeguarding Large Language Model Embedding Service Against Model Stealing Attack" [2024-11] [EMNLP 2024 Findings] [[paper](https://aclanthology.org/2024.findings-emnlp.441/)]

- "ALGEN: Few-shot Inversion Attacks on Textual Embeddings via Cross-Model Alignment and Generation" [2025-02] [ACL 2025] [[paper](https://arxiv.org/abs/2502.11308)]

- "Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models" [2025-07] [ACL 2025] [[paper](https://arxiv.org/abs/2507.18171)]

### Bias

- "Investigating the Frequency Distortion of Word Embeddings and Its Impact on Bias Metrics" [2022-11] [EMNLP 2023 Findings] [[paper](https://arxiv.org/abs/2211.08203)]

- "Is a Prestigious Job the same as a Prestigious Country? A Case Study on Multilingual Sentence Embeddings and European Countries" [2023-05] [EMNLP 2023 Findings] [[paper](https://arxiv.org/abs/2305.14482)]

- "Debiasing with Sufficient Projection: A General Theoretical Framework for Vector Representations" [2024-06] [NAACL 2024] [[paper](https://aclanthology.org/2024.naacl-long.332/)]

- "Discovering Biases in Information Retrieval Models Using Relevance Thesaurus as Global Explanation" [2024-10] [EMNLP 2024] [[paper](https://arxiv.org/abs/2410.03584)]

- "What is in a name? Mitigating Name Bias in Text Embedding Similarity via Anonymization" [2025-02] [ACL 2025 Findings] [[paper](https://arxiv.org/abs/2502.02903)]

- "PRISM: A Framework for Producing Interpretable Political Bias Embeddings with Political-Aware Cross-Encoder" [2025-05] [ACL 2025] [[paper](https://arxiv.org/abs/2505.24646)]

### Cross-Lingual Alignment

- "Understanding Linearity of Cross-Lingual Word Embedding Mappings" [2020-04] [TMLR 2022] [[paper](https://arxiv.org/abs/2004.01079)]

- "Linear Cross-Lingual Mapping of Sentence Embeddings" [2023-05] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2305.14256)]

- "Hyperpolyglot LLMs: Cross-Lingual Interpretability in Token Embeddings" [2023-11] [EMNLP 2023] [[paper](https://arxiv.org/abs/2311.18034)]

- "Enhancing Cross-lingual Sentence Embedding for Low-resource Languages with Word Alignment" [2024-04] [NAACL 2024 Findings] [[paper](https://arxiv.org/abs/2404.02490)]

- "mOthello: When Do Cross-Lingual Representation Alignment and Cross-Lingual Transfer Emerge in Multilingual Models?" [2024-04] [NAACL 2024 Findings] [[paper](https://arxiv.org/abs/2404.12444)]

- "The Semantic Hub Hypothesis: Language Models Share Semantic Representations Across Languages and Modalities" [2024-11] [ICLR 2025] [[paper](https://arxiv.org/abs/2411.04986)]

- "Steering into New Embedding Spaces: Analyzing Cross-Lingual Alignment Induced by Model Interventions in Multilingual Language Models" [2025-02] [ACL 2025] [[paper](https://arxiv.org/abs/2502.15639)]

### Others

- "The Trade-off between Universality and Label Efficiency of Representations from Contrastive Learning" [2023-03] [ICLR 2023] [[paper](https://arxiv.org/abs/2303.00106)]

- "Text Embeddings Reveal (Almost) As Much As Text" [2023-10] [EMNLP 2023] [[paper](https://arxiv.org/abs/2310.06816)]

- "When is an Embedding Model More Promising than Another?" [2024-06] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2406.07640)]

- "Semantics or spelling? Probing contextual word embeddings with orthographic noise" [2024-08] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2408.04162)]

- "What Should Embeddings Embed? Autoregressive Models Represent Latent Generating Distributions" [2024-06] [TMLR 2025] [[paper](https://arxiv.org/abs/2406.03707)]

- "Layer by Layer: Uncovering Hidden Representations in Language Models" [2025-02] [ICML 2025] [[paper](https://arxiv.org/abs/2502.02013)]

- "Embedding-Converter: A Unified Framework for Cross-Model Embedding Transformation" [2025-07] [ACL 2025] [[paper](https://aclanthology.org/2025.acl-long.1237/)]

### 1.4 Embedding Applications

#### Domain-Specific Embedding

- "Greenback Bears and Fiscal Hawks: Finance is a Jungle and Text Embeddings Must Adapt" [2024-11] [EMNLP 2024 Industry] [[paper](https://arxiv.org/abs/2411.07142)]

- "BALI: Enhancing Biomedical Language Representations through Knowledge Graph and Language Model Alignment" [2025-07] [SIGIR 2025] [[paper](https://arxiv.org/abs/2509.07588)]

#### Retrieval

- "Dense Passage Retrieval for Open-Domain Question Answering" [2020-04] [EMNLP 2020] [[paper](https://arxiv.org/abs/2004.04906)]

- "Unsupervised Dense Information Retrieval with Contrastive Learning" [2021-12] [TMLR 2022] [[paper](https://arxiv.org/abs/2112.09118)]

- "Promptagator: Few-shot Dense Retrieval From 8 Examples" [2022-09] [ICLR 2023] [[paper](https://arxiv.org/abs/2209.11755)]

- "Precise Zero-Shot Dense Retrieval without Relevance Labels" [2022-12] [ACL 2023] [[paper](https://arxiv.org/abs/2212.10496)]

- "Evaluating Embedding APIs for Information Retrieval" [2023-05] [ACL 2023 Industry] [[paper](https://arxiv.org/abs/2305.06300)]

- "Synergistic Interplay between Search and Large Language Models for Information Retrieval" [2023-05] [ACL 2024] [[paper](https://arxiv.org/abs/2305.07402)]

- "BERM: Training the Balanced and Extractable Representation for Matching to Improve Generalization Ability of Dense Retrieval" [2023-05] [ACL 2023] [[paper](https://arxiv.org/abs/2305.11052)]

- "Referral Augmentation for Zero-Shot Information Retrieval" [2023-05] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2305.15098)]

- "Typo-Robust Representation Learning for Dense Retrieval" [2023-06] [ACL 2023] [[paper](https://arxiv.org/abs/2306.10348)]

- "Large Language Models for Information Retrieval: A Survey" [2023-08] [arXiv] [[paper](https://arxiv.org/abs/2308.07107)]

- "Search-Adaptor: Embedding Customization for Information Retrieval" [2023-10] [ACL 2024] [[paper](https://arxiv.org/abs/2310.08750)]

- "Interpreting Conversational Dense Retrieval by Rewriting-Enhanced Inversion of Session Embedding" [2024-02] [ACL 2024] [[paper](https://arxiv.org/abs/2402.12774)]

- "Self-Retrieval: End-to-End Information Retrieval with One Large Language Model" [2024-03] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2403.00801)]

- "Spiral of Silence: How is Large Language Model Killing Information Retrieval?—A Case Study on Open Domain Question Answering" [2024-04] [ACL 2024] [[paper](https://arxiv.org/abs/2404.10496)]

- "Multivariate Representation Learning for Information Retrieval" [2024-04] [SIGIR 2023] [[paper](https://arxiv.org/abs/2304.14522)]

- "SetCSE: Set Operations using Contrastive Learning of Sentence Embeddings" [2024-04] [ICLR 2024] [[paper](https://arxiv.org/abs/2404.17606)]

- "USTAD: Unified Single-model Training Achieving Diverse Scores for Information Retrieval" [2024-05] [ICML 2024] [[paper](https://openreview.net/forum?id=LbEB39lZqp)]

- "ContrastiveMix: Overcoming Code-Mixing Dilemma in Cross-Lingual Transfer for Information Retrieval" [2024-06] [NAACL 2024] [[paper](https://aclanthology.org/2024.naacl-short.17/)]

- "A Fresh Take on Stale Embeddings: Improving Dense Retriever Training with Corrector Networks" [2024-09] [ICML 2024] [[paper](https://arxiv.org/abs/2409.01890)]

- "NUDGE: Lightweight Non-Parametric Fine-Tuning of Embeddings for Retrieval" [2024-09] [ICLR 2025] [[paper](https://arxiv.org/abs/2409.02343)]

- "Generative Retrieval Meets Multi-Graded Relevance" [2024-09] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2409.18409)]

- "Link, Synthesize, Retrieve: Universal Document Linking for Zero-Shot Information Retrieval" [2024-10] [EMNLP 2024] [[paper](https://arxiv.org/abs/2410.18385)]

- "Optimizing Multi-Hop Document Retrieval Through Intermediate Representations" [2025-03] [ACL 2025 Findings] [[paper](https://arxiv.org/abs/2503.04796)]

- "Search Query Embeddings via User-behavior-driven Contrastive Learning" [2025-04] [NAACL 2025 Industry] [[paper](https://aclanthology.org/2025.naacl-industry.12/)]

- "RetrieverGuard: Empowering Information Retrieval to Combat LLM-Generated Misinformation" [2025-04] [NAACL 2025 Findings] [[paper](https://aclanthology.org/2025.findings-naacl.249/)]

#### Reranking

- "Passage Re-ranking with BERT" [2019-01] [arXiv] [[paper](https://arxiv.org/abs/1901.04085)]

- "Multi-Stage Document Ranking with BERT" [2019-10] [arXiv] [[paper](https://arxiv.org/abs/1910.14424)]

- "Document Ranking with a Pretrained Sequence-to-Sequence Model" [2020-03] [EMNLP 2020 Findings] [[paper](https://arxiv.org/abs/2003.06713)]

- "Beyond [CLS] through Ranking by Generation" [2020-10] [EMNLP 2020] [[paper](https://arxiv.org/abs/2010.03073)]

- "The Expando-Mono-Duo Design Pattern for Text Ranking with Pretrained Sequence-to-Sequence Models" [2021-01] [arXiv] [[paper](https://arxiv.org/abs/2101.05667)]

- "Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline" [2021-01] [ECIR 2021] [[paper](https://arxiv.org/abs/2101.08751)]

- "InPars: Unsupervised Dataset Generation for Information Retrieval" [2022-02] [SIGIR 2022] [[paper](https://arxiv.org/abs/2202.05144)]

- "Improving Passage Retrieval with Zero-Shot Question Generation" [2022-04] [EMNLP 2022] [[paper](https://arxiv.org/abs/2204.07496)]

- "RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses" [2022-10] [SIGIR 2023] [[paper](https://arxiv.org/abs/2210.10634)]

- "ExaRanker: Synthetic Explanations Improve Neural Rankers" [2023-01] [SIGIR 2023] [[paper](https://arxiv.org/abs/2301.10521)]

- "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents" [2023-04] [EMNLP 2023] [[paper](https://arxiv.org/abs/2304.09542)]

- "Zero-Shot Listwise Document Reranking with a Large Language Model" [2023-05] [arXiv] [[paper](https://arxiv.org/abs/2305.02156)]

- "Discrete Prompt Optimization via Constrained Generation for Zero-shot Re-ranker" [2023-05] [ACL 2023 Findings] [[paper](https://arxiv.org/abs/2305.13729)]

- "Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting" [2023-06] [NAACL 2024 Findings] [[paper](https://arxiv.org/abs/2306.17563)]

- "RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models" [2023-09] [arXiv] [[paper](https://arxiv.org/abs/2309.15088)]

- "Fine-Tuning LLaMA for Multi-Stage Text Retrieval" [2023-10] [SIGIR 2024] [[paper](https://arxiv.org/abs/2310.08319)]

- "A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models" [2023-10] [SIGIR 2024] [[paper](https://arxiv.org/abs/2310.09497)]

- "Open-source Large Language Models are Strong Zero-shot Query Likelihood Models for Document Ranking" [2023-10] [EMNLP 2023 Findings] [[paper](https://arxiv.org/abs/2310.13243)]

- "Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels" [2023-10] [NAACL 2024] [[paper](https://arxiv.org/abs/2310.14122)]

- "PaRaDe: Passage Ranking using Demonstrations with LLMs" [2023-10] [EMNLP 2023 Findings] [[paper](https://arxiv.org/abs/2310.14408)]

- "A Two-Stage Adaptation of Large Language Models for Text Ranking" [2023-11] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2311.16720)]

- "RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!" [2023-12] [arXiv] [[paper](https://arxiv.org/abs/2312.02724)]

- "Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages" [2023-12] [ACL 2024] [[paper](https://arxiv.org/abs/2312.16159)]

- "Expand, Highlight, Generate: RL-driven Document Generation for Passage Reranking" [2023-12] [EMNLP 2023] [[paper](https://aclanthology.org/2023.emnlp-main.623/)]

- "EcoRank: Budget-Constrained Text Re-ranking Using Large Language Models" [2024-02] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2402.10866)]

- "ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval" [2024-02] [ACL 2024] [[paper](https://arxiv.org/abs/2402.15838)]

- "Consolidating Ranking and Relevance Predictions of Large Language Models through Post-Processing" [2024-04] [EMNLP 2024] [[paper](https://arxiv.org/abs/2404.11791)]

- "AGRaME: Any-Granularity Ranking with Multi-Vector Embeddings" [2024-05] [EMNLP 2024] [[paper](https://arxiv.org/abs/2405.15028)]

- "Leveraging Passage Embeddings for Efficient Listwise Reranking with Large Language Models" [2024-06] [WWW 2025] [[paper](https://arxiv.org/abs/2406.14848)]

- "FIRST: Faster Improved Listwise Reranking with Single Token Decoding" [2024-06] [EMNLP 2024] [[paper](https://arxiv.org/abs/2406.15657)]

- "RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs" [2024-07] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2407.02485)]

- "PRP-Graph: Pairwise Ranking Prompting to LLMs with Graph Aggregation for Effective Text Re-ranking" [2024-08] [ACL 2024] [[paper](https://aclanthology.org/2024.acl-long.313/)]

- "Few-shot Prompting for Pairwise Ranking: An Effective Non-Parametric Retrieval Model" [2024-09] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2409.17745)]

- "HyQE: Ranking Contexts with Hypothetical Query Embeddings" [2024-10] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2410.15262)]

- "Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models" [2024-12] [ACL 2025] [[paper](https://arxiv.org/abs/2412.14574)]

- "ASRank: Zero-Shot Re-Ranking with Answer Scent for Document Retrieval" [2025-01] [NAACL 2025 Findings] [[paper](https://arxiv.org/abs/2501.15245)]

- "Gumbel Reranking: Differentiable End-to-End Reranker Optimization" [2025-02] [ACL 2025] [[paper](https://arxiv.org/abs/2502.11116)]

- "Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering" [2025-03] [ACL 2025] [[paper](https://arxiv.org/abs/2503.01606)]

- "Shifting from Ranking to Set Selection for Retrieval Augmented Generation" [2025-07] [ACL 2025] [[paper](https://arxiv.org/abs/2507.06838)]

- "QDER: Query-Specific Document and Entity Representations for Multi-Vector Document Re-Ranking" [2025-07] [SIGIR 2025] [[paper](https://dl.acm.org/doi/10.1145/3726302.3730065)]

#### Classification & Clustering & STS

- "Robust Representation Learning with Reliable Pseudo-labels Generation via Self-Adaptive Optimal Transport for Short Text Clustering" [2023-05] [ACL 2023] [[paper](https://arxiv.org/abs/2305.16335)]

- "Going Beyond Sentence Embeddings: A Token-Level Matching Algorithm for Calculating Semantic Textual Similarity" [2023-07] [ACL 2023] [[paper](https://aclanthology.org/2023.acl-short.49/)]

- "Transductive Learning for Textual Few-Shot Classification in API-based Embedding Models" [2023-10] [EMNLP 2023] [[paper](https://arxiv.org/abs/2310.13998)]

- "Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings" [2025-06] [ACL 2025] [[paper](https://arxiv.org/abs/2506.00277)]

#### Sentiment Analysis

- "Label-Aware Hyperbolic Embeddings for Fine-grained Emotion Classification" [2023-06] [ACL 2023] [[paper](https://arxiv.org/abs/2306.14822)]

- "TATA: Stance Detection via Topic-Agnostic and Topic-Aware Embeddings" [2023-10] [EMNLP 2023] [[paper](https://arxiv.org/abs/2310.14450)]

- "Chain-of-Thought Embeddings for Stance Detection on Social Media" [2023-10] [EMNLP 2023 Findings] [[paper](https://arxiv.org/abs/2310.19750)]

- "Unmasking the Hidden Meaning: Bridging Implicit and Explicit Hate Speech Embedding Representations" [2023-12] [EMNLP 2023 Findings] [[paper](https://aclanthology.org/2023.findings-emnlp.441/)]

#### Discourse Analysis & Topic Modeling

- "Effective Neural Topic Modeling with Embedding Clustering Regularization" [2023-06] [ICML 2023] [[paper](https://arxiv.org/abs/2306.04217)]

- "Context-guided Embedding Adaptation for Effective Topic Modeling in Low-Resource Regimes" [2023-12] [NeurIPS 2023] [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fce176458ff542940fa3ed16e6f9c852-Abstract-Conference.html)]

- "Matching Varying-Length Texts via Topic-Informed and Decoupled Sentence Embeddings" [2024-06] [NAACL 2024 Findings] [[paper](https://aclanthology.org/2024.findings-naacl.81/)]

- "Story Embeddings — Narrative-Focused Representations of Fictional Stories" [2024-11] [EMNLP 2024] [[paper](https://aclanthology.org/2024.emnlp-main.339/)]

- "Topic Modeling: Contextual Token Embeddings Are All You Need" [2024-11] [EMNLP 2024 Findings] [[paper](https://aclanthology.org/2024.findings-emnlp.790/)]

- "Conditional Dichotomy Quantification via Geometric Embedding" [2025-07] [ACL 2025] [[paper](https://aclanthology.org/2025.acl-long.383/)]

#### LLM Self-Evaluation

- "Analyzing Transformers in Embedding Space" [2022-09] [ACL 2023] [[paper](https://arxiv.org/abs/2209.02535)]

- "SelfIE: Self-Interpretation of Large Language Model Embeddings" [2024-03] [ICML 2024] [[paper](https://arxiv.org/abs/2403.10949)]

- "Embedding Trajectory for Out-of-Distribution Detection in Mathematical Reasoning" [2024-05] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2405.14039)]

- "Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation" [2024-10] [ICLR 2025] [[paper](https://arxiv.org/abs/2410.13640)]

- "Embedding and Gradient Say Wrong: A White-Box Method for Hallucination Detection" [2024-11] [EMNLP 2024] [[paper](https://aclanthology.org/2024.emnlp-main.116/)]

- "CED: Comparing Embedding Differences for Detecting Out-of-Distribution and Hallucinated Text" [2024-11] [EMNLP 2024 Findings] [[paper](https://aclanthology.org/2024.findings-emnlp.874/)]

#### Regression

- "Understanding LLM Embeddings for Regression" [2024-11] [TMLR 2025] [[paper](https://arxiv.org/abs/2411.14708)]

## 2. Code Embedding

### 2.1 Models

- "Learning and Evaluating Contextual Embedding of Source Code" [2019-12] [ICML 2020] [[paper](https://arxiv.org/abs/2001.00059)]

- "CodeBERT: A Pre-Trained Model for Programming and Natural Languages" [2020-02] [EMNLP 2020 findings] [[paper](https://arxiv.org/abs/2002.08155)]

- "GraphCodeBERT: Pre-training Code Representations with Data Flow" [2020-09] [ICLR 2021] [[paper](https://arxiv.org/abs/2009.08366)]

- "Multi-task Learning based Pre-trained Language Model for Code Completion" [2020-12] [ASE 2020] [[paper](https://arxiv.org/abs/2012.14631)]

- "SynCoBERT: Syntax-Guided Multi-Modal Contrastive Pre-Training for Code Representation" [2021-08] [[paper](https://arxiv.org/abs/2108.04556)]

- "Towards Learning (Dis)-Similarity of Source Code from Program Contrasts" [2021-10] [ACL 2022] [[paper](https://arxiv.org/abs/2110.03868)]

- "UniXcoder: Unified Cross-Modal Pre-training for Code Representation" [2022-03] [ACL 2022] [[paper](https://arxiv.org/abs/2203.03850)]

- "CODE-MVP: Learning to Represent Source Code from Multiple Views with Contrastive Pre-Training" [2022-05] [NAACL 2022 Findings] [[paper](https://arxiv.org/abs/2205.02029)]

- "Pre-Training Representations of Binary Code Using Contrastive Learning" [2022-10] [TMLR 2025] [[paper](https://arxiv.org/abs/2210.05102)]

- "Code Representation Pre-training with Complements from Program Executions" [2023-09] [EMNLP 2024 Industry] [[paper](https://arxiv.org/abs/2309.09980)]

- "Language Agnostic Code Embeddings" [2023-10] [NAACL 2024] [[paper](https://arxiv.org/abs/2310.16803)]

- "Pass-Tuning: Towards Structure-Aware Parameter-Efficient Tuning for Code Representation Learning" [2023-12] [EMNLP 2023 Findings] [[paper](https://aclanthology.org/2023.findings-emnlp.42/)]

- "Code Representation Learning At Scale" [2024-02] [ICLR 2024] [[paper](https://arxiv.org/abs/2402.01935)]

- "GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding" [2024-09] [ACL 2025] [[paper](https://arxiv.org/abs/2409.04183)]

- "CodeXEmbed: A Generalist Embedding Model Family for Multiligual and Multi-task Code Retrieval" [2024-11] [[paper](https://arxiv.org/abs/2411.12644)]

- "CodeSSM: Towards State Space Models for Code Understanding" [2025-05] [EMNLP 2025] [[paper](https://arxiv.org/abs/2505.01475)]

- "Towards A Generalist Code Embedding Model Based On Massive Data Synthesis" [2025-05] [[paper](https://arxiv.org/abs/2505.12697)]

- "Efficient Code Embeddings from Code Generation Models" [2025-08] [[paper](https://arxiv.org/abs/2508.21290)]

### 2.2 Datasets & Benchmarks

- "StaQC: A Systematically Mined Question-Code Dataset from Stack Overflow" [2018-03] [WWW 2018] [[paper](https://arxiv.org/abs/1803.09371)]

- "Deep Code Search" [2018-05] [ICSE 2018] [[paper](https://dl.acm.org/doi/10.1145/3180155.3180167)]

- "Learning to Mine Aligned Code and Natural Language Pairs from Stack Overflow" [2018-05] [MSR 2018] [[paper](https://arxiv.org/abs/1805.08949)]

- "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search" [2019-09] [[paper](https://arxiv.org/abs/1909.09436)]

- "CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation" [2021-02] [NeurIPS 2021 Datasets and Benchmarks] [[paper](https://arxiv.org/abs/2102.04664)]

- "CoSQA: 20,000+ Web Queries for Code Search and Question Answering" [2021-05] [ACL 2021] [[paper](https://arxiv.org/abs/2105.13239)]

- "ProCQA: A Large-scale Community-based Programming Question Answering Dataset for Code Search" [2024-03] [LREC 2024] [[paper](https://arxiv.org/abs/2403.16702)]

- "CodeRAG-Bench: Can Retrieval Augment Code Generation?" [2024-06] [NAACL 2025 Findings] [[paper](https://arxiv.org/abs/2406.14497)]

- "CoIR: A Comprehensive Benchmark for Code Information Retrieval Models" [2024-07] [ACL 2025] [[paper](https://arxiv.org/abs/2407.02883)]

- "What can Large Language Models Capture about Code Functional Equivalence?" [2024-08] [NAACL 2025 Findings] [[paper](https://arxiv.org/abs/2408.11081)]

- "CmdCaliper: A Semantic-Aware Command-Line Embedding Model and Dataset for Security Research" [2024-11] [EMNLP 2024] [[paper](https://arxiv.org/abs/2411.01176)]

- "CoRNStack: High-Quality Contrastive Data for Better Code Retrieval and Reranking" [2024-12] [ICLR 2025] [[paper](https://arxiv.org/abs/2412.01007)]

### 2.3 Applications

#### Retrieval

- "Self-Supervised Contrastive Learning for Code Retrieval and Summarization via Semantic-Preserving Transformations" [2020-09] [SIGIR 2021] [[paper](https://arxiv.org/abs/2009.02731)]

- "REINFOREST: Reinforcing Semantic Code Similarity for Cross-Lingual Code Search Models" [2023-05] [[paper](https://arxiv.org/abs/2305.03843)]

- "Rewriting the Code: A Simple Method for Large Language Model Augmented Code Search" [2024-01] [ACL 2024] [[paper](https://arxiv.org/abs/2401.04514)]

- "Revisiting Code Similarity Evaluation with Abstract Syntax Tree Edit Distance" [2024-04] [ACL 2024 short] [[paper](https://arxiv.org/abs/2404.08817)]

- "You Augment Me: Exploring ChatGPT-based Data Augmentation for Semantic Code Search" [2024-08] [[paper](https://arxiv.org/abs/2408.05542)]

- "Instructive Code Retriever: Learn from Large Language Model's Feedback for Code Intelligence Tasks" [2024-10] [ASE 2024] [[paper](https://arxiv.org/abs/2410.11300)]

- "Optimizing Code Retrieval: High-Quality and Scalable Dataset Annotation through Large Language Models" [2024-11] [EMNLP 2024] [[paper](https://aclanthology.org/2024.emnlp-main.123/)]

- "Enhancing Learning-Based Binary Code Similarity Detection Model through Adversarial Training with Multiple Function Variants" [2024-11] [EMNLP 2024 Findings] [[paper](https://aclanthology.org/2024.findings-emnlp.673/)]

- "OASIS: Order-Augmented Strategy for Improved Code Search" [2025-03] [ACL 2025] [[paper](https://arxiv.org/abs/2503.08161)]

- "Zero-Shot Cross-Domain Code Search without Fine-Tuning" [2025-04] [[paper](https://arxiv.org/abs/2504.07740)]

- "CoRet: Improved Retriever for Code Editing" [2025-05] [ACL 2025] [[paper](https://arxiv.org/abs/2505.24715)]

- "Beyond the Surface: A Solution-Aware Retrieval Model for Competition-level Code Generation" [2025-09] [EMNLP 2025 Findings] [[paper](https://arxiv.org/abs/2509.01129)]

- "Beyond Function-Level Search: Repository-Aware Dual-Encoder Code Retrieval with Adversarial Verification" [2025-10] [EMNLP 2025 Findings] [[paper](https://arxiv.org/abs/2510.24749)]

#### Reranking

- "Fault-Aware Neural Code Rankers" [2022-06] [NeurIPS 2022] [[paper](https://arxiv.org/abs/2206.03865)]

- "Coder Reviewer Reranking for Code Generation" [2022-11] [ICML 2023] [[paper](https://arxiv.org/abs/2211.16490)]

- "LEVER: Learning to Verify Language-to-Code Generation with Execution" [2023-02] [ICML 2023] [[paper](https://arxiv.org/abs/2302.08468)]

- "Functional Overlap Reranking for Neural Code Generation" [2023-10] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2311.03366)]

- "Top Pass: Improve Code Generation by Pass@k-Maximized Code Ranking" [2024-08] [[paper](https://arxiv.org/abs/2408.05715)]

- "Sifting through the Chaff: On Utilizing Execution Feedback for Ranking the Generated Code Candidates" [2024-08] [ASE 2024] [[paper](https://arxiv.org/abs/2408.13976)]

- "B4: Towards Optimal Assessment of Plausible Code Solutions with Plausible Tests" [2024-09] [ASE 2024] [[paper](https://arxiv.org/abs/2409.08692)]

#### Others

- "Coding-PTMs: How to Find Optimal Code Pre-trained Models for Code Embedding in Vulnerability Detection?" [2024-08] [ASE 2024] [[paper](https://arxiv.org/abs/2408.04863)]

- "Learning Cross-Architecture Instruction Embeddings for Binary Code Analysis in Low-Resource Architectures" [2024-08] [NAACL 2024 Findings] [[paper](https://aclanthology.org/2024.findings-naacl.84/)]

- "CLeVeR: Multi-modal Contrastive Learning for Vulnerability Code Representation" [2025-07] [ACL 2025 Findings] [[paper](https://aclanthology.org/2025.findings-acl.414/)]

## 3. Vision Embedding

### 3.1 Models

#### General Representation Models

- "Diffusion Based Representation Learning" [2021-05] [ICML 2023] [[paper](https://arxiv.org/abs/2105.14257)]

- "Simplicial Embeddings in Self-Supervised Learning and Downstream Classification" [2022-04] [ICLR 2023] [[paper](http://arxiv.org/abs/2204.00616)]

- "Minimalistic Unsupervised Representation Learning with the Sparse Manifold Transform" [2022-09] [ICLR 2023] [[paper](http://arxiv.org/abs/2209.15261)]

- "OPERA: Omni-Supervised Representation Learning with Hierarchical Supervisions" [2022-10] [ICCV 2023] [[paper](http://arxiv.org/abs/2210.05557)]

- "Spatio-Temporal Crop Aggregation for Video Representation Learning" [2022-11] [ICCV 2023] [[paper](http://arxiv.org/abs/2211.17042)]

- "Semantics-Consistent Feature Search for Self-Supervised Visual Representation Learning" [2022-12] [ICCV 2023] [[paper](http://arxiv.org/abs/2212.06486)]

- "STAIR: Learning Sparse Text and Image Representation in Grounded Tokens" [2023-01] [EMNLP 2023] [[paper](http://arxiv.org/abs/2301.13081)]

- "Image-text embedding learning via visual and textual semantic reasoning" [2023-01] [TPAMI 2023] [[paper](https://ieeexplore.ieee.org/document/9706340)]

- "3D Neural Embedding Likelihood: Probabilistic Inverse Graphics for Robust 6D Pose Estimation" [2023-02] [ICCV 2023] [[paper](http://arxiv.org/abs/2302.03744)]

- "Embedding Fourier for Ultra-High-Definition Low-Light Image Enhancement" [2023-02] [ICLR 2023] [[paper](http://arxiv.org/abs/2302.11831)]

- "ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation" [2023-02] [ICCV 2023] [[paper](http://arxiv.org/abs/2302.13848)]

- "Layer Grafted Pre-training: Bridging Contrastive Learning And Masked Image Modeling For Label-Efficient Representations" [2023-02] [ICLR 2023] [[paper](https://arxiv.org/abs/2302.14138)]

- "NAISR: A 3D Neural Additive Model for Interpretable Shape Representation" [2023-03] [ICLR 2024] [[paper](http://arxiv.org/abs/2303.09234)]

- "Open-vocabulary Panoptic Segmentation with Embedding Modulation" [2023-03] [ICCV 2023] [[paper](http://arxiv.org/abs/2303.11324)]

- "Rotation and Translation Invariant Representation Learning with Implicit Neural Representations" [2023-04] [ICML 2023] [[paper](http://arxiv.org/abs/2304.13995)]

- "ManagerTower: Aggregating the Insights of Uni-Modal Experts for Vision-Language Representation Learning" [2023-05] [ACL 2023] [[paper](http://arxiv.org/abs/2306.00103)]

- "Isometric Quotient Variational Auto-Encoders for Structure-Preserving Representation Learning" [2023-05] [NeurIPS 2023] [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7af8e3dfefe6e3141144197b8fa44f79-Abstract-Conference.html)]

- "ADDP: Learning General Representations for Image Recognition and Generation with Alternating Denoising Diffusion Process" [2023-06] [ICLR 2024] [[paper](http://arxiv.org/abs/2306.05423)]

- "MOFI: Learning Image Representations from Noisy Entity Annotated Images" [2023-06] [ICLR 2024] [[paper](http://arxiv.org/abs/2306.07952)]

- "MOCA: Self-supervised Representation Learning by Predicting Masked Online Codebook Assignments" [2023-07] [TMLR 2024] [[paper](https://arxiv.org/abs/2307.09361)]

- "Conditional Cross Attention Network for Multi-Space Embedding without Entanglement in Only a SINGLE Network" [2023-07] [ICCV 2023] [[paper](http://arxiv.org/abs/2307.13254)]

- "SimFIR: A Simple Framework for Fisheye Image Rectification with Self-supervised Representation Learning" [2023-08] [ICCV 2023] [[paper](http://arxiv.org/abs/2308.09040)]

- "Boosting Semantic Segmentation from the Perspective of Explicit Class Embeddings" [2023-08] [ICCV 2023] [[paper](http://arxiv.org/abs/2308.12894)]

- "Dynamics-inspired Neuromorphic Visual Representation Learning" [2023-08] [ICML 2023] [[paper](https://proceedings.mlr.press/v202/pei23b.html)]

- "RevColV2: Exploring Disentangled Representations in Masked Image Modeling" [2023-09] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2309.01005)]

- "URLOST: Unsupervised Representation Learning without Stationarity or Topology" [2023-10] [ICLR 2025] [[paper](http://arxiv.org/abs/2310.04496)]

- "DyST: Towards Dynamic Neural Scene Representations on Real-World Videos" [2023-10] [ICLR 2024] [[paper](http://arxiv.org/abs/2310.06020)]

- "E2PNet: Event to Point Cloud Registration with Spatio-Temporal Representation Learning" [2023-10] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2311.18433)]

- "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks" [2023-11] [CVPR 2024] [[paper](https://arxiv.org/abs/2311.06242)]

- "VIT-LENS: Towards Omni-modal Representations" [2023-11] [CVPR 2024] [[paper](https://arxiv.org/abs/2311.16081)]

- "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model" [2024-01] [ICML 2024] [[paper](http://arxiv.org/abs/2401.09417)]

- "Unified Generation, Reconstruction, and Representation: Generalized Diffusion with Adaptive Latent Encoding-Decoding" [2024-02] [ICML 2024] [[paper](http://arxiv.org/abs/2402.19009)]

- "Revisiting Feature Prediction for Learning Visual Representations from Video" [2024-02] [TMLR 2024] [[paper](https://arxiv.org/abs/2404.08471)]

- "Tripod: Three Complementary Inductive Biases for Disentangled Representation Learning" [2024-04] [ICML 2024] [[paper](http://arxiv.org/abs/2404.10282)]

- "Diffusion Bridge AutoEncoders for Unsupervised Representation Learning" [2024-05] [ICLR 2025] [[paper](http://arxiv.org/abs/2405.17111)]

- "Harmony: A Joint Self-Supervised and Weakly-Supervised Framework for Learning General Purpose Visual Representations" [2024-05] [TMLR 2025] [[paper](https://arxiv.org/abs/2405.14239)]

- "MSPE: Multi-Scale Patch Embedding Prompts Vision Transformers to Any Resolution" [2024-05] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2405.18240)]

- "Coarse-To-Fine Tensor Trains for Compact Visual Representations" [2024-06] [ICML 2024] [[paper](http://arxiv.org/abs/2406.04332)]

- "Learning 1D Causal Visual Representation with De-focus Attention Networks" [2024-06] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2406.04342)]

- "Learning Color Equivariant Representations" [2024-06] [ICLR 2025] [[paper](http://arxiv.org/abs/2406.09588)]

- "SuperSVG: Superpixel-based Scalable Vector Graphics Synthesis" [2024-06] [CVPR 2024] [[paper](https://arxiv.org/abs/2406.09794)]

- "Autoencoding Conditional Neural Processes for Representation Learning" [2024-08] [ICML 2024] [[paper](https://openreview.net/forum?id=XuQPA4D396)]

- "Structuring Representation Geometry with Rotationally Equivariant Contrastive Learning" [2024-08] [ICLR 2024] [[paper](https://openreview.net/forum?id=lgaFMvZHSJ)]

- "Denoising Autoregressive Representation Learning" [2024-09] [ICML 2024] [[paper](https://openreview.net/forum?id=dW29JZj0G5)]

- "BiGR: Harnessing Binary Latent Codes for Image Generation and Improved Visual Representation Capabilities" [2024-10] [ICLR 2025] [[paper](http://arxiv.org/abs/2410.14672)]

- "Pretrained Reversible Generation as Unsupervised Visual Representation Learning" [2024-11] [ICCV 2025] [[paper](http://arxiv.org/abs/2412.01787)]

- "Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation" [2025-03] [ICML 2025] [[paper](http://arxiv.org/abs/2503.01776)]

- "Spectral State Space Model for Rotation-Invariant Visual Representation Learning" [2025-03] [CVPR 2025] [[paper](https://arxiv.org/abs/2503.06369)]

- "End-to-End Implicit Neural Representations for Classification" [2025-03] [CVPR 2025] [[paper](https://arxiv.org/abs/2503.18123)]

- "MergeVQ: A Unified Framework for Visual Generation and Representation with Disentangled Token Merging and Quantization" [2025-03] [CVPR 2025] [[paper](https://arxiv.org/abs/2504.00999)]

- "Scaling Language-Free Visual Representation Learning" [2025-04] [ICCV 2025] [[paper](http://arxiv.org/abs/2504.01017)]

- "CTRL-O: Language-Controllable Object-Centric Visual Representation Learning" [2025-03] [CVPR 2025] [[paper](https://arxiv.org/abs/2503.21747)]

- "Breaking the Modality Barrier: Universal Embedding Learning with Multimodal LLMs" [2025-04] [ACM MM 2025] [[paper](http://arxiv.org/abs/2504.17432)]

- "PIN: Prolate Spheroidal Wave Function-based Implicit Neural Representations" [2025-05] [ICLR 2025] [[paper](https://openreview.net/forum?id=Eh1QM3OK51)]

- "GECO: Geometrically Consistent Embedding with Lightspeed Inference" [2025-08] [ICCV 2025] [[paper](http://arxiv.org/abs/2508.00746)]

- "Efficient Object-Centric Representation Learning using Masked Generative Modeling" [2025-09] [TMLR 2025] [[paper](https://openreview.net/forum?id=t9KvOYPeL3)]

#### 3D Representation

- "Implicit Autoencoder for Point-Cloud Self-Supervised Representation Learning" [2022-01] [ICCV 2023] [[paper](http://arxiv.org/abs/2201.00785)]

- "Autoencoders as Cross-Modal Teachers: Can Pretrained 2D Image Transformers Help 3D Representation Learning?" [2022-12] [ICLR 2023] [[paper](http://arxiv.org/abs/2212.08320)]

- "Contrast with Reconstruct: Contrastive 3D Representation Learning Guided by Generative Pretraining" [2023-02] [ICML 2023] [[paper](http://arxiv.org/abs/2302.02318)]

- "Clustering based Point Cloud Representation Learning for 3D Analysis" [2023-07] [ICCV 2023] [[paper](http://arxiv.org/abs/2307.14605)]

- "LinNet: Linear Network for Efficient Point Cloud Representation Learning" [2024-02] [NeurIPS 2024] [[paper](http://papers.nips.cc/paper_files/paper/2024/hash/4bfcebedf7a2967c410b64670f27f904-Abstract-Conference.html)]

- "Unsupervised 3D Scene Representation Learning via Movable Object Inference" [2024-03] [TMLR 2024] [[paper](https://par.nsf.gov/biblio/10521413-unsupervised-scene-representation-learning-via-movable-object-inference)]

- "NeRM: Learning Neural Representations for High-Framerate Human Motion Synthesis" [2024-05] [ICLR 2024] [[paper](https://openreview.net/forum?id=sOJriBlOFd)]

- "Neural Pose Representation Learning for Generating and Transferring Non-Rigid Object Poses" [2024-06] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2406.09728)]

- "Uni3D: Exploring Unified 3D Representation at Scale" [2024-07] [ICLR 2024] [[paper](https://openreview.net/forum?id=wcaE4Dfgt8)]

- "Positional Prompt Tuning for Efficient 3D Representation Learning" [2024-08] [ACM MM 2025] [[paper](http://arxiv.org/abs/2408.11567)]

- "LaGeM: A Large Geometry Model for 3D Representation Learning and Diffusion" [2024-10] [ICLR 2025] [[paper](http://arxiv.org/abs/2410.01295)]

- "SE(3) Equivariant Ray Embeddings for Implicit Multi-View Depth Estimation" [2024-11] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2411.07326)]

- "SEGS-SLAM: Structure-enhanced 3D Gaussian Splatting SLAM with Appearance Embedding" [2025-01] [ICCV 2025] [[paper](http://arxiv.org/abs/2501.05242)]

- "UniMamba: Unified Spatial-Channel Representation Learning with Group-Efficient Mamba for LiDAR-based 3D Object Detection" [2025-03] [CVPR 2025] [[paper](https://arxiv.org/abs/2503.12009)]

- "Retri3D: 3D Neural Graphics Representation Retrieval" [2025-05] [ICLR 2025] [[paper](https://openreview.net/forum?id=q3EbOXb4y1)]

- "StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning" [2025-06] [ICCV 2025] [[paper](http://arxiv.org/abs/2506.21541)]

- "A Neural Representation Framework with LLM-Driven Spatial Reasoning for Open-Vocabulary 3D Visual Grounding" [2025-07] [ACM MM 2025] [[paper](http://arxiv.org/abs/2507.06719)]

- "SPHERE: Semantic-PHysical Engaged REpresentation for 3D Semantic Scene Completion" [2025-09] [ACM MM 2025] [[paper](http://arxiv.org/abs/2509.11171)]

- "Unifi3D: A Study on 3D Representations for Generation and Reconstruction in a Common Framework" [2025-09] [TMLR 2025] [[paper](https://arxiv.org/abs/2509.02474)]

#### Video Representation

- "Entropy-driven Unsupervised Keypoint Representation Learning in Videos" [2022-09] [ICML 2023] [[paper](http://arxiv.org/abs/2209.15404)]

- "Progressive Fourier Neural Representation for Sequential Video Compilation" [2023-06] [ICLR 2024] [[paper](http://arxiv.org/abs/2306.11305)]

- "Hierarchical Spatio-Temporal Representation Learning for Gait Recognition" [2023-07] [ICCV 2023] [[paper](http://arxiv.org/abs/2307.09856)]

- "Is Overfitting Necessary for Implicit Video Representation?" [2023-08] [ICML 2023] [[paper](https://proceedings.mlr.press/v202/choi23b.html)]

- "Video-LLaVA: Learning United Visual Representation by Alignment Before Projection" [2023-11] [EMNLP 2024] [[paper](http://arxiv.org/abs/2311.10122)]

- "Multi-view Masked Contrastive Representation Learning for Endoscopic Video Analysis" [2024-03] [NeurIPS 2024] [[paper](http://papers.nips.cc/paper_files/paper/2024/hash/55cb562b1f5af71f6707f3ff3c7941e6-Abstract-Conference.html)]

- "MaskCLR: Attention-Guided Contrastive Learning for Robust Action Representation Learning" [2024-03] [CVPR 2024] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Abdelfattah_MaskCLR_Attention-Guided_Contrastive_Learning_for_Robust_Action_Representation_Learning_CVPR_2024_paper.pdf)]

- "Combining Frame and GOP Embeddings for Neural Video Representation" [2024-03] [CVPR 2024] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Saethre_Combining_Frame_and_GOP_Embeddings_for_Neural_Video_Representation_CVPR_2024_paper.pdf)]

- "ARVideo: Autoregressive Pretraining for Self-Supervised Video Representation Learning" [2024-05] [TMLR 2025] [[paper](https://arxiv.org/abs/2405.15160)]

- "Visual Representation Learning with Stochastic Frame Prediction" [2024-06] [ICML 2024] [[paper](http://arxiv.org/abs/2406.07398)]

- "VEDIT: Latent Prediction Architecture For Procedural Video Representation Learning" [2024-10] [ICLR 2025] [[paper](http://arxiv.org/abs/2410.03478)]

- "SEAL: SEmantic Attention Learning for Long Video Representation" [2024-12] [CVPR 2025] [[paper](https://arxiv.org/abs/2412.01798)]

- "REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder" [2025-03] [ICCV 2025] [[paper](http://arxiv.org/abs/2503.08665)]

- "Bootstrap Your Own Views: Masked Ego-Exo Modeling for Fine-grained View-invariant Video Representations" [2025-03] [CVPR 2025] [[paper](https://arxiv.org/abs/2503.19706)]

- "LV-MAE: Learning Long Video Representations through Masked-Embedding Autoencoders" [2025-04] [ICCV 2025] [[paper](http://arxiv.org/abs/2504.03501)]

#### Vision-Language Model

- "Universal Vision-Language Dense Retrieval: Learning A Unified Representation Space for Multi-Modal Retrieval" [2022-09] [ICLR 2023] [[paper](https://arxiv.org/abs/2209.00179)]

- "Incorporating Structured Representations into Pretrained Vision & Language Models Using Scene Graphs" [2023-05] [EMNLP 2023] [[paper](http://arxiv.org/abs/2305.06343)]

- "Improved Probabilistic Image-Text Representations" [2023-05] [ICLR 2024] [[paper](http://arxiv.org/abs/2305.18171)]

- "BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing" [2023-05] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2305.14720)]

- "Weakly Supervised Vision-and-Language Pre-training with Relative Representations" [2023-05] [ACL 2023] [[paper](http://arxiv.org/abs/2305.15483)]

- "Learning Mask-aware CLIP Representations for Zero-Shot Segmentation" [2023-09] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2310.00240)]

- "Achieving Cross Modal Generalization with Multimodal Unified Representation" [2023-10] [NeurIPS 2023] [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c89f09849eb5af489abb122394ff0f0b-Abstract-Conference.html)]

- "G2D: From Global to Dense Radiography Representation Learning via Vision-Language Pre-training" [2023-12] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2312.01522)]

- "PolCLIP: A Unified Image-Text Word Sense Disambiguation Model via Generating Multimodal Complementary Representations" [2024-01] [ACL 2024] [[paper](https://doi.org/10.18653/v1/2024.acl-long.575)]

- "Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models" [2024-02] [ICML 2024] [[paper](http://arxiv.org/abs/2402.12336)]

- "Infrared and Visible Image Fusion with Language-Driven Loss in CLIP Embedding Space" [2024-02] [ACM MM 2025] [[paper](http://arxiv.org/abs/2402.16267)]

- "CLIPLoss and Norm-Based Data Selection Methods for Multimodal Contrastive Learning" [2024-02] [NeurIPS 2024] [[paper](http://papers.nips.cc/paper_files/paper/2024/hash/1b33deb9e1e7c31f05300b1c2d4a4f7d-Abstract-Conference.html)]

- "Multilingual Diversity Improves Vision-Language Representations" [2024-02] [NeurIPS 2024] [[paper](http://papers.nips.cc/paper_files/paper/2024/hash/a6678e2be4ce7aef9d2192e03cd586b7-Abstract-Conference.html)]

- "Demonstrating and Reducing Shortcuts in Vision-Language Representation Learning" [2024-02] [TMLR 2024] [[paper](https://arxiv.org/abs/2402.17510)]

- "If CLIP Could Talk: Understanding Vision-Language Model Representations Through Their Preferred Concept Descriptions" [2024-03] [EMNLP 2024] [[paper](http://arxiv.org/abs/2403.16442)]

- "Cascade-CLIP: Cascaded Vision-Language Embeddings Alignment for Zero-Shot Semantic Segmentation" [2024-06] [ICML 2024] [[paper](http://arxiv.org/abs/2406.00670)]

- "OLIVE: Object Level In-Context Visual Embeddings" [2024-06] [ACL 2024] [[paper](http://arxiv.org/abs/2406.00872)]

- "VISTA: Visualized Text Embedding For Universal Multi-Modal Retrieval" [2024-06] [ACL 2024] [[paper](http://arxiv.org/abs/2406.04292)]

- "RWKV-CLIP: A Robust Vision-Language Representation Learner" [2024-06] [EMNLP 2024] [[paper](http://arxiv.org/abs/2406.06973)]

- "MATE: Meet At The Embedding - Connecting Images with Long Texts" [2024-06] [EMNLP 2024 Findings] [[paper](http://arxiv.org/abs/2407.09541)]

- "VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks" [2024-10] [ICLR 2025] [[paper](http://arxiv.org/abs/2410.05160)]

- "Interfacing Foundation Models' Embeddings" [2024-10] [NeurIPS 2024] [[paper](http://papers.nips.cc/paper_files/paper/2024/hash/46e3b98045760c8cd9a0728d9e9f158d-Abstract-Conference.html)]

- "MIRe: Enhancing Multimodal Queries Representation via Fusion-Free Modality Interaction for Multimodal Retrieval" [2024-11] [ACL 2025 Findings] [[paper](http://arxiv.org/abs/2411.08334)]

- "Verbalized Representation Learning for Interpretable Few-Shot Generalization" [2024-11] [ICCV 2025] [[paper](http://arxiv.org/abs/2411.18651)]

- "Beyond Logit Lens: Contextual Embeddings for Robust Hallucination Detection & Grounding in VLMs" [2024-11] [NAACL 2025] [[paper](http://arxiv.org/abs/2411.19187)]

- "End-to-end Training for Text-to-Image Synthesis using Dual-Text Embeddings" [2025-02] [TMLR 2025] [[paper](http://arxiv.org/abs/2502.01507)]

- "mmE5: Improving Multimodal Multilingual Embeddings via High-quality Synthetic Data" [2025-02] [ACL 2025 Findings] [[paper](http://arxiv.org/abs/2502.08468)]

- "CalliReader: Contextualizing Chinese Calligraphy via an Embedding-Aligned Vision-Language Model" [2025-03] [ICCV 2025] [[paper](http://arxiv.org/abs/2503.06472)]

- "LangBridge: Interpreting Image as a Combination of Language Embeddings" [2025-03] [ICCV 2025] [[paper](http://arxiv.org/abs/2503.19404)]

- "Not Only Text: Exploring Compositionality of Visual Representations in Vision-Language Models" [2025-03] [CVPR 2025] [[paper](https://arxiv.org/abs/2503.17142)]

- "BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models" [2025-08] [ICCV 2025] [[paper](http://arxiv.org/abs/2508.06895)]

- "ABC: Achieving Better Control of Visual Embeddings using VLLMs" [2025-09] [TMLR 2025] [[paper](https://openreview.net/forum?id=RezANmBpxW)]

#### Domain-Specific Model

- "Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning" [2022-12] [ICCV 2023] [[paper](http://arxiv.org/abs/2212.14532)]

- "Parametric Depth Based Feature Representation Learning for Object Detection and Segmentation in Bird's-Eye View" [2023-07] [ICCV 2023] [[paper](http://arxiv.org/abs/2307.04106)]

- "GEDepth: Ground Embedding for Monocular Depth Estimation" [2023-09] [ICCV 2023] [[paper](http://arxiv.org/abs/2309.09975)]

- "Rethinking Self-Supervised Visual Representation Learning in Pre-training for 3D Human Pose and Shape Estimation" [2023-09] [ICLR 2023] [[paper](https://arxiv.org/abs/2303.05370)]

- "Unsupervised Learning of Object-Centric Embeddings for Cell Instance Segmentation in Microscopy Images" [2023-10] [ICCV 2023] [[paper](http://arxiv.org/abs/2310.08501)]

- "Improving Representation Learning for Histopathologic Images with Cluster Constraints" [2023-10] [ICCV 2023] [[paper](http://arxiv.org/abs/2310.12334)]

- "Semantic-aware Representation Learning for Homography Estimation" [2024-03] [ACM MM 2024] [[paper](https://arxiv.org/abs/2407.13284)]

- "Learned Trajectory Embedding for Subspace Clustering" [2024-03] [CVPR 2024] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Lochman_Learned_Trajectory_Embedding_for_Subspace_Clustering_CVPR_2024_paper.pdf)]

- "Multi-Scale Representations by Varying Window Attention for Semantic Segmentation" [2024-04] [ICLR 2024] [[paper](http://arxiv.org/abs/2404.16573)]

- "2M-AF: A Strong Multi-Modality Framework For Human Action Quality Assessment with Self-supervised Representation Learning" [2024-04] [ACM MM 2024] [[paper](https://dl.acm.org/doi/10.1145/3664647.3681084)]

- "Masked Face Recognition with Generative-to-Discriminative Representations" [2024-05] [ICML 2024] [[paper](http://arxiv.org/abs/2405.16761)]

- "MemeCLIP: Leveraging CLIP Representations for Multimodal Meme Classification" [2024-06] [EMNLP 2024] [[paper](https://doi.org/10.18653/v1/2024.emnlp-main.959)]

- "DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation" [2024-06] [ICLR 2024] [[paper](https://openreview.net/forum?id=h1sFUGlI09)]

- "Self-Supervised Visual Representation Learning for Medical Image Analysis: A Comprehensive Survey" [2024-07] [TMLR 2024] [[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9455147/)]

- "Learning Uniformly Distributed Embedding Clusters of Stylistic Skills for Physically Simulated Characters" [2024-11] [ACM MM 2025] [[paper](http://arxiv.org/abs/2411.06459)]

- "2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification" [2024-12] [CVPR 2025] [[paper](https://arxiv.org/abs/2412.00678)]

- "Any2AnyTryon: Leveraging Adaptive Position Embeddings for Versatile Virtual Clothing Tasks" [2025-01] [ICCV 2025] [[paper](http://arxiv.org/abs/2501.15891)]

- "Contextual Gesture: Co-Speech Gesture Video Generation through Context-aware Gesture Representation" [2025-02] [ACM MM 2025] [[paper](http://arxiv.org/abs/2502.07239)]

- "Modeling Fine-Grained Hand-Object Dynamics for Egocentric Video Representation Learning" [2025-03] [ICLR 2025] [[paper](http://arxiv.org/abs/2503.00986)]

- "Parameter-Efficient Adaptation of Geospatial Foundation Models through Embedding Deflection" [2025-03] [ICCV 2025] [[paper](http://arxiv.org/abs/2503.09493)]

- "EditCLIP: Representation Learning for Image Editing" [2025-03] [ICCV 2025] [[paper](http://arxiv.org/abs/2503.20318)]

- "Easy-editable Image Vectorization with Multi-layer Multi-scale Distributed Visual Feature Embedding" [2025-03] [CVPR 2025] [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Easy-editable_Image_Vectorization_with_Multi-layer_Multi-scale_Distributed_Visual_Feature_Embedding_CVPR_2025_paper.pdf)]

- "MEDTalk: Multimodal Controlled 3D Facial Animation with Dynamic Emotions by Disentangled Embedding" [2025-07] [ACM MM 2025] [[paper](http://arxiv.org/abs/2507.06071)]

- "BrainFLORA: Uncovering Brain Concept Representation via Multimodal Neural Embeddings" [2025-07] [ACM MM 2025] [[paper](http://arxiv.org/abs/2507.09747)]

- "HAMLET-FFD: Hierarchical Adaptive Multi-modal Learning Embeddings Transformation for Face Forgery Detection" [2025-07] [ACM MM 2025] [[paper](http://arxiv.org/abs/2507.20913)]

- "Capturing More: Learning Multi-Domain Representations for Robust Online Handwriting Verification" [2025-08] [ACM MM 2025] [[paper](http://arxiv.org/abs/2508.01427)]

### 3.2 Training Methods

#### Contrastive & Self-Supervised Training

- "SemPPL: Predicting Pseudo-Labels for Better Contrastive Representations" [2023-01] [ICLR 2023] [[paper](http://arxiv.org/abs/2301.05158)]

- "RoPAWS: Robust Semi-supervised Representation Learning from Uncurated Data" [2023-02] [ICLR 2023] [[paper](http://arxiv.org/abs/2302.14483)]

- "Adaptive Similarity Bootstrapping for Self-Distillation Based Representation Learning" [2023-03] [ICCV 2023] [[paper](http://arxiv.org/abs/2303.13606)]

- "Soft Neighbors are Positive Supporters in Contrastive Visual Representation Learning" [2023-03] [ICLR 2023] [[paper](http://arxiv.org/abs/2303.17142)]

- "Connecting Multi-modal Contrastive Representations" [2023-05] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2305.14381)]

- "CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations" [2023-05] [ICML 2023] [[paper](http://arxiv.org/abs/2305.01118)]

- "Disambiguated Attention Embedding for Multi-Instance Partial-Label Learning" [2023-05] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2305.16912)]

- "Self-Supervised Set Representation Learning for Unsupervised Meta-Learning" [2023-05] [ICLR 2023] [[paper](https://iclr.cc/virtual/2023/poster/12007)]

- "Learning Fine-grained View-Invariant Representations from Unpaired Ego-Exo Videos via Temporal Alignment" [2023-06] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2306.05526)]

- "Patch-Level Contrasting without Patch Correspondence for Accurate and Dense Contrastive Representation Learning" [2023-06] [ICLR 2023] [[paper](http://arxiv.org/abs/2306.13337)]

- "Semantic Positive Pairs for Enhancing Visual Representation Learning of Instance Discrimination Methods" [2023-06] [TMLR 2024] [[paper](https://arxiv.org/abs/2306.16122)]

- "Hallucination Improves the Performance of Unsupervised Visual Representation Learning" [2023-07] [ICCV 2023] [[paper](http://arxiv.org/abs/2307.12168)]

- "StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners" [2023-07] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2306.00984)]

- "ArCL: Enhancing Contrastive Learning with Augmentation-Robust Representations" [2023-07] [ICLR 2023] [[paper](https://openreview.net/forum?id=n0Pb9T5kmb)]

- "Self-Weighted Contrastive Learning among Multiple Views for Mitigating Representation Degeneration" [2023-07] [NeurIPS 2023] [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/03b13b0db740b95cb741e007178ef5e5-Abstract-Conference.html)]

- "Motion-Guided Masking for Spatiotemporal Representation Learning" [2023-08] [ICCV 2023] [[paper](http://arxiv.org/abs/2308.12962)]

- "Time to augment self-supervised visual representation learning" [2023-08] [ICLR 2023] [[paper](https://openreview.net/forum?id=o8xdgmwCP8l)]

- "Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language" [2023-08] [ICML 2023] [[paper](https://proceedings.mlr.press/v202/baevski23a.html)]

- "Probabilistic Self-supervised Representation Learning via Scoring Rules Minimization" [2023-09] [ICLR 2024] [[paper](http://arxiv.org/abs/2309.02048)]

- "Multi-Object Representation Learning via Feature Connectivity and Object-Centric Regularization" [2023-09] [NeurIPS 2023] [[paper](https://neurips.cc/virtual/2023/poster/72499)]

- "Mosaic Representation Learning for Self-supervised Visual Pre-training" [2023-09] [ICLR 2023] [[paper](https://openreview.net/forum?id=JAezPMehaUu)]

- "Sub-token ViT Embedding via Stochastic Resonance Transformers" [2023-10] [ICML 2024] [[paper](http://arxiv.org/abs/2310.03967)]

- "Representation Learning via Consistent Assignment of Views over Random Partitions" [2023-10] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2310.12692)]

- "Combating Representation Learning Disparity with Geometric Harmonization" [2023-10] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2310.17622)]

- "Embedding Space Interpolation Beyond Mini-Batch, Beyond Pairs and Beyond Examples" [2023-10] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2311.05538)]

- "Progressively Compressed Auto-Encoder for Self-supervised Representation Learning" [2023-10] [ICLR 2023] [[paper](https://openreview.net/forum?id=8T4qmZbTkW7)]

- "Rejuvenating image-GPT as Strong Visual Representation Learners" [2023-12] [ICML 2024] [[paper](http://arxiv.org/abs/2312.02147)]

- "ProFeAT: Projected Feature Adversarial Training for Self-Supervised Learning of Robust Representations" [2024-01] [TMLR 2024] [[paper](https://arxiv.org/abs/2406.05796)]

- "Separating common from salient patterns with Contrastive Representation Learning" [2024-02] [ICLR 2024] [[paper](http://arxiv.org/abs/2402.11928)]

- "Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning" [2024-02] [NeurIPS 2024] [[paper](http://papers.nips.cc/paper_files/paper/2024/hash/04a80267ad46fc730011f8760f265054-Abstract-Conference.html)]

- "No Train, all Gain: Self-Supervised Gradients Improve Deep Frozen Representations" [2024-02] [NeurIPS 2024] [[paper](http://papers.nips.cc/paper_files/paper/2024/hash/1bf4cad47f5a54c98fbe7d10516ebf77-Abstract-Conference.html)]

- "Easy Regional Contrastive Learning of Expressive Fashion Representations" [2024-02] [NeurIPS 2024] [[paper](http://papers.nips.cc/paper_files/paper/2024/hash/2492288f6878e6f99124b362604e58f5-Abstract-Conference.html)]

- "LeOCLR: Leveraging Original Images for Contrastive Learning of Visual Representations" [2024-03] [TMLR 2024] [[paper](https://arxiv.org/abs/2403.06813)]

- "Targeted Representation Alignment for Open-World Semi-Supervised Learning" [2024-03] [CVPR 2024] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Targeted_Representation_Alignment_for_Open-World_Semi-Supervised_Learning_CVPR_2024_paper.pdf)]

- "D3still: Decoupled Differential Distillation for Asymmetric Image Retrieval" [2024-03] [CVPR 2024] [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xie_D3still_Decoupled_Differential_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2024_paper.pdf)]

- "Self-supervised Representation Learning from Random Data Projectors" [2024-05] [ICLR 2024] [[paper](https://openreview.net/forum?id=EpYnZpDpsQ)]

- "Contrasting Multiple Representations with the Multi-Marginal Matching Gap" [2024-05] [ICML 2024] [[paper](http://arxiv.org/abs/2405.19532)]

- "On the Role of Discrete Tokenization in Visual Representation Learning" [2024-07] [ICLR 2024] [[paper](http://arxiv.org/abs/2407.09087)]

- "T-MARS: Improving Visual Representations by Circumventing Text Feature Learning" [2024-08] [ICLR 2024] [[paper](https://openreview.net/forum?id=ViPtjIVzUw)]

- "AUC-CL: A Batchsize-Robust Framework for Self-Supervised Contrastive Representation Learning" [2024-08] [ICLR 2024] [[paper](https://openreview.net/forum?id=YgMdDQB09U)]

- "Hybrid Active Learning with Uncertainty-Weighted Embeddings" [2024-08] [TMLR 2024] [[paper](https://openreview.net/forum?id=jD761b5OaE)]

- "SCoRe: Submodular Combinatorial Representation Learning" [2024-09] [ICML 2024] [[paper](https://openreview.net/forum?id=G8zDeKOp0R)]

- "FreSh: Frequency Shifting for Accelerated Neural Representation Learning" [2024-10] [ICLR 2025] [[paper](http://arxiv.org/abs/2410.05050)]

- "On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning" [2024-10] [ICLR 2025] [[paper](http://arxiv.org/abs/2410.09156)]

- "When does perceptual alignment benefit vision representations?" [2024-10] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2410.10817)]

- "Learning predictable and robust neural representations by straightening image sequences" [2024-11] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2411.01777)]

- "One Leaf Reveals the Season: Occlusion-Based Contrastive Learning with Semantic-Aware Views for Efficient Visual Representation" [2024-11] [ICML 2025] [[paper](http://arxiv.org/abs/2411.09858)]

- "Self-supervised Transformation Learning for Equivariant Representations" [2025-02] [NeurIPS 2024] [[paper](https://doi.org/10.48550/arXiv.2501.08712)]

- "Implicit Contrastive Representation Learning with Guided Stop-gradient" [2025-03] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2503.09058)]

- "Self-Organizing Visual Prototypes for Non-Parametric Representation Learning" [2025-05] [ICML 2025] [[paper](http://arxiv.org/abs/2505.21533)]

- "MIM-Refiner: A Contrastive Learning Boost from Intermediate Pre-Trained Masked Image Modeling Representations" [2025-05] [ICLR 2025] [[paper](https://openreview.net/forum?id=0PxLpVURTl)]

- "Maximal Matching Matters: Preventing Representation Collapse for Robust Cross-Modal Retrieval" [2025-06] [ACL 2025] [[paper](https://arxiv.org/abs/2506.21538)]

- "Region-based Cluster Discrimination for Visual Representation Learning" [2025-07] [ICCV 2025] [[paper](http://arxiv.org/abs/2507.20025)]

- "CR2PQ: Continuous Relative Rotary Positional Query for Dense Visual Representation Learning" [2025-08] [ICLR 2025] [[paper](https://openreview.net/forum?id=3l6PwssLNY)]

- "LayerLock: Non-collapsing Representation Learning with Progressive Freezing" [2025-09] [ICCV 2025] [[paper](http://arxiv.org/abs/2509.10156)]

#### Disentanglement & Causality & Invariance

- "Weakly Supervised Disentangled Generative Causal Representation Learning" [2020-10] [ICML 2023] [[paper](http://arxiv.org/abs/2010.02637)]

- "Disentangling visual embeddings for attributes and objects" [2022-05] [CVPR 2022] [[paper](https://arxiv.org/abs/2205.08536)]

- "Simple Disentanglement of Style and Content in Visual Representations" [2023-02] [ICML 2023] [[paper](http://arxiv.org/abs/2302.09795)]

- "Leveraging sparse and shared feature activations for disentangled representation learning" [2023-04] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2304.07939)]

- "Learning Structured Representations by Embedding Class Hierarchy" [2023-05] [ICLR 2023] [[paper](https://openreview.net/forum?id=7J-30ilaUZM)]

- "Orthogonality-Enforced Latent Space in Autoencoders: An Approach to Learning Disentangled Representations" [2023-08] [ICML 2023] [[paper](https://proceedings.mlr.press/v202/cha23b.html)]

- "InfoDiffusion: Representation Learning Using Information Maximizing Diffusion Models" [2023-08] [ICML 2023] [[paper](https://proceedings.mlr.press/v202/wang23ah.html)]

- "Flow Factorized Representation Learning" [2023-09] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2309.13167)]

- "Learning to Receive Help: Intervention-Aware Concept Embedding Models" [2023-12] [NeurIPS 2023] [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/770cabd044c4eacb6dc5924d9a686dce-Abstract-Conference.html)]

- "Attribute-driven Disentangled Representation Learning for Multimodal Recommendation" [2023-12] [ACM MM 2024] [[paper](http://arxiv.org/abs/2312.14433)]

- "Exploring Diffusion Time-steps for Unsupervised Representation Learning" [2024-01] [ICLR 2024] [[paper](http://arxiv.org/abs/2401.11430)]

- "Explicitly Disentangled Representations in Object-Centric Learning" [2024-01] [TMLR 2025] [[paper](https://arxiv.org/abs/2401.10148)]

- "Unity by Diversity: Improved Representation Learning for Multimodal VAEs" [2024-02] [NeurIPS 2024] [[paper](http://papers.nips.cc/paper_files/paper/2024/hash/87726969ce38e9a676ca1fd4459ba77d-Abstract-Conference.html)]

- "Equilibrated Diffusion: Frequency-aware Textual Embedding for Equilibrated Image Customization" [2024-03] [ACM MM 2024] [[paper](https://dl.acm.org/doi/10.1145/3664647.3680729)]

- "Graph-based Unsupervised Disentangled Representation Learning via Multimodal Large Language Models" [2024-04] [NeurIPS 2024] [[paper](http://papers.nips.cc/paper_files/paper/2024/hash/bac4d92b3f6decfe47eab9a5893dd1f6-Abstract-Conference.html)]

- "Isometric Representation Learning for Disentangled Latent Space of Diffusion Models" [2024-07] [ICML 2024] [[paper](http://arxiv.org/abs/2407.11451)]

- "Towards the Causal Complete Cause of Multi-Modal Representation Learning" [2024-07] [ICML 2025] [[paper](http://arxiv.org/abs/2407.14058)]

- "Object centric architectures enable efficient causal representation learning" [2024-07] [ICLR 2024] [[paper](https://openreview.net/forum?id=r9FsiXZxZt)]

- "Imaginary-Connected Embedding in Complex Space for Unseen Attribute-Object Discrimination" [2024-10] [TPAMI 2024] [[paper](https://ieeexplore.ieee.org/document/10737702)]

- "Synergy Between Sufficient Changes and Sparse Mixing Procedure for Disentangled Representation Learning" [2025-03] [ICLR 2025] [[paper](http://arxiv.org/abs/2503.00639)]

- "Disentangled Embedding through Style and Mutual Information for Domain Generalization" [2025-07] [TMLR 2025] [[paper](https://openreview.net/forum?id=552tedTByb)]

#### Robustness, Privacy, Federated & Efficient Training

- "How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection?" [2022-03] [ICLR 2023] [[paper](http://arxiv.org/abs/2203.04450)]

- "k-Median Clustering via Metric Embedding: Towards Better Initialization with Differential Privacy" [2022-06] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2206.12895)]

- "L-DAWA: Layer-wise Divergence Aware Weight Aggregation in Federated Self-Supervised Visual Representation Learning" [2023-07] [ICCV 2023] [[paper](http://arxiv.org/abs/2307.07393)]

- "DenoiseRep: Denoising Model for Representation Learning" [2024-02] [NeurIPS 2024] [[paper](http://papers.nips.cc/paper_files/paper/2024/hash/46907c2ff9fafd618095161d76461842-Abstract-Conference.html)]

- "Differentially Private Representation Learning via Image Captioning" [2024-03] [ICML 2024] [[paper](http://arxiv.org/abs/2403.02506)]

- "Data-free Neural Representation Compression with Riemannian Neural Dynamics" [2024-09] [ICML 2024] [[paper](https://openreview.net/forum?id=LTifAl5bKb)]

- "Robustness Reprogramming for Representation Learning" [2024-10] [ICLR 2025] [[paper](http://arxiv.org/abs/2410.04577)]

- "BendVLM: Test-Time Debiasing of Vision-Language Embeddings" [2024-11] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2411.04420)]

- "Learning to Merge Tokens via Decoupled Embedding for Efficient Vision Transformers" [2024-12] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2412.10569)]

- "I0T: Embedding Standardization Method Towards Zero Modality Gap" [2024-12] [ACL 2025] [[paper](https://arxiv.org/abs/2412.14384)]

- "Gradient Extrapolation for Debiased Representation Learning" [2025-03] [ICCV 2025] [[paper](http://arxiv.org/abs/2503.13236)]

- "On the Importance of Gaussianizing Representations" [2025-05] [ICML 2025] [[paper](http://arxiv.org/abs/2505.00685)]

- "Geometry of Long-Tailed Representation Learning: Rebalancing Features for Skewed Distributions" [2025-05] [ICLR 2025] [[paper](https://openreview.net/forum?id=GySIAKEwtZ)]

- "TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance" [2025-07] [ICCV 2025] [[paper](http://arxiv.org/abs/2507.18192)]

- "Learning Along the Arrow of Time: Hyperbolic Geometry for Backward-Compatible Representation Learning" [2025-07] [ICML 2025] [[paper](https://doi.org/10.48550/arXiv.2506.05826)]

#### Semi, Weak, Few-Shot Supervision

- "Few-shot Adaptation to Distribution Shifts By Mixing Source and Target Embeddings" [2023-05] [ICML 2024] [[paper](http://arxiv.org/abs/2305.14521)]

- "An Embedding is Worth a Thousand Noisy Labels" [2025-06] [TMLR 2025] [[paper](https://openreview.net/forum?id=X3gSvQjShh)]

#### Multimodal Training

- "Universal multimodal representation for language understanding" [2023-01] [TPAMI 2023] [[paper](https://arxiv.org/abs/2301.03344)]

- "RLEG: Vision-Language Representation Learning with Diffusion-based Embedding Generation" [2023-03] [ICML 2023] [[paper](https://proceedings.mlr.press/v202/zhao23l.html)]

- "Enhancing Multimodal Unified Representations for Cross Modal Generalization" [2024-03] [ACL 2025 Findings] [[paper](http://arxiv.org/abs/2403.05168)]

- "Towards Cross-modal Backward-compatible Representation Learning for Vision-Language Models" [2024-05] [ICCV 2025] [[paper](http://arxiv.org/abs/2405.14715)]

- "Multimodal Physiological Signals Representation Learning via Multiscale Contrasting for Depression Recognition" [2024-06] [ACM MM 2024] [[paper](http://arxiv.org/abs/2406.16968)]

- "MERLIN: Multimodal Embedding Refinement via LLM-based Iterative Navigation for Text-Video Retrieval-Rerank Pipeline" [2024-07] [EMNLP 2024 Industry] [[paper](http://arxiv.org/abs/2407.12508)]

- "PSM: Learning Probabilistic Embeddings for Multi-scale Zero-shot Soundscape Mapping" [2024-08] [ACM MM 2024] [[paper](http://arxiv.org/abs/2408.07050)]

- "DySarl: Dynamic Structure-Aware Representation Learning for Multimodal Knowledge Graph Reasoning" [2024-08] [ACM MM 2024] [[paper](https://dl.acm.org/doi/10.1145/3664647.3681020)]

- "Deeply Fusing Semantics and Interactions for Item Representation Learning via Topology-driven Pre-training" [2024-08] [ACM MM 2024] [[paper](https://dl.acm.org/doi/10.1145/3664647.3681639)]

- "Retrieval-based Disentangled Representation Learning with Natural Language Supervision" [2024-08] [ICLR 2024] [[paper](https://openreview.net/forum?id=ZlQRiFmq7Y)]

- "From Vision to Audio and Beyond: A Unified Model for Audio-Visual Representation and Generation" [2024-09] [ICML 2024] [[paper](http://arxiv.org/abs/2409.19132)]

- "Diving Deep into the Motion Representation of Video-Text Models" [2024-09] [ACL 2024 Findings] [[paper](https://doi.org/10.18653/v1/2024.findings-acl.747)]

- "MB2C: Multimodal Bidirectional Cycle Consistency for Learning Robust Visual Neural Representations" [2024-10] [ACM MM 2024] [[paper](https://dl.acm.org/doi/10.1145/3664647.3681292)]

- "Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning" [2025-05] [ACL 2025] [[paper](https://arxiv.org/abs/2505.13628)]

- "Kernel-based Unsupervised Embedding Alignment for Enhanced Visual Representation in Vision-language Models" [2025-06] [ICML 2025] [[paper](http://arxiv.org/abs/2506.02557)]

- "DALR: Dual-level Alignment Learning for Multimodal Sentence Representation Learning" [2025-06] [ACL 2025 Findings] [[paper](http://arxiv.org/abs/2506.21096)]

### 3.3 Datasets & Benchmarks

- "How robust is unsupervised representation learning to distribution shift?" [2022-06] [ICLR 2023] [[paper](http://arxiv.org/abs/2206.08871)]

- "Self-supervised learning of Split Invariant Equivariant representations" [2023-02] [ICML 2023] [[paper](http://arxiv.org/abs/2302.10283)]

- "Unicom: Universal and Compact Representation Learning for Image Retrieval" [2023-02] [ICLR 2023] [[paper](https://openreview.net/forum?id=3YFDsSRSxB-)]

- "A Large-scale Study of Spatiotemporal Representation Learning with a New Benchmark on Action Recognition" [2023-03] [ICCV 2023] [[paper](http://arxiv.org/abs/2303.13505)]

- "PLIP: Language-Image Pre-training for Person Representation Learning" [2023-05] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2305.08386)]

- "Babel-ImageNet: Massively Multilingual Evaluation of Vision-and-Language Representations" [2023-06] [ACL 2024] [[paper](http://arxiv.org/abs/2306.08658)]

- "Cross-Domain Product Representation Learning for Rich-Content E-Commerce" [2023-08] [ICCV 2023] [[paper](http://arxiv.org/abs/2308.05550)]

- "PUG: Photorealistic and Semantically Controllable Synthetic Data for Representation Learning" [2023-08] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2308.03977)]

- "RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank" [2023-08] [ICML 2023] [[paper](https://proceedings.mlr.press/v202/garrido23a.html)]

- "Towards Universal Image Embeddings: A Large-Scale Dataset and Challenge for Generic Image Representations" [2023-09] [ICCV 2023] [[paper](http://arxiv.org/abs/2309.01858)]

- "FORB: A Flat Object Retrieval Benchmark for Universal Image Embedding" [2023-09] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2309.16249)]

- "SciMMIR: Benchmarking Scientific Multi-modal Information Retrieval" [2024-01] [ACL 2024 Findings] [[paper](http://arxiv.org/abs/2401.13478)]

- "Mapping the Multiverse of Latent Representations" [2024-02] [ICML 2024] [[paper](http://arxiv.org/abs/2402.01514)]

- "TorchSpatial: A Location Encoding Framework and Benchmark for Spatial Representation Learning" [2024-06] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2406.15658)]

- "HyperFace: Generating Synthetic Face Recognition Datasets by Exploring Face Embedding Hypersphere" [2024-11] [ICLR 2025] [[paper](http://arxiv.org/abs/2411.08470)]

- "Scendi Score: Prompt-Aware Diversity Evaluation via Schur Complement of CLIP Embeddings" [2024-12] [ICCV 2025] [[paper](http://arxiv.org/abs/2412.18645)]

- "SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings" [2025-02] [ACL 2025] [[paper](https://arxiv.org/abs/2502.12562)]

- "Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates" [2025-05] [ACL 2025] [[paper](https://arxiv.org/abs/2505.22943)]

- "On the Transfer of Object-Centric Representation Learning" [2025-05] [ICLR 2025] [[paper](https://openreview.net/forum?id=bSq0XGS3kW)]

- "Learning Fine-Grained Representations through Textual Token Disentanglement in Composed Video Retrieval" [2025-07] [ICLR 2025] [[paper](https://openreview.net/forum?id=wGa2plE8ka)]

- "Color Me Correctly: Bridging Perceptual Color Spaces and Text Embeddings for Improved Diffusion Generation" [2025-09] [ACM MM 2025] [[paper](http://arxiv.org/abs/2509.10058)]

### 3.4 Embedding Analysis

#### Representation Properties & Mechanisms

- "Your Contrastive Learning Is Secretly Doing Stochastic Neighbor Embedding" [2022-05] [ICLR 2023] [[paper](http://arxiv.org/abs/2205.14814)]

- "Bag of Image Patch Embedding Behind the Success of Self-Supervised Learning" [2022-06] [TMLR 2023] [[paper](https://arxiv.org/abs/2206.08954)]

- "Learning Efficient Coding of Natural Images with Maximum Manifold Capacity Representations" [2023-03] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2303.03307)]

- "Neural Harmonics: Bridging Spectral Embedding and Matrix Completion in Self-Supervised Learning" [2023-05] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2305.19818)]

- "Are Neurons Actually Collapsed? On the Fine-Grained Structure in Neural Representations" [2023-06] [ICML 2023] [[paper](http://arxiv.org/abs/2306.17105)]

- "Improving neural network representations using human similarity judgments" [2023-06] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2306.04507)]

- "Is a Caption Worth a Thousand Images? A Study on Representation Learning" [2023-07] [ICLR 2023] [[paper](https://openreview.net/forum?id=cYijsVZhb5)]

- "ViLLA: Fine-Grained Vision-Language Representation Learning from Real-World Data" [2023-08] [ICCV 2023] [[paper](http://arxiv.org/abs/2308.11194)]

- "Analyzing Vision Transformers for Image Classification in Class Embedding Space" [2023-10] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2310.18969)]

- "RanDumb: Random Representations Outperform Online Continually Learned Representations" [2024-02] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2402.08823)]

- "Embedding Dimension of Contrastive Learning and k-Nearest Neighbors" [2024-02] [NeurIPS 2024] [[paper](http://papers.nips.cc/paper_files/paper/2024/hash/487c9d6ef55e73aa9dfd4b48fe3713a6-Abstract-Conference.html)]

- "A representation-learning game for classes of prediction tasks" [2024-03] [ICLR 2024] [[paper](http://arxiv.org/abs/2403.06971)]

- "Weighted Point Set Embedding for Multimodal Contrastive Learning Toward Optimal Similarity Metric" [2024-04] [ICLR 2025] [[paper](http://arxiv.org/abs/2404.19228)]

- "Transport of Algebraic Structure to Latent Embeddings" [2024-05] [ICML 2024] [[paper](http://arxiv.org/abs/2405.16763)]

- "Decomposing and Interpreting Image Representations via Text in ViTs Beyond CLIP" [2024-06] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2406.01583)]

- "Preserving Pre-trained Representation Space: On Effectiveness of Prefix-tuning for Large Multi-modal Models" [2024-06] [EMNLP 2024 Findings] [[paper](https://doi.org/10.18653/v1/2024.findings-emnlp.44)]

- "Prompt-Softbox-Prompt: A Free-Text Embedding Control for Image Editing" [2024-08] [ACM MM 2025] [[paper](http://arxiv.org/abs/2408.13623)]

- "Intriguing Properties of Hyperbolic Embeddings in Vision-Language Models" [2024-08] [TMLR 2024] [[paper](https://openreview.net/forum?id=P5D2gfi4Gg)]

- "Implicit Neural Representations and the Algebra of Complex Wavelets" [2024-08] [ICLR 2024] [[paper](https://openreview.net/forum?id=uZfjFyPAvn)]

- "A Cat Is A Cat (Not A Dog!): Unraveling Information Mix-ups in Text-to-Image Encoders through Causal Analysis and Embedding Optimization" [2024-10] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2410.00321)]

- "Semantic Token Reweighting for Interpretable and Controllable Text Embeddings in CLIP" [2024-10] [EMNLP 2024 Findings] [[paper](http://arxiv.org/abs/2410.08469)]

- "Investigating the Benefits of Projection Head for Representation Learning" [2024-10] [ICLR 2024] [[paper](https://openreview.net/forum?id=GgEAdqYPNA)]

- "Narrowing Information Bottleneck Theory for Multimodal Image-Text Representations Interpretability" [2025-02] [ICLR 2025] [[paper](http://arxiv.org/abs/2502.14889)]

- "Generalization Guarantees for Representation Learning via Data-Dependent Gaussian Mixture Priors" [2025-02] [ICLR 2025] [[paper](http://arxiv.org/abs/2502.15540)]

- "Feature Learning beyond the Lazy-Rich Dichotomy: Insights from Representational Geometry" [2025-03] [ICML 2025] [[paper](http://arxiv.org/abs/2503.18114)]

- "A Unifying Framework for Representation Learning" [2025-05] [ICLR 2025] [[paper](https://openreview.net/forum?id=WfaQrKCr4X)]

- "On the Similarities of Embeddings in Contrastive Learning" [2025-06] [ICML 2025] [[paper](http://arxiv.org/abs/2506.09781)]

- "Disappearance of Timestep Embedding: A Case Study on Neural ODE and Diffusion Models" [2025-06] [TMLR 2025] [[paper](https://openreview.net/forum?id=bpaLYaf6Dp)]

- "Discovering Divergent Representations between Text-to-Image Models" [2025-09] [ICCV 2025] [[paper](http://arxiv.org/abs/2509.08940)]

#### Robustness & Human Alignment

- "Self-supervised video pretraining yields robust and more human-aligned visual representations" [2022-10] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2210.06433)]

- "Embedding an Ethical Mind: Aligning Text-to-Image Synthesis via Lightweight Value Optimization" [2024-10] [ACM MM 2024] [[paper](http://arxiv.org/abs/2410.12700)]

- "PRISM: Reducing Spurious Implicit Biases in Vision-Language Models with LLM-Guided Embedding Projection" [2025-07] [ICCV 2025] [[paper](http://arxiv.org/abs/2507.08979)]

#### Interpretability

- "Identifying Interpretable Subspaces in Image Representations" [2023-08] [ICML 2023] [[paper](https://proceedings.mlr.press/v202/kalibhat23a.html)]

- "Interpreting CLIP's Image Representation via Text-Based Decomposition" [2023-10] [ICLR 2024] [[paper](http://arxiv.org/abs/2310.05916)]

- "Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE)" [2024-02] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2402.10376)]

- "Finding NEM-U: Explaining unsupervised representation learning through neural network generated explanation masks" [2024-09] [ICML 2024] [[paper](https://openreview.net/forum?id=Hzpt1Gws9g)]

- "AKRMap: Adaptive Kernel Regression for Trustworthy Visualization of Cross-Modal Embeddings" [2025-05] [ICML 2025] [[paper](http://arxiv.org/abs/2505.14664)]

- "Enhancing Pre-trained Representation Classifiability can Boost its Interpretability" [2025-10] [ICLR 2025] [[paper](http://arxiv.org/abs/2510.24105)]

### 3.5 Embedding Applications

#### Segmentation & Detection

- "Referring Image Segmentation via Joint Mask Contextual Embedding Learning and Progressive Alignment Network" [2023-04] [EMNLP 2023] [[paper](https://doi.org/10.18653/v1/2023.emnlp-main.481)]

- "Cross Paradigm Representation and Alignment Transformer for Image Deraining" [2025-04] [ACM MM 2025] [[paper](http://arxiv.org/abs/2504.16455)]

#### Retrieval & Reranking

- "TempCLR: Temporal Alignment Representation with Contrastive Learning" [2023-07] [ICLR 2023] [[paper](https://openreview.net/forum?id=CIFOsnhZvON)]

- "Unifying Multimodal Retrieval via Document Screenshot Embedding" [2024-06] [EMNLP 2024] [[paper](http://arxiv.org/abs/2406.11251)]

- "CLaMP 2: Multimodal Music Information Retrieval Across 101 Languages Using Large Language Models" [2024-10] [NAACL 2025 Findings] [[paper](http://arxiv.org/abs/2410.13267)]

- "Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings" [2025-06] [ACL 2025 Findings] [[paper](http://arxiv.org/abs/2506.04997)]

- "Modeling Uncertainty in Composed Image Retrieval via Probabilistic Embeddings" [2025-07] [ACL 2025] [[paper](https://aclanthology.org/2025.acl-long.61/)]

- "Queries Are Not Alone: Clustering Text Embeddings for Video Search" [2025-10] [SIGIR 2025] [[paper](https://arxiv.org/abs/2510.07720)]

#### Identification & Surveillance

- "Identity-Seeking Self-Supervised Representation Learning for Generalizable Person Re-Identification" [2023-08] [ICCV 2023] [[paper](http://arxiv.org/abs/2308.08887)]

- "Camera-Driven Representation Learning for Unsupervised Domain Adaptive Person Re-identification" [2023-08] [ICCV 2023] [[paper](http://arxiv.org/abs/2308.11901)]

- "Learning Continual Compatible Representation for Re-indexing Free Lifelong Person Re-identificationç" [2024-03] [CVPR 2024] [[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Cui_Learning_Continual_Compatible_Representation_for_Re-indexing_Free_Lifelong_Person_Re-identification_CVPR_2024_paper.html)]

#### Autonomous Driving

- "Unsupervised Self-Driving Attention Prediction via Uncertainty Mining and Knowledge Embedding" [2023-03] [ICCV 2023] [[paper](http://arxiv.org/abs/2303.09706)]

- "FreqPDE: Rethinking Positional Depth Embedding for Multi-View 3D Object Detection Transformers" [2025-10] [ICCV 2025] [[paper](http://arxiv.org/abs/2510.15385)]

#### Enhancement & Generation & Editing

- "StegaNeRF: Embedding Invisible Information within Neural Radiance Fields" [2022-12] [ICCV 2023] [[paper](http://arxiv.org/abs/2212.01602)]

- "DiffV2S: Diffusion-Based Video-to-Speech Synthesis with Vision-Guided Speaker Embedding" [2023-08] [ICCV 2023] [[paper](http://arxiv.org/abs/2308.07787)]

- "GAN-based Symmetric Embedding Costs Adjustment for Enhancing Image Steganographic Security" [2024-03] [ACM MM 2024] [[paper](https://dl.acm.org/doi/10.1145/3664647.3681311)]

- "CLIPAway: Harmonizing focused embeddings for removing objects via diffusion models" [2024-06] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2406.09368)]

- "Addressing Text Embedding Leakage in Diffusion-based Image Editing" [2024-12] [ICCV 2025] [[paper](http://arxiv.org/abs/2412.04715)]

- "DRC: Enhancing Personalized Image Generation via Disentangled Representation Composition" [2025-04] [ACM MM 2025] [[paper](http://arxiv.org/abs/2504.17349)]

- "StereoINR: Cross-View Geometry Consistent Stereo Super Resolution with Implicit Neural Representation" [2025-05] [ACM MM 2025] [[paper](http://arxiv.org/abs/2505.05509)]

- "LightBSR: Towards Lightweight Blind Super-Resolution via Discriminative Implicit Degradation Representation Learning" [2025-06] [ICCV 2025] [[paper](http://arxiv.org/abs/2506.22710)]

- "LotteryCodec: Searching the Implicit Representation in a Random Network for Low-Complexity Image Compression" [2025-07] [ICML 2025] [[paper](http://arxiv.org/abs/2507.01204)]

- "Text Embedding Knows How to Quantize Text-Guided Diffusion Models" [2025-07] [ICCV 2025] [[paper](http://arxiv.org/abs/2507.10340)]

- "Translation of Text Embedding via Delta Vector to Suppress Strongly Entangled Content in Text-to-Image Diffusion Models" [2025-08] [ICCV 2025] [[paper](http://arxiv.org/abs/2508.10407)]

#### Others

- "Multi-Level Information Retrieval Augmented Generation for Knowledge-based Visual Question Answering" [2024-06] [EMNLP 2024] [[paper](https://doi.org/10.18653/v1/2024.emnlp-main.922)]

- "Autogenic Language Embedding for Coherent Point Tracking" [2024-07] [ACM MM 2024] [[paper](http://arxiv.org/abs/2407.20730)]

- "Hierarchical Visual Categories Modeling: A Joint Representation Learning and Density Estimation Framework for Out-of-Distribution Detection" [2024-08] [ICCV 2023] [[paper](http://arxiv.org/abs/2408.15580)]

- "Dual Advancement of Representation Learning and Clustering for Sparse and Noisy Images" [2024-09] [ACM MM 2024] [[paper](http://arxiv.org/abs/2409.01781)]

- "DeSPITE: Exploring Contrastive Deep Skeleton-Pointcloud-IMU-Text Embeddings for Advanced Point Cloud Human Activity Understanding" [2025-06] [ICCV 2025] [[paper](http://arxiv.org/abs/2506.13897)]

- "Multimodal Invariant Sentiment Representation Learning" [2025-07] [ACL 2025 Findings] [[paper](https://aclanthology.org/2025.findings-acl.761/)]

- "GT-Loc: Unifying When and Where in Images through a Joint Embedding Space" [2025-07] [ICCV 2025] [[paper](http://arxiv.org/abs/2507.10473)]

- "MiraGe: Multimodal Discriminative Representation Learning for Generalizable AI-Generated Image Detection" [2025-08] [ACM MM 2025] [[paper](http://arxiv.org/abs/2508.01525)]

- "DRKF: Decoupled Representations with Knowledge Fusion for Multimodal Emotion Recognition" [2025-08] [ACM MM 2025] [[paper](http://arxiv.org/abs/2508.01644)]

## 4. Audio Embedding

### 4.1 Models

#### Speaker Embedding

- "High-Resolution Embedding Extractor for Speaker Diarisation" [2022-11] [ICASSP 2023] [[paper](http://arxiv.org/abs/2211.04060)]

- "ECAPA++: Fine-grained Deep Embedding Learning for TDNN Based Speaker Verification" [2023-05] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-777)]

- "Frame-Wise and Overlap-Robust Speaker Embeddings for Meeting Diarization" [2023-06] [ICASSP 2023] [[paper](http://arxiv.org/abs/2306.00625)]

- "Ordered and Binary Speaker Embedding" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1565)]

- "Contrastive Speaker Embedding With Sequential Disentanglement" [2023-09] [ICASSP 2024] [[paper](http://arxiv.org/abs/2309.13253)]

- "Multi-View Speaker Embedding Learning for Enhanced Stability and Discriminability" [2024-05] [ICASSP 2024] [[paper](https://doi.org/10.1109/ICASSP48485.2024.10448494)]

- "SVSNet+: Enhancing Speaker Voice Similarity Assessment Models with Representations from Speech Foundation Models" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.08445)]

- "Residual Speaker Representation for One-Shot Voice Conversion" [2024-09] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2309.08166)]

- "Guided Speaker Embedding" [2024-10] [ICASSP 2025] [[paper](http://arxiv.org/abs/2410.12182)]

- "SEED: Speaker Embedding Enhancement Diffusion Model" [2025-05] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2505.16798)]

- "Codec-ASV: Exploring Neural Audio Codec For Speaker Representation Learning" [2025-07] [ICASSP 2025] [[paper](https://doi.org/10.1109/ICASSP49660.2025.10888177)]

- "Diarization-Guided Multi-Speaker Embeddings" [2025-09] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-1807)]

#### General Speech Representation

- "Robust Data2VEC: Noise-Robust Speech Representation Learning for ASR by Combining Regression and Improved Contrastive Learning" [2022-10] [ICASSP 2023] [[paper](http://arxiv.org/abs/2210.15324)]

- "An Improved Optimal Transport Kernel Embedding Method with Gating Mechanism for Singing Voice Separation and Speaker Identification" [2023-01] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10096651)]

- "DinoSR: Self-Distillation and Online Clustering for Self-supervised Speech Representation Learning" [2023-05] [NeurIPS 2023] [[paper](http://arxiv.org/abs/2305.10005)]

- "Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation" [2023-05] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2305.14079)]

- "MT4SSL: Boosting Self-Supervised Speech Representation Learning by Integrating Multiple Targets" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-822)]

- "Towards Effective and Compact Contextual Representation for Conformer Transducer Speech Recognition Systems" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.13307)]

- "Self-Supervised Acoustic Word Embedding Learning via Correspondence Transformer Encoder" [2023-07] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2307.09871)]

- "CoBERT: Self-Supervised Speech Representation Learning Through Code Representation Learning" [2023-08] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2210.04062)]

- "RepCodec: A Speech Representation Codec for Speech Tokenization" [2023-08] [ACL 2024] [[paper](http://arxiv.org/abs/2309.00169)]

- "Unsupervised Learning of Discrete Latent Representations with Data-Adaptive Dimensionality from Continuous Speech Streams" [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/interspeech.2023-1321)]

- "Self-supervised Neural Factor Analysis for Disentangling Utterance-level Speech Representations" [2023-08] [ICML 2023] [[paper](https://proceedings.mlr.press/v202/lin23e.html)]

- "EnCodecMAE: leveraging neural codecs for universal audio representation learning" [2023-09] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2309.07391)]

- "Audio Barlow Twins: Self-Supervised Audio Representation Learning" [2023-09] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10095041)]

- "R-Spin: Efficient Speaker and Noise-invariant Representation Learning with Acoustic Pieces" [2023-11] [NAACL 2024] [[paper](http://arxiv.org/abs/2311.09117)]

- "Spoken Word2Vec: Learning Skipgram Embeddings from Speech" [2023-11] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2311.09319)]

- "Enc-Dec RNN Acoustic Word Embeddings learned via Pairwise Prediction" [2023-11] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-483)]

- "Language-Codec: Bridging Discrete Codec Representations and Speech Language Models" [2024-02] [ACL 2025] [[paper](http://arxiv.org/abs/2402.12208)]

- "Audio Mamba: Selective State Spaces for Self-Supervised Audio Representations" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.02178)]

- "MS-HuBERT: Mitigating Pre-training and Inference Mismatch in Masked Language Modelling methods for learning Speech Representations" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.05661)]

- "MMM: Multi-Layer Multi-Residual Multi-Stream Discrete Speech Representation from Self-supervised Learning Model" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.09869)]

- "AxLSTMs: learning self-supervised audio representations with xLSTMs" [2024-08] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2408.16568)]

- "Compositional Audio Representation Learning" [2024-09] [ICASSP 2025] [[paper](http://arxiv.org/abs/2409.09619)]

- "Sylber: Syllabic Embedding Representation of Speech from Raw Audio" [2024-10] [ICLR 2025] [[paper](http://arxiv.org/abs/2410.07168)]

- "UniWav: Towards Unified Pre-training for Speech Representation Learning and Generation" [2025-03] [ICLR 2025] [[paper](http://arxiv.org/abs/2503.00733)]

- "LiSTEN: Learning Soft Token Embeddings for Neural Audio LLMs" [2025-05] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2505.18517)]

- "Spectrotemporal Modulation: Efficient and Interpretable Feature Representation for Classifying Speech, Music, and Environmental Sounds" [2025-05] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2505.23509)]

- "Representing Speech Through Autoregressive Prediction of Cochlear Tokens" [2025-08] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2508.11598)]

#### Speech Content, Phoneme & Articulatory Representation

- "Learning to Compute the Articulatory Representations of Speech with the MIRRORNET" [2022-10] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2210.16454)]

- "Improving Bilingual TTS Using Language And Phonology Embedding With Embedding Strength Modulator" [2022-12] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2212.03435)]

- "TranUSR: Phoneme-to-word Transcoder Based Unified Speech Representation Learning for Cross-lingual Speech Recognition" [2023-05] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2305.13629)]

- "XPhoneBERT: A Pre-trained Multilingual Model for Phoneme Representations for Text-to-Speech" [2023-05] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2305.19709)]

- "Deep Speech Synthesis from MRI-Based Articulatory Representations" [2023-07] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2307.02471)]

- "Discovering Phonetic Feature Event Patterns in Transformer Embeddings" [2023-10] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1985)]

- "Are Articulatory Feature Overlaps Shrouded in Speech Embeddings?" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-1039)]

- "SingOMD: Singing Oriented Multi-resolution Discrete Representation Construction from Speech Models" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.08905)]

- "Neurodyne: Neural Pitch Manipulation with Representation Learning and Cycle-Consistency GAN" [2025-05] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2505.15368)]

- "Binary Representation Learning for Discriminative Acoustic Unit Discovery" [2025-07] [ICASSP 2025] [[paper](https://doi.org/10.1109/ICASSP49660.2025.10889906)]

- "Learning Optimal Prosody Embedding Codebook based on F0 and Energy" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-1020)]

- "ASDA: Audio Spectrogram Differential Attention Mechanism for Self-Supervised Representation Learning" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.48550/arXiv.2507.02666)]

#### Emotion, Paralinguistic & Prosody Embedding

- "Self-FiLM: Conditioning GANs with self-supervised representations for bandwidth extension based speaker recognition" [2023-01] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-2031)]

- "Learning Representation of Therapist Empathy in Counseling Conversation Using Siamese Hierarchical Attention Network" [2023-05] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2305.16690)]

- "FusedF0: Improving DNN-based F0 Estimation by Fusion of Summary-Correlograms and Raw Waveform Representations of Speech Signals" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-2229)]

- "Improved Contextualized Speech Representations for Tonal Analysis" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-283)]

- "Speech Synthesis with Self-Supervisedly Learnt Prosodic Representations" [2023-08] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2307.05132)]

- "Revealing Emotional Clusters in Speaker Embeddings: A Contrastive Learning Strategy for Speech Emotion Recognition" [2024-01] [ICASSP 2024] [[paper](http://arxiv.org/abs/2401.11017)]

- "Speech-Driven Emotional 3d Talking Face Animation Using Emotional Embeddings" [2024-01] [ICASSP 2024] [[paper](https://doi.org/10.1109/ICASSP48485.2024.10446842)]

- "Adaptive Speech Emotion Representation Learning Based On Dynamic Graph" [2024-05] [ICASSP 2024] [[paper](http://arxiv.org/abs/2405.03956)]

- "emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation" [2024-06] [ACL 2024 Findings] [[paper](https://doi.org/10.18653/v1/2024.findings-acl.931)]

- "Emotion-Aware Speech Self-Supervised Representation Learning with Intensity Knowledge" [2024-09] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-2518)]

- "EmoSphere-SER: Enhancing Speech Emotion Recognition Through Spherical Representation with Auxiliary Classification" [2025-05] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2505.19693)]

- "HYFuse: Aligning Heterogeneous Speech Pre-Trained Representations in Hyperbolic Space for Speech Emotion Recognition" [2025-06] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2506.03403)]

- "MATER: Multi-level Acoustic and Textual Emotion Representation for Interpretable Speech Emotion Recognition" [2025-06] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2506.19887)]

- "SupraDoRAL: Automatic Word Prominence Detection Using Suprasegmental Dependencies of Representations with Acoustic and Linguistic Context" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-2519)]

#### Multilingual Embedding

- "DistilXLSR: A Light Weight Cross-Lingual Speech Representation Model" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.01303)]

- "DSE-TTS: Dual Speaker Embedding for Cross-Lingual Text-to-Speech" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.14145)]

- "Conformer-based Language Embedding with Self-Knowledge Distillation for Spoken Language Identification" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1557)]

- "Language-Universal Phonetic Representation in Multilingual Speech Pretraining for Low-Resource Speech Recognition" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-617)]

- "MUST&P-SRL: Multi-lingual and Unified Syllabification in Text and Phonetic Domains for Speech Representation Learning" [2023-10] [EMNLP 2023 Industry] [[paper](http://arxiv.org/abs/2310.11541)]

- "Wave to Interlingua: Analyzing Representations of Multilingual Speech Transformers for Spoken Language Translation" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-2109)]

- "AfriHuBERT: A self-supervised speech representation model for African languages" [2024-09] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2409.20201)]

#### Multimodal

- "ImagineNet: Target Speaker Extraction with Intermittent Visual Cue Through Embedding Inpainting" [2022-10] [ICASSP 2023] [[paper](http://arxiv.org/abs/2211.00109)]

- "Jointly Learning Visual and Auditory Speech Representations from Raw Data" [2022-12] [ICLR 2023] [[paper](http://arxiv.org/abs/2212.06246)]

- "Continuous Interaction with A Smart Speaker via Low-Dimensional Embeddings of Dynamic Hand Pose" [2023-02] [ICASSP 2023] [[paper](http://arxiv.org/abs/2302.14566)]

- "ModEFormer: Modality-Preserving Embedding for Audio-Video Synchronization using Transformers" [2023-03] [ICASSP 2023] [[paper](http://arxiv.org/abs/2303.11551)]

- "ChatGPT-EDSS: Empathetic Dialogue Speech Synthesis Trained from ChatGPT-derived Context Word Embeddings" [2023-05] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2305.13724)]

- "MIR-GAN: Refining Frame-Level Modality-Invariant Representations with Adversarial Network for Audio-Visual Speech Recognition" [2023-06] [ACL 2023] [[paper](http://arxiv.org/abs/2306.10567)]

- "Fusion of Audio and Visual Embeddings for Sound Event Localization and Detection" [2023-12] [ICASSP 2024] [[paper](http://arxiv.org/abs/2312.09034)]

- "EnCLAP: Combining Neural Audio Codec and Audio-Text Joint Embedding for Automated Audio Captioning" [2024-01] [ICASSP 2024] [[paper](http://arxiv.org/abs/2401.17690)]

- "Invariant Motion Representation Learning for 3D Talking Face Synthesis" [2024-01] [ICASSP 2024] [[paper](https://doi.org/10.1109/ICASSP48485.2024.10446379)]

- "M2D-CLAP: Masked Modeling Duo Meets CLAP for Learning General-purpose Audio-Language Representation" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.02032)]

- "Audio-text Retrieval with Transformer-based Hierarchical Alignment and Disentangled Cross-modal Representation" [2024-09] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2409.09256)]

- "Learning Spatially-Aware Language and Audio Embeddings" [2024-09] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2409.11369)]

- "XLAVS-R: Cross-Lingual Audio-Visual Speech Representation Learning for Noise-Robust Speech Perception" [2024-10] [ACL 2024] [[paper](https://doi.org/10.18653/v1/2024.acl-long.697)]

- "Vela: Scalable Embeddings with Voice Large Language Models for Multimodal Retrieval" [2025-06] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2506.14445)]

- "SKE-MSA: Enhancing Representation Learning with VAD Lexicon for Multimodal Sentiment Analysis" [2025-08] [ICASSP 2025] [[paper](https://doi.org/10.1109/ICASSP49660.2025.10890333)]

### 4.2 Training Methods

#### Disentanglement & Decoupling

- "Disentangling Speech from Surroundings with Neural Embeddings" [2022-03] [ICASSP 2023] [[paper](http://arxiv.org/abs/2203.15578)]

- "CCSRD: Content-Centric Speech Representation Disentanglement Learning for End-to-End Speech Translation" [2023-04] [EMNLP 2023 Findings] [[paper](https://doi.org/10.18653/v1/2023.findings-emnlp.394)]

- "Mutual Information-based Embedding Decoupling for Generalizable Speaker Verification" [2023-04] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1314)]

- "Self-supervised Fine-tuning for Improved Content Representations by Speaker-invariant Clustering" [2023-05] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2305.11072)]

- "Disentangled Representation Learning for Multilingual Speaker Recognition" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1603)]

- "MT-SLVR: Multi-Task Self-Supervised Learning for Transformation In(Variant) Representations" [2023-07] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1064)]

- "Generalizable Zero-Shot Speaker Adaptive Speech Synthesis with Disentangled Representations" [2023-08] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2308.13007)]

- "Disentangled Representation Learning for Environment-agnostic Speaker Recognition" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-1124)]

- "Disentangling prosody and timbre embeddings via voice conversion" [2024-09] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-207)]

- "Universal Semantic Disentangled Privacy-preserving Speech Representation Learning" [2025-05] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2505.13085)]

- "DiEmo-TTS: Disentangled Emotion Representations via Self-Supervised Distillation for Cross-Speaker Emotion Transfer in Text-to-Speech" [2025-05] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2505.19687)]

- "HASRD: Hierarchical Acoustic and Semantic Representation Disentanglement" [2025-06] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2506.00843)]

#### Contrastive, Generative & Multi-Objective Learning

- "Spectral Clustering-Aware Learning of Embeddings for Speaker Diarisation" [2022-10] [ICASSP 2023] [[paper](http://arxiv.org/abs/2210.13576)]

- "Simultaneously Learning Robust Audio Embeddings and balanced Hash codes for Query-by-Example" [2022-11] [ICASSP 2023] [[paper](http://arxiv.org/abs/2211.11060)]

- "Contrastive Representation Learning for Acoustic Parameter Estimation" [2023-02] [ICASSP 2023] [[paper](http://arxiv.org/abs/2302.11205)]

- "Joint Generative-Contrastive Representation Learning for Anomalous Sound Detection" [2023-05] [ICASSP 2023] [[paper](http://arxiv.org/abs/2305.12111)]

- "Pushing the Limits of Unsupervised Unit Discovery for SSL Speech Representation" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.08920)]

- "Semantic Enrichment Towards Efficient Speech Representations" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-2234)]

- "ReCLR: Reference-Enhanced Contrastive Learning of Audio Representation for Depression Detection" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-2474)]

- "On The Effect Of Data-Augmentation On Local Embedding Properties In The Contrastive Learning Of Music Audio Representations" [2024-01] [ICASSP 2024] [[paper](http://arxiv.org/abs/2401.08889)]

- "Embedding Learning for Preference-based Speech Quality Assessment" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-1243)]

- "Articulatory synthesis using representations learnt through phonetic label-aware contrastive loss" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-1756)]

- "Refining Self-supervised Learnt Speech Representation using Brain Activations" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.08266)]

- "LASER: Learning by Aligning Self-supervised Representations of Speech for Improving Content-related Tasks" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.09153)]

- "Towards Robust Few-shot Class Incremental Learning in Audio Classification using Contrastive Representation" [2024-07] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2407.19265)]

- "Neural Compression Augmentation for Contrastive Audio Representation Learning" [2024-09] [InterSpeech 2024] [[paper](https://doi.org/10.21437/interspeech.2024-1156)]

- "REWIND: Speech Time Reversal for Enhancing Speaker Representations in Diffusion-based Voice Conversion" [2025-05] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2505.20756)]

- "InfoMin-based Query Embedding Optimization For Query-based Universal Sound Separation" [2025-07] [ICASSP 2025] [[paper](https://doi.org/10.1109/ICASSP49660.2025.10887870)]

- "Enhancing Target-speaker Automatic Speech Recognition Using Multiple Speaker Embedding Extractors with Virtual Speaker Embedding" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-2486)]

#### Distillation, Pruning & Efficiency-Oriented Training

- "Application of Knowledge Distillation to Multi-Task Speech Representation Learning" [2022-10] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2210.16611)]

- "Self-Supervised Speech Representation Learning for Keyword-Spotting With Light-Weight Transformers" [2023-01] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10095929)]

- "Masking Kernel for Learning Energy-Efficient Representations for Speaker Recognition and Mobile Health" [2023-02] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2302.04161)]

- "Automatic Data Augmentation for Domain Adapted Fine-Tuning of Self-Supervised Speech Representations" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.00481)]

- "Task-Agnostic Structured Pruning of Speech Representation Models" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.01385)]

- "On-Device Constrained Self-Supervised Speech Representation Learning for Keyword Spotting via Knowledge Distillation" [2023-07] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2307.02720)]

- "Knowledge Distillation from Self-Supervised Representation Learning Model with Discrete Speech Units for Any-to-Any Streaming Voice Conversion" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-924)]

- "DAISY: Data Adaptive Self-Supervised Early Exit for Speech Representation Models" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.05464)]

- "PRVAE-VC2: Non-Parallel Voice Conversion by Distillation of Speech Representations" [2024-09] [InterSpeech 2024] [[paper](http://arxiv.org/abs/1904.04631)]

- "EH-MAM: Easy-to-Hard Masked Acoustic Modeling for Self-Supervised Speech Representation Learning" [2024-10] [EMNLP 2024] [[paper](http://arxiv.org/abs/2410.13179)]

- "DuRep: Dual-Mode Speech Representation Learning via ASR-Aware Distillation" [2025-05] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2505.19774)]

- "Metric Learning with Progressive Self-Distillation for Audio-Visual Embedding Learning" [2025-07] [ICASSP 2025] [[paper](https://doi.org/10.1109/ICASSP49660.2025.10888698)]

#### Multilingual Training

- "Leveraging Language Embeddings for Cross-lingual Self-supervised Speech Representation Learning" [2023-01] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10096681)]

- "Acoustic Word Embeddings for Untranscribed Target Languages with Continued Pretraining and Learned Pooling" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.02153)]

- "Embedding Articulatory Constraints for Low-resource Speech Recognition Based on Large Pre-trained Model" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1437)]

- "Towards Robust Speech Representation Learning for Thousands of Languages" [2024-06] [EMNLP 2024] [[paper](http://arxiv.org/abs/2407.00837)]

- "Introducing Multilingual Phonetic Information to Speaker Embedding for Speaker Verification" [2024-09] [ICASSP 2024] [[paper](https://doi.org/10.1109/ICASSP48485.2024.10446546)]

#### Privacy-Preserving Representation Learning

- "Utility-Preserving Privacy-Enabled Speech Embeddings for Emotion Detection" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1075)]

- "Privacy-preserving Representation Learning for Speech Understanding" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-2138)]

- "On-Device Speaker Anonymization of Acoustic Embeddings for ASR based on Flexible Location Gradient Reversal Layer" [2023-07] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2307.13343)]

- "Asynchronous Voice Anonymization Using Adversarial Perturbation On Speaker Embedding" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.08200)]

- "Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation" [2025-05] [ACL 2025 Findings] [[paper](http://arxiv.org/abs/2505.19273)]

- "WavShape: Information-Theoretic Speech Representation Learning for Fair and Privacy-Aware Audio Processing" [2025-06] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2506.22789)]

- "Privacy-Preserving Speaker Verification via End-to-End Secure Representation Learning" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-1096)]

#### Robustness-Oriented Training

- "Learning Emotional Representations from Imbalanced Speech Data for Speech Emotion Recognition and Emotional Text-to-Speech" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.05709)]

- "Downstream Task Agnostic Speech Enhancement with Self-Supervised Representation Loss" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1578)]

- "Don’t Stop Self-Supervision: Accent Adaptation of Speech Representations via Residual Adapters" [2023-07] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2307.00453)]

- "Rethinking Session Variability: Leveraging Session Embeddings for Session Robustness in Speaker Verification" [2023-09] [ICASSP 2024] [[paper](http://arxiv.org/abs/2309.14741)]

- "Learning Repeatable Speech Embeddings Using An Intra-class Correlation Regularizer" [2023-10] [NeurIPS 2023] [[paper](http://arxiv.org/abs/2310.17049)]

- "CA-SSLR: Condition-Aware Self-Supervised Learning Representation for Generalized Speech Processing" [2024-02] [NeurIPS 2024] [[paper](http://papers.nips.cc/paper_files/paper/2024/hash/59a9cc95f046e9125d8816ef971873e7-Abstract-Conference.html)]

- "Real-time scheme for rapid extraction of speaker embeddings in challenging recording conditions" [2024-05] [InterSpeech 2024] [[paper](https://www.isca-archive.org/interspeech_2024/liu24s_interspeech.html)]

- "Tackling Missing Modalities in Audio-Visual Representation Learning Using Masked Autoencoders" [2024-09] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2505.14562)]

- "Balanced-Wav2Vec: Enhancing Stability and Robustness of Representation Learning Through Sample Reweighting Techniques" [2024-09] [InterSpeech 2024] [[paper](https://doi.org/10.21437/interspeech.2024-1875)]

- "Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation" [2025-01] [ICLR 2025] [[paper](http://arxiv.org/abs/2504.18539)]

- "Mitigating Non-Target Speaker Bias in Guided Speaker Embedding" [2025-06] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2506.12500)]

- "Inter- and Intra-Sentence Cuer-Invariant Representation Learning for Generalizable Cued Speech Recognition" [2025-07] [ICASSP 2025] [[paper](https://doi.org/10.1109/ICASSP49660.2025.10888246)]

- "Robust Target Speaker Diarization and Separation via Augmented Speaker Embedding Sampling" [2025-08] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2508.06393)]

- "Adaptive Across-Subcenter Representation Learning for Imbalanced Anomalous Sound Detection" [2025-09] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-1584)]

#### Other Methods

- "Semi-supervised Learning for Continuous Emotional Intensity Controllable Speech Synthesis with Disentangled Representations" [2022-11] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2211.06160)]

- "Adapting Self-Supervised Models to Multi-Talker Speech Recognition Using Speaker Embeddings" [2023-01] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10097139)]

- "Design Choices for Learning Embeddings from Auxiliary Tasks for Domain Generalization in Anomalous Sound Detection" [2023-01] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10097176)]

- "Incorporating Uncertainty from Speaker Embedding Estimation to Speaker Verification" [2023-02] [ICASSP 2023] [[paper](http://arxiv.org/abs/2302.11763)]

- "Transforming the Embeddings: A Lightweight Technique for Speech Emotion Recognition Tasks" [2023-03] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-2561)]

- "Label Aware Speech Representation Learning For Language Identification" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.04374)]

- "Emotion Label Encoding Using Word Embeddings for Speech Emotion Recognition" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1591)]

- "Towards Paralinguistic-Only Speech Representations for End-to-End Speech Emotion Recognition" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-497)]

- "Improving Joint Speech-Text Representations Without Alignment" [2023-08] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2308.06125)]

- "LABERT: A Combination of Local Aggregation and Self-Supervised Speech Representation Learning for Detecting Informative Hidden Units in Low-Resource ASR Systems" [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/interspeech.2023-2001)]

- "Dual Acoustic Linguistic Self-supervised Representation Learning for Cross-Domain Speech Recognition" [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/interspeech.2023-387)]

- "Text-Only Domain Adaptation for End-to-End Speech Recognition through Down-Sampling Acoustic Representation" [2023-09] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2309.02459)]

- "MixRep: Hidden Representation Mixup for Low-Resource Speech Recognition" [2023-10] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2310.18450)]

- "Adapter-tuning with Effective Token-dependent Representation Shift for Automatic Speech Recognition" [2023-11] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1221)]

- "Consistent and Relevant: Rethink the Query Embedding in General Sound Separation" [2023-12] [ICASSP 2024] [[paper](http://arxiv.org/abs/2312.15463)]

- "GR0: Self-Supervised Global Representation Learning for Zero-Shot Voice Conversion" [2024-03] [ICASSP 2024] [[paper](https://doi.org/10.1109/ICASSP48485.2024.10448232)]

- "ASTRA: Aligning Speech and Text Representations for Asr without Sampling" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-1924)]

- "Challenging margin-based speaker embedding extractors by using the variational information bottleneck" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.12622)]

- "Self-supervised learning of speech representations with Dutch archival data" [2025-07] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2507.04554)]

- "R2S: Real-to-Synthetic Representation Learning for Training Speech Recognition Models on Synthetic Data" [2025-09] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-1109)]

- "SiamCTC: Learning Speech Representations through Monotonic Temporal Alignment" [2025-09] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-2746)]

- "Towards Classification of Typical and Atypical Disfluencies: A Self Supervised Representation Approach" [2025-09] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-964)]

### 4.3 Embedding Analysis

#### Representation Properties

- "Evaluating context-invariance in unsupervised speech representations" [2022-10] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2210.15775)]

- "Perceptual Analysis of Speaker Embeddings for Voice Discrimination between Machine And Human Listening" [2023-01] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10094782)]

- "Analyzing Acoustic Word Embeddings from Pre-trained Self-supervised Models" [2023-01] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10096099)]

- "TRUST-SER: On The Trustworthiness Of Fine-Tuning Pre-Trained Speech Embeddings For Speech Emotion Recognition" [2023-05] [ICASSP 2024] [[paper](http://arxiv.org/abs/2305.11229)]

- "An Information-Theoretic Analysis of Self-supervised Discrete Representations of Speech" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.02405)]

- "Investigating wav2vec2 context representations and the effects of fine-tuning, a case-study of a Finnish model" [2023-03] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-837)]

- "On the (In)Efficiency of Acoustic Feature Extractors for Self-Supervised Speech Representation Learning" [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/interspeech.2023-1510)]

- "On The Choice of the Optimal Temporal Support for Audio Classification with Pre-Trained Embeddings" [2023-12] [ICASSP 2024] [[paper](http://arxiv.org/abs/2312.14005)]

- "A Closer Look at Wav2vec2 Embeddings for On-Device Single-Channel Speech Enhancement" [2024-03] [ICASSP 2024] [[paper](http://arxiv.org/abs/2403.01369)]

- "Following the Embedding: Identifying Transition Phenomena in Wav2vec 2.0 Representations of Speech Audio" [2024-03] [ICASSP 2024] [[paper](https://doi.org/10.1109/ICASSP48485.2024.10446494)]

- "Searching for Structure: Appraising the Organisation of Speech Features in wav2vec 2.0 Embeddings" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-2047)]

- "Wav2vec 2.0 Embeddings Are No Swiss Army Knife -- A Case Study for Multiple Sclerosis" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-995)]

- "Self-Supervised Speech Representations are More Phonetic than Semantic" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.08619)]

- "On the Encoding of Gender in Transformer-based ASR Representations" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.09855)]

- "Orthogonality and isotropy of speaker and phonetic information in self-supervised speech representations" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.09200)]

- "Investigating the Sensitivity of Pre-trained Audio Embeddings to Common Effects" [2025-01] [ICASSP 2025] [[paper](http://arxiv.org/abs/2501.15900)]

- "Evaluating the Effectiveness of Pre-Trained Audio Embeddings for Classification of Parkinson's Disease Speech Data" [2025-06] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2506.02078)]

- "Acoustic Representation and Realization of Weak Elements Subcategories: In the Case of Tianjin Mandarin" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-1835)]

- "A Study of Speech Embedding Similarities Between Australian Aboriginal and High-Resource Languages" [2025-09] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2509.01419)]

- "Evaluating Deep Speaker Embedding Robustness to Domain, Sampling Rate, and Codec Variations" [2025-09] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-2167)]

#### Speaker Characteristics

- "Quantitative Evidence on Overlooked Aspects of Enrollment Speaker Embeddings for Target Speaker Separation" [2022-10] [ICASSP 2023] [[paper](http://arxiv.org/abs/2210.12635)]

- "Speaker Embeddings as Individuality Proxy for Voice Stress Detection" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.05915)]

- "Speaker Verification Across Ages: Investigating Deep Speaker Embedding Sensitivity to Age Mismatch in Enrollment and Test Speech" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.07501)]

- "Behavioral Analysis of Pathological Speaker Embeddings of Patients During Oncological Treatment of Oral Cancer" [2023-07] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2307.04744)]

- "Controllable Generation of Artificial Speaker Embeddings through Discovery of Principal Directions" [2023-10] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2310.17502)]

- "Geodesic Interpolation of Frame-Wise Speaker Embeddings for the Diarization of Meeting Scenarios" [2024-01] [ICASSP 2024] [[paper](http://arxiv.org/abs/2401.03963)]

- "A Study on Graph Embedding for Speaker Recognition" [2024-03] [ICASSP 2024] [[paper](https://doi.org/10.1109/ICASSP48485.2024.10448308)]

- "Spoofed Speech Detection with a Focus on Speaker Embedding" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-481)]

- "The reasonable effectiveness of speaker embeddings for violence detection" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.06798)]

- "Gradual modeling of the Lombard effect by modifying speaker embeddings from a Text-To-Speech model" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-787)]

#### Benchmarking & Evaluating Extractors

- "In search of strong embedding extractors for speaker diarisation" [2022-10] [ICASSP 2023] [[paper](http://arxiv.org/abs/2210.14682)]

- "A Reality Check and A Practical Baseline for Semantic Speech Embedding" [2023-01] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10096254)]

- "Speech Self-Supervised Representation Benchmarking: Are We Doing it Right?" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1087)]

- "On the Usefulness of Speaker Embeddings for Speaker Retrieval in the Wild: A Comparative Study of x-vector and ECAPA-TDNN Models" [2024-07] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-161)]

- "Gender Representation in TV and Radio: Automatic Information Extraction methods versus Manual Analyses" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.10316)]

- "Rethinking Leveraging Pre-Trained Multi-Layer Representations for Speaker Verification" [2025-09] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-628)]

#### Interpretability

- "Learning Interpretable Low-dimensional Representation via Physical Symmetry" [2023-02] [NeurIPS 2023] [[paper](http://arxiv.org/abs/2302.10890)]

- "Similar Hierarchical Representation of Speech and Other Complex Sounds In the Brain and Deep Residual Networks: An MEG Study" [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/interspeech.2023-1347)]

- "What do self-supervised speech representations encode? An analysis of languages, varieties, speaking styles and speakers" [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/interspeech.2023-951)]

- "What Do Language Models Hear? Probing for Auditory Representations in Language Models" [2024-02] [ACL 2024] [[paper](http://arxiv.org/abs/2402.16998)]

- "From Sound to Meaning in the Auditory Cortex: A Neuronal Representation and Classification Analysis" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-2531)]

- "XANE: eXplainable Acoustic Neural Embeddings" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.05199)]

- "Form and Function in Prosodic Representation: In the Case of 'ma' in Tianjin Mandarin" [2024-09] [InterSpeech 2024] [[paper](https://doi.org/10.21437/interspeech.2024-1909)]

#### Cross-Domain Generalization

- "Can Self-Supervised Neural Representations Pre-Trained on Human Speech distinguish Animal Callers?" [2023-05] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2305.14035)]

- "On the Benefits of Self-supervised Learned Speech Representations for Predicting Human Phonetic Misperceptions" [2023-05] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1476)]

- "Towards hate speech detection in low-resource languages: Comparing ASR to acoustic word embeddings on Wolof and Swahili" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.00410)]

- "Investigation of Layer-Wise Speech Representations in Self-Supervised Learning Models: A Cross-Lingual Study in Detecting Depression" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-1737)]

- "Self-supervised Speech Representations Still Struggle with African American Vernacular English" [2024-08] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2408.14262)]

- "Exploring Self-Supervised Speech Representations for Cross-lingual Acoustic-to-Articulatory Inversion" [2024-09] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2309.01108)]

- "Gender and Language Identification in Multilingual Models of Speech: Exploring the Genericity and Robustness of Speech Representations" [2024-09] [InterSpeech 2024] [[paper](https://doi.org/10.21437/interspeech.2024-953)]

- "Representation of Perceived Prosodic Similarity of Conversational Feedback" [2025-05] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2505.13268)]

- "Recreating Neural Activity During Speech Production with Language and Speech Model Embeddings" [2025-05] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2505.14074)]

- "Dirichlet process mixture model based on topologically augmented signal representation for clustering infant vocalizations" [2024-07] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2407.05760)]

### 4.4 Benchmarks & Toolkits

- "Wespeaker: A Research and Production Oriented Speaker Embedding Learning Toolkit" [2022-10] [ICASSP 2023] [[paper](http://arxiv.org/abs/2210.17016)]

- "MARBLE: Music Audio Representation Benchmark for Universal Evaluation" [2023-06] [NeurIPS 2023] [[paper](http://arxiv.org/abs/2306.10548)]

- "ESPnet-SPK: full pipeline speaker embedding toolkit with reproducible recipes, self-supervised front-ends, and off-the-shelf models" [2024-01] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2401.17230)]

### 4.5 Embedding Applications

#### Speaker-Related Application

- "Advancing the Dimensionality Reduction of Speaker Embeddings for Speaker Diarisation: Disentangling Noise and Informing Speech Activity" [2021-10] [ICASSP 2023] [[paper](http://arxiv.org/abs/2110.03380)]

- "Exploiting Speaker Embeddings for Improved Microphone Clustering and Speech Separation in ad-hoc Microphone Arrays" [2023-03] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10094862)]

- "Towards Single Integrated Spoofing-aware Speaker Verification Embeddings" [2023-05] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2305.19051)]

- "A Teacher-Student Approach for Extracting Informative Speaker Embeddings From Speech Mixtures" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.00634)]

- "Improving End-to-End Neural Diarization Using Conversational Summary Representations" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.13863)]

- "SEF-Net: Speaker Embedding Free Target Speaker Extraction Network" [2023-07] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1749)]

- "Real-Time Personalised Speech Enhancement Transformers with Dynamic Cross-attended Speaker Representations" [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1066)]

- "SEF-VC: Speaker Embedding Free Zero-Shot Voice Conversion with Cross Attention" [2023-12] [ICASSP 2024] [[paper](http://arxiv.org/abs/2312.08676)]

- "Neural Speaker Diarization Using Memory-Aware Multi-Speaker Embedding with Sequence-to-Sequence Architecture" [2024-01] [ICASSP 2024] [[paper](https://doi.org/10.1109/ICASSP48485.2024.10446661)]

- "Speakers Unembedded: Embedding-free Approach to Long-form Neural Diarization" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-1174)]

- "Efficient Speaker Embedding Extraction Using a Twofold Sliding Window Algorithm for Speaker Diarization" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-1874)]

- "Fully Few-shot Class-incremental Audio Classification Using Expandable Dual-embedding Extractor" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.08122)]

- "Personalized Speech Enhancement Without a Separate Speaker Embedding Model" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.09928)]

- "Audio Fingerprinting with Holographic Reduced Representations" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.13139)]

- "Specializing Self-Supervised Speech Representations for Speaker Segmentation" [2024-09] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2501.05310)]

- "Leveraging Boolean Directivity Embedding for Binaural Target Speaker Extraction" [2025-07] [ICASSP 2025] [[paper](https://doi.org/10.1109/ICASSP49660.2025.10888158)]

- "Spatio-Spectral Diarization of Meetings by Combining TDOA-based Segmentation and Speaker Embedding-based Clustering" [2025-07] [InterSpeech 2025] [[paper](https://doi.org/10.48550/arXiv.2506.16228)]

- "A Siamese Network-Based Framework for Voice Mimicry Proficiency Assessment Using X-Vector Embeddings" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-1103)]

- "Bridging Speech and Singing: Multi-stage Speech-Prompted Singing Voice Conversion with Speaker Embedding Adaptation" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-816)]

#### Speech Recognition & Transcription

- "Multi-Lingual Pronunciation Assessment with Unified Phoneme Set and Language-Specific Embeddings" [2023-01] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10095673)]

- "Context-Aware end-to-end ASR Using Self-Attentive Embedding and Tensor Fusion" [2023-01] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10095204)]

- "Improvements to Embedding-Matching Acoustic-to-Word ASR Using Multiple-Hypothesis Pronunciation-Based Embeddings" [2023-01] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10095705)]

- "Self-supervised Learning Representation based Accent Recognition with Persistent Accent Memory" [2023-02] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-1702)]

- "Text-only Domain Adaptation using Unified Speech-Text Representation in Transducer" [2023-06] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.04076)]

- "TokenSplit: Using Discrete Speech Representations for Direct, Refined, and Transcript-Conditioned Speech Separation and Recognition" [2023-08] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2308.10415)]

- "Dual Audio Encoders Based Mandarin Prosodic Boundary Prediction by Using Multi-Granularity Prosodic Representations" [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/interspeech.2023-2242)]

- "Transducers with Pronunciation-Aware Embeddings for Automatic Speech Recognition" [2024-01] [ICASSP 2024] [[paper](https://doi.org/10.1109/ICASSP48485.2024.10447685)]

- "CIF-RNNT: Streaming ASR Via Acoustic Word Embeddings with Continuous Integrate-and-Fire and RNN-Transducers" [2024-01] [ICASSP 2024] [[paper](https://doi.org/10.1109/ICASSP48485.2024.10448492)]

- "Codec-ASR: Training Performant Automatic Speech Recognition Systems with Discrete Speech Representations" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-330)]

- "Dysarthric Speech Recognition Using Curriculum Learning and Articulatory Feature Embedding" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-444)]

- "CTC-aligned Audio-Text Embedding for Streaming Open-vocabulary Keyword Spotting" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.07923)]

- "Enhancing Multilingual ASR for Unseen Languages via Language Embedding Modeling" [2024-12] [ICASSP 2025] [[paper](http://arxiv.org/abs/2412.16474)]

#### Speech Emotion, Paralinguistic, Health & Cognitive Applications

- "Efficient Speech Quality Assessment Using Self-Supervised Framewise Embeddings" [2022-11] [ICASSP 2023] [[paper](http://arxiv.org/abs/2211.06646)]

- "Understanding Spoken Language Development of Children with ASD Using Pre-trained Speech Embeddings" [2023-05] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2305.14117)]

- "Robust Self Supervised Speech Embeddings for Child-Adult Classification in Interactions involving Children with Autism" [2023-07] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2307.16398)]

- "Classification of Vocal Intensity Category from Speech using the Wav2vec2 and Whisper Embeddings" [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-2038)]

- "AsthmaSCELNet: A Lightweight Supervised Contrastive Embedding Learning Framework for Asthma Classification Using Lung Sounds" [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-428)]

- "A Compressed Synthetic Speech Detection Method with Compression Feature Embedding" [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/interspeech.2023-1696)]

- "Classifying depression symptom severity: Assessment of speech representations in personalized and generalized machine learning models." [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/interspeech.2023-1721)]

- "Automated Multiple Sclerosis Screening Based on Encoded Speech Representations" [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/interspeech.2023-234)]

- "Obstructive Sleep Apnea Detection using Pre-trained Speech Representations" [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/interspeech.2023-278)]

- "Enhancing Child Vocalization Classification with Phonetically-Tuned Embeddings for Assisting Autism Diagnosis" [2023-09] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2309.07287)]

- "Fusing Multi-Level Features from Audio and Contextual Sentence Embedding from Text for Interview-Based Depression Detection" [2024-01] [ICASSP 2024] [[paper](https://doi.org/10.1109/ICASSP48485.2024.10446253)]

- "Are Paralinguistic Representations all that is needed for Speech Emotion Recognition?" [2024-02] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2402.01579)]

- "Whister: Using Whisper’s representations for Stuttering detection" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-2293)]

- "Automatic Assessment of Speech Production Skills for Children with Cochlear Implants Using Wav2Vec2.0 Acoustic Embeddings" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-234)]

- "Automatic Classification of News Subjects in Broadcast News: Application to a Gender Bias Representation Analysis" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-1854)]

- "Self-Supervised Embeddings for Detecting Individual Symptoms of Depression" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.17229)]

- "Developing vocal system impaired patient-aimed voice quality assessment approach using ASR representation-included multiple features" [2024-08] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2408.12279)]

- "Leveraging Universal Speech Representations for Detecting and Assessing the Severity of Mild Cognitive Impairment Across Languages" [2024-09] [InterSpeech 2024] [[paper](https://doi.org/10.21437/interspeech.2024-2030)]

- "Multimodal Fusion of Music Theory-Inspired and Self-Supervised Representations for Improved Emotion Recognition" [2024-09] [InterSpeech 2024] [[paper](https://doi.org/10.21437/interspeech.2024-2350)]

- "Multimodal Emotion Diarization: Frame-Wise Integration of Text and Audio Representations" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-2009)]

- "Advancing Emotion Recognition via Ensemble Learning: Integrating Speech, Context, and Text Representations" [2025-09] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-1445)]

- "Interactive Fusion of Multi-View Speech Embeddings via Pretrained Large-Scale Speech Models for Speech Emotional Attribute Prediction in Naturalistic Conditions" [2025-09] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-1662)]

- "Voice-Based Dysphagia Detection: Leveraging Self-Supervised Speech Representation" [2025-09] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-761)]

#### Text-To-Speech

- "Exploiting Emotion Information in Speaker Embeddings for Expressive Text-to-Speech" [2023-08] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2010.03909)]

- "SALTTS: Leveraging Self-Supervised Speech Representations for improved Text-to-Speech Synthesis" [2023-08] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2308.01018)]

- "Accent Conversion with Articulatory Representations" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.05947)]

- "Enhancing Multilingual TTS with Voice Conversion Based Data Augmentation and Posterior Embedding" [2024-07] [ICASSP 2024] [[paper](https://doi.org/10.1109/ICASSP48485.2024.10448471)]

#### Spoofing & Security

- "Learning A Self-Supervised Domain-Invariant Feature Representation for Generalized Audio Deepfake Detection" [2023-08] [InterSpeech 2023] [[paper](https://doi.org/10.21437/interspeech.2023-1383)]

- "An Efficient Temporary Deepfake Location Approach Based Embeddings for Partially Spoofed Audio Detection" [2023-09] [ICASSP 2024] [[paper](http://arxiv.org/abs/2309.03036)]

- "Interpretable Temporal Class Activation Representation for Audio Spoofing Detection" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.08825)]

- "Attentive Merging of Hidden Embeddings from Pre-trained Speech Model for Anti-spoofing Detection" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.10283)]

- "Towards generalisable and calibrated audio deepfake detection with self-supervised representations" [2024-09] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2309.05384)]

- "An Explainable Probabilistic Attribute Embedding Approach for Spoofed Speech Characterization" [2024-09] [ICASSP 2025] [[paper](http://arxiv.org/abs/2409.11027)]

- "Exploring Self-supervised Embeddings and Synthetic Data Augmentation for Robust Audio Deepfake Detection" [2024-09] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-942)]

- "SpeechForensics: Audio-Visual Speech Representation Learning for Face Forgery Detection" [2025-08] [NeurIPS 2024] [[paper](http://arxiv.org/abs/2508.09913)]

- "Generalizable Audio Spoofing Detection using Non-Semantic Representations" [2025-08] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2509.00186)]

- "Enhancing Audio Deepfake Detection by Improving Representation Similarity of Bonafide Speech" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-422)]

#### Others

- "Feature Selection and Text Embedding for Detecting Dementia from Spontaneous Cantonese" [2023-01] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10095140)]

- "Fully Unsupervised Topic Clustering of Unlabelled Spoken Audio Using Self-Supervised Representation Learning and Topic Model" [2023-03] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10095280)]

- "Understanding Disrupted Sentences Using Underspecified Abstract Meaning Representation" [2023-06] [InterSpeech 2023] [[paper](https://doi.org/10.21437/Interspeech.2023-307)]

- "End to End Spoken Language Diarization with Wav2vec Embeddings" [2023-08] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2306.12913)]

- "Flexible Keyword Spotting Based on Homogeneous Audio-Text Embedding" [2023-08] [ICASSP 2024] [[paper](http://arxiv.org/abs/2308.06472)]

- "Joint Prediction of Audio Event and Annoyance Rating in an Urban Soundscape by Hierarchical Graph Representation Learning" [2023-08] [InterSpeech 2023] [[paper](http://arxiv.org/abs/2308.11980)]

- "Enhanced Embeddings in Zero-Shot Learning for Environmental Audio" [2023-08] [ICASSP 2023] [[paper](https://doi.org/10.1109/ICASSP49357.2023.10096134)]

- "Improving Audio Captioning Models with Fine-Grained Audio Features, Text Embedding Supervision, and LLM Mix-Up Augmentation" [2023-09] [ICASSP 2024] [[paper](http://arxiv.org/abs/2309.17352)]

- "A Deep Representation Learning-Based Speech Enhancement Method Using Complex Convolution Recurrent Variational Autoencoder" [2023-12] [ICASSP 2024] [[paper](http://arxiv.org/abs/2312.09620)]

- "Similar but Faster: Manipulation of Tempo in Music Audio Embeddings for Tempo Prediction and Search" [2024-01] [ICASSP 2024] [[paper](http://arxiv.org/abs/2401.08902)]

- "Improving Oral Reading Fluency Assessment Through Sub-Sequence Matching of Acoustic Word Embeddings" [2024-01] [ICASSP 2024] [[paper](https://doi.org/10.1109/ICASSP48485.2024.10447029)]

- "Ainur: Harmonizing Speed and Quality in Deep Music Generation Through Lyrics-Audio Embeddings" [2024-01] [ICASSP 2024] [[paper](https://doi.org/10.1109/ICASSP48485.2024.10448078)]

- "Sound of Vision: Audio Generation from Visual Text Embedding through Training Domain Discriminator" [2024-05] [InterSpeech 2024] [[paper](https://doi.org/10.21437/Interspeech.2024-1451)]

- "CALL system using pitch-accent feature representations reflecting listeners’ subjective adequacy" [2024-05] [InterSpeech 2024] [[paper](https://www.isca-archive.org/interspeech_2024/masudakatsuse24_interspeech.html)]

- "RevRIR: Joint Reverberant Speech and Room Impulse Response Embedding using Contrastive Learning with Application to Room Shape Classification" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.03120)]

- "Joint Learning of Context and Feedback Embeddings in Spoken Dialogue" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.07291)]

- "Multimodal Representation Loss Between Timed Text and Audio for Regularized Speech Separation" [2024-06] [InterSpeech 2024] [[paper](http://arxiv.org/abs/2406.08328)]

- "Text2FX: Harnessing CLAP Embeddings for Text-Guided Audio Effects" [2024-09] [ICASSP 2025] [[paper](http://arxiv.org/abs/2409.18847)]

- "Zero-shot Musical Stem Retrieval with Joint-Embedding Predictive Architectures" [2024-11] [ICASSP 2025] [[paper](http://arxiv.org/abs/2411.19806)]

- "Music2Latent2: Audio Compression with Summary Embeddings and Autoregressive Decoding" [2025-01] [ICASSP 2025] [[paper](http://arxiv.org/abs/2501.17578)]

- "Learning Musical Representations for Music Performance Question Answering" [2025-02] [EMNLP 2024 Findings] [[paper](http://arxiv.org/abs/2502.06710)]

- "Discrete Audio Representations for Automated Audio Captioning" [2025-05] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2505.14989)]

- "CLAP-ART: Automated Audio Captioning with Semantic-rich Audio Representation Tokenizer" [2025-06] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2506.00800)]

- "Efficient Speech Enhancement via Embeddings from Pre-trained Generative Audioencoders" [2025-06] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2506.11514)]

- "Fully Few-shot Class-incremental Audio Classification Using Multi-level Embedding Extractor and Ridge Regression Classifier" [2025-06] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2506.18406)]

- "Listen through the Sound: Generative Speech Restoration Leveraging Acoustic Context Representation" [2025-08] [InterSpeech 2025] [[paper](http://arxiv.org/abs/2508.08953)]

- "Dog2vec: Self-Supervised Pre-Training for Canine Vocal Representation" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-1287)]

- "Simple and Effective Content Encoder for Singing Voice Conversion via SSL-Embedding Dimension Reduction" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-1531)]

- "GoP2Vec: A few shot learning for pronunciation assessment with goodness of pronunciation (GoP) based representations from an i-vector framework and augmentation" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-2359)]

- "FUSE-MOS: Fusion of Speech Embeddings for MOS Prediction with Uncertainty Quantification" [2025-08] [InterSpeech 2025] [[paper](https://doi.org/10.21437/interspeech.2025-2532)]

- "Causal Speech Enhancement Based on a Two-Branch Nested U-Net Architecture Using Self-Supervised Speech Embeddings" [2025-08] [ICASSP 2025] [[paper](https://doi.org/10.1109/ICASSP49660.2025.10888248)]

## 5. Graph Embedding

### 5.1 Node Embedding

- "Deepwalk: Online learning of social representations" [2014-03] [KDD 2014] [[paper](https://arxiv.org/abs/1403.6652)]

- "Line: Large-scale information network embedding" [2015-03] [WWW 2015] [[paper](https://arxiv.org/abs/1503.03578)]

- "node2vec: Scalable feature learning for networks" [2016-07] [KDD 2016] [[paper](https://arxiv.org/abs/1607.00653)]

- "Asymmetric transitivity preserving graph embedding" [2016-08] [KDD 2016] [[paper](https://dl.acm.org/doi/10.1145/2939672.2939751)]

- "Semi-Supervised Classification with Graph Convolutional Networks" [2016-09] [ICLR 2017] [[paper](https://arxiv.org/abs/1609.02907)]

- "Inductive representation learning on large graphs" [2017-06] [NeurIPS 2017] [[paper](https://arxiv.org/abs/1706.02216)]

- "metapath2vec: Scalable representation learning for heterogeneous networks" [2017-08] [KDD 2017] [[paper](https://dl.acm.org/doi/10.1145/3097983.3098036)]

- "Graph attention networks" [2017-10] [ICLR 2018] [[paper](https://arxiv.org/abs/1710.10903)]

- "How powerful are graph neural networks?" [2018-10] [ICLR 2019] [[paper](https://arxiv.org/abs/1810.00826)]

- "Relational graph attention networks" [2019-04] [KDD 2020] [[paper](https://arxiv.org/abs/1904.05811)]

- "Do transformers really perform badly for graph representation?" [2021-06] [NeurIPS 2021] [[paper](https://arxiv.org/abs/2106.05234)]

- "Nodepiece: Compositional and parameter-efficient representations of large knowledge graphs" [2021-06] [ICLR 2022] [[paper](https://arxiv.org/abs/2106.12144)]

- "Sign and Basis Invariant Networks for Spectral Graph Representation Learning" [2022-02] [ICLR 2023] [[paper](https://arxiv.org/abs/2202.13013)]

- "Translating Subgraphs to Nodes Makes Simple GNNs Strong and Efficient for Subgraph Representation Learning" [2022-04] [ICML 2024] [[paper](https://arxiv.org/abs/2204.04510)]

- "Empowering Graph Representation Learning with Test-Time Graph Transformation" [2022-10] [ICLR 2023] [[paper](https://arxiv.org/abs/2210.03561)]

- "DyG2Vec: Efficient Representation Learning for Dynamic Graphs" [2022-10] [TMLR 2024] [[paper](https://arxiv.org/abs/2210.16906)]

- "Learning Fair Graph Representations via Automated Data Augmentations" [2023-02] [ICLR 2023] [[paper](https://openreview.net/forum?id=1_OGWcP1s9w)]

- "Chasing All-Round Graph Representation Robustness: Model, Training, and Optimization" [2023-02] [ICLR 2023] [[paper](https://openreview.net/forum?id=7jk5gWjC18M)]

- "Spacetime Representation Learning" [2023-02] [ICLR 2023] [[paper](https://openreview.net/forum?id=qV_M_rhYajc)]

- "Towards Better Graph Representation Learning with Parameterized Decomposition & Filtering" [2023-05] [ICML 2023] [[paper](https://arxiv.org/abs/2305.06102)]

- "Fisher Information Embedding for Node and Graph Learning" [2023-05] [ICML 2023] [[paper](https://arxiv.org/abs/2305.07580)]

- "Tractable Probabilistic Graph Representation Learning with Graph-Induced Sum-Product Networks" [2023-05] [ICLR 2024] [[paper](https://arxiv.org/abs/2305.10544)]

- "Seq-HGNN: Learning Sequential Node Representation on Heterogeneous Graph" [2023-05] [SIGIR 2023] [[paper](https://arxiv.org/abs/2305.10771)]

- "Node Embedding from Neural Hamiltonian Orbits in Graph Neural Networks" [2023-05] [ICML 2023] [[paper](https://arxiv.org/abs/2305.18965)]

- "Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning" [2023-05] [ICLR 2024] [[paper](https://arxiv.org/abs/2305.19523)]

- "SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations" [2023-06] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2306.10759)]

- "Directional diffusion models for graph representation learning" [2023-06] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2306.13210)]

- "When Sparsity Meets Contrastive Models: Less Graph Data Can Bring Better Class-Balanced Representations" [2023-06] [ICML 2023] [[paper](https://openreview.net/forum?id=3jV525Hmqr)]

- "Disentangled Multiplex Graph Representation Learning" [2023-06] [ICML 2023] [[paper](https://openreview.net/forum?id=lYZOjMvxws)]

- "VQGraph: Rethinking Graph Representation Space for Bridging GNNs and MLPs" [2023-08] [ICLR 2024] [[paper](https://arxiv.org/abs/2308.02117)]

- "Graph-enhanced Optimizers for Structure-aware Recommendation Embedding Evolution" [2023-09] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2310.03032)]

- "LD2: Scalable Heterophilous Graph Neural Network with Decoupled Embeddings" [2023-09] [NeurIPS 2023] [[paper](https://openreview.net/forum?id=7zkFc9TGKz)]

- "WalkLM: A Uniform Language Model Fine-tuning Framework for Attributed Graph Embedding" [2023-09] [NeurIPS 2023] [[paper](https://openreview.net/forum?id=ZrG8kTbt70)]

- "FiGURe: Simple and Efficient Unsupervised Node Representations with Filter Augmentations" [2023-10] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2310.01892)]

- "GRENADE: Graph-Centric Language Model for Self-Supervised Representation Learning on Text-Attributed Graphs" [2023-10] [EMNLP 2023 Findings] [[paper](https://arxiv.org/abs/2310.15109)]

- "Community Detection Guarantees using Embeddings Learned by Node2Vec" [2023-10] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2310.17712)]

- "Zero-shot Node Classification with Graph Contrastive Embedding Network" [2023-10] [TMLR 2023] [[paper](https://openreview.net/forum?id=8wGXnjRLSy)]

- "Content- and Topology-Aware Representation Learning for Scientific Multi-Literature" [2023-12] [EMNLP 2023] [[paper](https://aclanthology.org/2023.emnlp-main.465/)]

- "Recurrent Distance Filtering for Graph Representation Learning" [2023-12] [ICML 2024] [[paper](https://arxiv.org/abs/2312.01538)]

- "HypeBoy: Generative Self-Supervised Representation Learning on Hypergraphs" [2024-01] [ICLR 2024] [[paper](https://arxiv.org/abs/2404.00638)]

- "UNR-Explainer: Counterfactual Explanations for Unsupervised Node Representation Learning Models" [2024-01] [ICLR 2024] [[paper](https://openreview.net/forum?id=0j9ZDzMPqr)]

- "Node2ket: Efficient High-Dimensional Network Embedding in Quantum Hilbert Space" [2024-01] [ICLR 2024] [[paper](https://openreview.net/forum?id=lROh08eK6n)]

- "Learning Invariant Representations of Graph Neural Networks via Cluster Generalization" [2024-03] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2403.03599)]

- "High-Frequency-aware Hierarchical Contrastive Selective Coding for Representation Learning on Text Attributed Graphs" [2024-04] [WWW 2024] [[paper](https://arxiv.org/abs/2402.16240)]

- "Node Identifiers: Compact, Discrete Representations for Efficient Graph Learning" [2024-05] [ICLR 2025] [[paper](https://arxiv.org/abs/2405.16435)]

- "Enhancing Size Generalization in Graph Neural Networks through Disentangled Representation Learning" [2024-06] [ICML 2024] [[paper](https://arxiv.org/abs/2406.04601)]

- "Learning Divergence Fields for Shift-Robust Graph Representations" [2024-06] [ICML 2024] [[paper](https://arxiv.org/abs/2406.04963)]

- "DUPLEX: Dual GAT for Complex Embedding of Directed Graphs" [2024-06] [ICML 2024] [[paper](https://arxiv.org/abs/2406.05391)]

- "Explaining Node Embeddings" [2024-06] [TMLR 2025] [[paper](https://arxiv.org/abs/2406.07642)]

- "Leveraging Contrastive Learning for Enhanced Node Representations in Tokenized Graph Transformers" [2024-06] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2406.19258)]

- "Non-Euclidean Mixture Model for Social Network Embedding" [2024-09] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2411.04876)]

- "Learning Representations for Hierarchies with Minimal Support" [2024-09] [NeurIPS 2024] [[paper](https://openreview.net/forum?id=HFS800reZK)]

- "Disentangled and Self-Explainable Node Representation Learning" [2024-10] [TMLR 2025] [[paper](https://arxiv.org/abs/2410.21043)]

- "LASE: Learned Adjacency Spectral Embeddings" [2024-12] [TMLR 2025] [[paper](https://arxiv.org/abs/2412.17734)]

- "Generalizable Spectral Embedding with an Application to UMAP" [2025-01] [TMLR 2025] [[paper](https://arxiv.org/abs/2501.11305)]

- "Holographic Node Representations: Pre-training Task-Agnostic Node Embeddings" [2025-01] [ICLR 2025] [[paper](https://openreview.net/forum?id=tGYFikNONB)]

- "Disobeying Directions: Switching Random Walk Filters for Unsupervised Node Embedding Learning on Directed Graphs" [2025-01] [TMLR 2025] [[paper](https://openreview.net/forum?id=yngjRgVA5A)]

- "Genetic-Evolutionary Graph Neural Networks: A Paradigm for Improved Graph Representation Learning" [2025-02] [TMLR 2025] [[paper](https://openreview.net/forum?id=qzYTklXVAB)]

- "Balancing Graph Embedding Smoothness in Self-supervised Learning via Information-Theoretic Decomposition" [2025-04] [WWW 2025] [[paper](https://arxiv.org/abs/2504.12011)]

- "GPEN: Global Position Encoding Network for Enhanced Subgraph Representation Learning" [2025-05] [ICML 2025] [[paper](https://openreview.net/forum?id=7QFmZ7i7sr)]

- "Primphormer: Efficient Graph Transformers with Primal Representations" [2025-05] [ICML 2025] [[paper](https://openreview.net/forum?id=fMAihjfJij)]

- "SDMG: Smoothing Your Diffusion Models for Powerful Graph Representation Learning" [2025-05] [ICML 2025] [[paper](https://openreview.net/forum?id=lNyaQIJ5Z7)]

- "Stable Fair Graph Representation Learning with Lipschitz Constraint" [2025-05] [ICML 2025] [[paper](https://openreview.net/forum?id=oJQWvsStNh)]

- "iN2V: Bringing Transductive Node Embeddings to Inductive Graphs" [2025-06] [ICML 2025] [[paper](https://arxiv.org/abs/2506.05039)]

- "Full-Rank Unsupervised Node Embeddings for Directed Graphs via Message Aggregation" [2025-06] [TMLR 2025] [[paper](https://openreview.net/forum?id=3ECbEZg2If)]

- "Node2binary: Compact Graph Node Embeddings using Binary Vectors" [2025-06] [WWW 2025] [[paper](https://openreview.net/forum?id=s3KIzcRdll)]

### 5.2 Graph Embedding

- "Evaluating Self-Supervised Learning for Molecular Graph Embeddings" [2022-06] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2206.08005)]

- "Tight and fast generalization error bound of graph embedding in metric space" [2023-05] [ICML 2023] [[paper](https://arxiv.org/abs/2305.07971)]

- "Expectation-Complete Graph Representations with Homomorphisms" [2023-06] [ICML 2023] [[paper](https://arxiv.org/abs/2306.05838)]

- "PlanE: Representation Learning over Planar Graphs" [2023-07] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2307.01180)]

- "Rethinking the Power of Graph Canonization in Graph Representation Learning with Stability" [2023-09] [ICLR 2024] [[paper](https://arxiv.org/abs/2309.00738)]

- "Graph-level Representation Learning with Joint-Embedding Predictive Architectures" [2023-09] [TMLR 2025] [[paper](https://arxiv.org/abs/2309.16014)]

- "Lovász Principle for Unsupervised Graph Representation Learning" [2023-09] [NeurIPS 2023] [[paper](https://openreview.net/forum?id=0vdEHDwamk)]

- "Laplacian Canonization: A Minimalist Approach to Sign and Basis Invariant Spectral Embedding" [2023-10] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2310.18716)]

- "Normed Spaces for Graph Embedding" [2023-12] [ICLR 2025] [[paper](https://arxiv.org/abs/2312.01502)]

- "A Simple and Scalable Representation for Graph Generation" [2023-12] [ICLR 2024] [[paper](https://arxiv.org/abs/2312.02230)]

- "Weisfeiler and Leman Go Loopy: A New Hierarchy for Graph Representational Learning" [2024-03] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2403.13749)]

- "HC-GAE: The Hierarchical Cluster-based Graph Auto-Encoder for Graph Representation Learning" [2024-05] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2405.14742)]

- "Learning Graph Representation via Graph Entropy Maximization" [2024-07] [ICML 2024] [[paper](https://proceedings.mlr.press/v235/sun24i.html)]

- "Neural Spacetimes for DAG Representation Learning" [2024-08] [ICLR 2025] [[paper](https://arxiv.org/abs/2408.13885)]

- "LLMs as Zero-shot Graph Learners: Alignment of GNN Representations with LLM Token Embeddings" [2024-08] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2408.14512)]

- "Exploitation of a Latent Mechanism in Graph Contrastive Learning: Representation Scattering" [2024-09] [NeurIPS 2024] [[paper](https://openreview.net/forum?id=R8SolCx62K)]

- "Exploring Consistency in Graph Representations: from Graph Kernels to Graph Neural Networks" [2024-10] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2410.23748)]

- "ICLR: In-Context Learning of Representations" [2025-01] [ICLR 2025] [[paper](https://arxiv.org/abs/2501.00070)]

- "How Low Can You Go? Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node Embeddings" [2025-01] [ICLR 2025] [[paper](https://arxiv.org/abs/2503.01723)]

- "Charting the Design Space of Neural Graph Representations for Subgraph Matching" [2025-01] [ICLR 2025] [[paper](https://openreview.net/forum?id=5pd78GmXC6)]

- "A Hubness Perspective on Representation Learning for Graph-Based Multi-View Clustering" [2025-06] [CVPR 2025] [[paper](https://cvpr.thecvf.com/virtual/2025/poster/32724)]

- "Heterogeneous Graph Embedding Made More Practical" [2025-07] [SIGIR 2025] [[paper](https://dl.acm.org/doi/10.1145/3726302.3729993)]

### 5.3 Edge Embedding

- "Edgeformers: Graph-Empowered Transformers for Representation Learning on Textual-Edge Networks" [2023-02] [ICLR 2023] [[paper](https://arxiv.org/abs/2302.11050)]

- "Towards characterizing the value of edge embeddings in Graph Neural Networks" [2024-10] [ICML 2025] [[paper](https://arxiv.org/abs/2410.09867)]

### 5.4 Knowledge Graph Embedding

- "Translating embeddings for modeling multi-relational data" [2013-12] [NeurIPS 2013] [[paper](https://dl.acm.org/doi/10.5555/2999792.2999923)]

- "Rotate: Knowledge graph embedding by relational rotation in complex space" [2019-02] [ICLR 2019] [[paper](https://arxiv.org/abs/1902.10197)]

- "Knowledge Hypergraph Embedding Meets Relational Algebra" [2021-02] [ICML 2023] [[paper](https://arxiv.org/abs/2102.09557)]

- "ExpressivE: A Spatio-Functional Embedding For Knowledge Graph Completion" [2022-06] [ICLR 2023] [[paper](https://arxiv.org/abs/2206.04192)]

- "RulE: Knowledge Graph Reasoning with Rule Embedding" [2022-10] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2210.14905)]

- "Wasserstein-Fisher-Rao Embedding: Logical Query Embeddings with Local Comparison and Global Transport" [2023-05] [ACL 2023 Findings] [[paper](https://arxiv.org/abs/2305.04034)]

- "Polar Ducks and Where to Find Them: Enhancing Entity Linking with Duck Typing and Polar Box Embeddings" [2023-05] [EMNLP 2023] [[paper](https://arxiv.org/abs/2305.12027)]

- "How to Turn Your Knowledge Graph Embeddings into Generative Models" [2023-05] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2305.15944)]

- "InGram: Inductive Knowledge Graph Embedding via Relation Graphs" [2023-05] [ICML 2023] [[paper](https://arxiv.org/abs/2305.19987)]

- "Shrinking Embeddings for Hyper-Relational Knowledge Graphs" [2023-06] [ACL 2023] [[paper](https://arxiv.org/abs/2306.02199)]

- "What Makes Entities Similar? A Similarity Flooding Perspective for Multi-sourced Knowledge Graph Embeddings" [2023-06] [ICML 2023] [[paper](https://arxiv.org/abs/2306.02622)]

- "Knowledge Graph Embeddings using Neural Ito Process: From Multiple Walks to Stochastic Trajectories" [2023-07] [ACL 2023 Findings] [[paper](https://aclanthology.org/2023.findings-acl.448/)]

- "SConE: Simplified Cone Embeddings with Symbolic Operators for Complex Logical Queries" [2023-07] [ACL 2023 Findings] [[paper](https://aclanthology.org/2023.findings-acl.755/)]

- "Contrastive Learning with Generated Representations for Inductive Knowledge Graph Embedding" [2023-07] [ACL 2023 Findings] [[paper](https://aclanthology.org/2023.findings-acl.900/)]

- "Concept2Box: Joint Geometric Embeddings for Learning Two-View Knowledge Graphs" [2023-07] [ACL 2023 Findings] [[paper](https://arxiv.org/abs/2307.01933)]

- "Weighted Knowledge Graph Embedding" [2023-07] [SIGIR 2023] [[paper](https://dl.acm.org/doi/10.1145/3539618.3591784)]

- "Relation-aware Ensemble Learning for Knowledge Graph Embedding" [2023-10] [EMNLP 2023] [[paper](https://arxiv.org/abs/2310.08917)]

- "Solving Hard Analogy Questions with Relation Embedding Chains" [2023-10] [EMNLP 2023] [[paper](https://arxiv.org/abs/2310.12379)]

- "Are Embedded Potatoes Still Vegetables? On the Limitations of WordNet Embeddings for Lexical Semantics" [2023-12] [EMNLP 2023] [[paper](https://aclanthology.org/2023.emnlp-main.542/)]

- "Block-Diagonal Orthogonal Relation and Matrix Entity for Knowledge Graph Embedding" [2024-01] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2401.05967)]

- "MQuinE: a Cure for “Z-paradox” in Knowledge Graph Embedding" [2024-02] [EMNLP 2024] [[paper](https://arxiv.org/abs/2402.03583)]

- "Dynamic Graph Representation with Knowledge-aware Attention for Histopathology Whole Slide Image Analysis" [2024-03] [CVPR 2024] [[paper](https://arxiv.org/abs/2403.07719)]

- "PAC-Bayesian Generalization Bounds for Knowledge Graph Representation Learning" [2024-05] [ICML 2024] [[paper](https://arxiv.org/abs/2405.06418)]

- "Generalizing Knowledge Graph Embedding with Universal Orthogonal Parameterization" [2024-05] [ICML 2024] [[paper](https://arxiv.org/abs/2405.08540)]

- "Multiple Heads are Better than One: Mixture of Modality Knowledge Experts for Entity Representation Learning" [2024-05] [ICLR 2025] [[paper](https://arxiv.org/abs/2405.16869)]

- "Bridging the Space Gap: Unifying Geometry Knowledge Graph Embedding with Optimal Transport" [2024-05] [WWW 2024] [[paper](https://dl.acm.org/doi/10.1145/3589334.3645565)]

- "SpeedE: Euclidean Geometric Knowledge Graph Embedding Strikes Back" [2024-06] [NAACL 2024 Findings] [[paper](https://aclanthology.org/2024.findings-naacl.6/)]

- "Improving Multi-hop Logical Reasoning in Knowledge Graphs with Context-Aware Query Representation Learning" [2024-06] [ACL 2024 Findings] [[paper](https://arxiv.org/abs/2406.07034)]

- "Croppable Knowledge Graph Embedding" [2024-07] [ACL 2025] [[paper](https://arxiv.org/abs/2407.02779)]

- "Learning Low-dimensional Multi-domain Knowledge Graph Embedding via Dual Archimedean Spirals" [2024-08] [ACL 2024 Findings] [[paper](https://aclanthology.org/2024.findings-acl.118/)]

- "HyperCL: A Contrastive Learning Framework for Hyper-Relational Knowledge Graph Embedding with Hierarchical Ontology" [2024-08] [ACL 2024 Findings] [[paper](https://aclanthology.org/2024.findings-acl.171/)]

- "Enhancing Hyperbolic Knowledge Graph Embeddings via Lorentz Transformations" [2024-08] [ACL 2024 Findings] [[paper](https://aclanthology.org/2024.findings-acl.272/)]

- "Predictive Multiplicity of Knowledge Graph Embeddings in Link Prediction" [2024-08] [EMNLP 2024 Findings] [[paper](https://arxiv.org/abs/2408.08226)]

- "Conformalized Answer Set Prediction for Knowledge Graph Embedding" [2024-08] [NAACL 2025] [[paper](https://arxiv.org/abs/2408.08248)]

- "Clustering then Propagation: Select Better Anchors for Knowledge Graph Embedding" [2024-09] [NeurIPS 2024] [[paper](https://openreview.net/forum?id=BpJ6OTfWw3)]

- "DECRL: A Deep Evolutionary Clustering Jointed Temporal Knowledge Graph Representation Learning Approach" [2024-10] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2410.22631)]

- "Joint Pre-Encoding Representation and Structure Embedding for Efficient and Low-Resource Knowledge Graph Completion" [2024-11] [EMNLP 2024] [[paper](https://aclanthology.org/2024.emnlp-main.851/)]

- "Optimal Embedding Guided Negative Sample Generation for Knowledge Graph Link Prediction" [2025-04] [TMLR 2025] [[paper](https://arxiv.org/abs/2504.03327)]

- "A Mutual Information Perspective on Knowledge Graph Embedding" [2025-05] [ACL 2025] [[paper](https://aclanthology.org/2025.acl-long.1077/)]

- "Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings" [2025-05] [ACL 2025 Findings] [[paper](https://arxiv.org/abs/2505.16877)]

- "RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph" [2025-05] [ACL 2025] [[paper](https://arxiv.org/abs/2505.20813)]

- "Structure Is All You Need: Structural Representation Learning on Hyper-Relational Knowledge Graphs" [2025-05] [ICML 2025] [[paper](https://openreview.net/forum?id=2tH2vexW1Z)]

- "From Knowledge Forgetting to Accumulation: Evolutionary Relation Path Passing for Lifelong Knowledge Graph Embedding" [2025-07] [SIGIR 2025] [[paper](https://dl.acm.org/doi/10.1145/3726302.3729982)]

- "Rethinking Continual Knowledge Graph Embedding: Benchmarks and Analysis" [2025-07] [SIGIR 2025] [[paper](https://dl.acm.org/doi/10.1145/3726302.3730073)]

## 6. Time Series Embedding

### 6.1 Foundation Models

- "Towards a General Time Series Forecasting Model with Unified Representation and Adaptive Transfer" [2024-05] [ICML 2025] [[paper](https://arxiv.org/abs/2405.17478)]

- "On the Regularization of Learnable Embeddings for Time Series Forecasting" [2025-02] [TMLR 2025] [[paper](https://arxiv.org/abs/2410.14630)]

- "Exploring Representations and Interventions in Time Series Foundation Models" [2025-06] [ICML 2025] [[paper](https://arxiv.org/abs/2409.12915)]

### 6.2 Model Architecture & Training Methods

- "SOM-CPC: Unsupervised Contrastive Learning with Self-Organizing Maps for Structured Representations of High-Rate Time Series" [2022-05] [ICML 2023] [[paper](https://arxiv.org/abs/2205.15875)]

- "Out-of-distribution Representation Learning for Time Series Classification" [2022-09] [ICLR 2023] [[paper](https://arxiv.org/abs/2209.07027)]

- "TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series" [2023-08] [ICLR 2024] [[paper](https://arxiv.org/abs/2308.08241)]

- "T-Rep: Representation Learning for Time Series using Time-Embeddings" [2023-10] [ICLR 2024] [[paper](https://arxiv.org/abs/2310.04486)]

- "NuTime: Numerically Multi-Scaled Embedding for Large-Scale Time-Series Pretraining" [2023-10] [TMLR 2024] [[paper](https://arxiv.org/abs/2310.07402)]

- "Time Series Kernels based on Nonlinear Vector AutoRegressive Delay Embeddings" [2023-12] [NeurIPS 2023] [[paper](https://openreview.net/pdf?id=UBUWFEwn7p)]

- "CaRiNG: Learning Temporal Causal Representation under Non-Invertible Generation Process" [2024-01] [ICML 2024] [[paper](https://arxiv.org/abs/2401.14535)]

- "TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis" [2024-02] [TMLR 2024] [[paper](https://arxiv.org/abs/2402.16412)]

- "Multi-Patch Prediction: Adapting Language Models for Time Series Representation Learning" [2024-03] [ICML 2024] [[paper](https://arxiv.org/abs/2402.04852)]

- "TSLANet: Rethinking Transformers for Time Series Representation Learning" [2024-04] [ICML 2024] [[paper](https://arxiv.org/abs/2404.08472)]

- "Segment, Shuffle, and Stitch: A Simple Layer for Improving Time-Series Representations" [2024-05] [NeurIPS 2024] [[paper](https://arxiv.org/abs/2405.20082)]

- "GAFormer: Enhancing Timeseries Transformers Through Group-Aware Embeddings" [2024-05] [ICLR 2024] [[paper](https://openreview.net/pdf?id=c56TWtYp0W)]

- "Disentangling Time Series Representations via Contrastive Independence-of-Support on l-Variational Inference" [2024-05] [ICLR 2024] [[paper](https://openreview.net/pdf?id=iI7hZSczxE)]

- "Nonlinear Sequence Embedding by Monotone Variational Inequality" [2024-06] [ICLR 2025] [[paper](https://arxiv.org/abs/2406.06894)]

- "MF-CLR: Multi-Frequency Contrastive Learning Representation for Time Series" [2024-07] [ICML 2024] [[paper](https://openreview.net/pdf?id=ecO7WOIlMD)]

- "SigDiffusions: Score-Based Diffusion Models for Time Series via Log-Signature Embeddings" [2025-02] [ICLR 2025] [[paper](https://arxiv.org/abs/2406.10354)]

- "LETS-C: Leveraging Text Embedding for Time Series Classification" [2025-05] [ACL 2025] [[paper](https://arxiv.org/abs/2407.06533)]

- "TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation" [2025-06] [ICML 2025] [[paper](https://arxiv.org/abs/2410.05711)]

- "MERIT: Multi-Agent Collaboration for Unsupervised Time Series Representation Learning" [2025-07] [ACL 2025 Findings] [[paper](https://aclanthology.org/2025.findings-acl.1231.pdf)]

- "Time Series Representations with Hard-Coded Invariances" [2025-07] [ICML 2025] [[paper](https://openreview.net/pdf?id=SaKPKyjDp6)]

- "Learning Time-Series Representations by Hierarchical Uniformity-Tolerance Latent Balancing" [2025-10] [TMLR 2025] [[paper](https://www.arxiv.org/abs/2510.01658)]

### 6.3 Temporal Knowledge Graph

- "ECOLA: Enhancing Temporal Knowledge Embeddings with Contextualized Language Representations" [2022-03] [ACL 2023 Findings] [[paper](https://arxiv.org/abs/2203.09590)]

- "TFLEX: Temporal Feature-Logic Embedding Framework for Complex Reasoning over Temporal Knowledge Graph" [2022-05] [NeurIPS 2023] [[paper](https://arxiv.org/abs/2205.14307)]

- "TeAST: Temporal Knowledge Graph Embedding via Archimedean Spiral Timeline" [2023-07] [ACL 2023] [[paper](https://aclanthology.org/2023.acl-long.862.pdf)]

- "Learning Joint Structural and Temporal Contextualized Knowledge Embeddings for Temporal Knowledge Graph Completion" [2023-07] [ACL 2023 Findings] [[paper](https://aclanthology.org/2023.findings-acl.28.pdf)]

- "Noether Embedding: Efficient Learning of Temporal Regularities" [2023-12] [NeurIPS 2023] [[paper](https://openreview.net/pdf?id=27CRbwewyb)]

- "Mitigating Heterogeneity among Factor Tensors via Lie Group Manifolds for Tensor Decomposition Based Temporal Knowledge Graph Embedding" [2025-02] [NAACL 2025] [[paper](https://arxiv.org/abs/2404.09155)]

### 6.4 Temporal Networks

- "Direct Embedding of Temporal Network Edges via Time-Decayed Line Graphs" [2022-09] [ICLR 2023] [[paper](https://arxiv.org/abs/2210.00032)]

- "HiT-MDP: Learning the SMDP option framework on MDPs with Hidden Temporal Embeddings" [2023-05] [ICLR 2023] [[paper](https://openreview.net/pdf?id=VuuDXDgujAc)]
