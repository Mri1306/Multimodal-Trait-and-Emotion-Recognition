# Multimodal-Trait-and-Emotion-Recognition
📌 Overview

This repository implements a modular, agentic AI pipeline for inferring Big Five personality traits and emotional states from multimodal interview-style data enriched with behavioral metadata (response time, body language, speech features).
The system integrates three coordinated agents:

Perception Agent – Emotion classification

Inference Agent – Big Five trait estimation

Dialogue Agent – Personality- and emotion-aware response generation

A retrieval-augmented memory module connects the agents, enabling context continuity and adaptive, psychologically informed dialogue.

✨ Features

Multimodal input enrichment (text + behavioral metadata)

Agentic workflow loop: Observe → Reflect → Act → Self-Audit

Dual LLM backbone support: LLaMA 3.2 1B and Falcon-RW-1B

Big Five personality trait estimation (OCEAN model)

Emotionally adaptive dialogue generation

Retrieval-Augmented Generation (RAG) for contextual grounding

Benchmark evaluation with latency, diversity, and empathy metrics


🏗 System Architecture
User Input + Metadata
       │
       ▼
 Perception Agent  →  Emotion Classification
       │
       ▼
 Inference Agent   →  Big Five Trait Scoring
       │
       ▼
 Retrieval Memory  →  Context from past interactions
       │
       ▼
 Dialogue Agent    →  Empathic, trait-aware response
