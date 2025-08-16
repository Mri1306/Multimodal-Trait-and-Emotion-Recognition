import json
import numpy as np
from collections import Counter

def distinct_2(text):
    words = text.split()
    bigrams = list(zip(words, words[1:]))
    if len(bigrams) == 0:
        return 0.0
    unique_bigrams = set(bigrams)
    return len(unique_bigrams) / len(bigrams)

# load files
with open("llama_aligned_traits.json", "r", encoding="utf-8") as f:
    llama_data = json.load(f)

with open("falcon_aligned_traits.json", "r", encoding="utf-8") as f:
    falcon_data = json.load(f)

# check equal lengths
if len(llama_data) != len(falcon_data):
    print(f"⚠️ WARNING: datasets not same length ({len(llama_data)} vs {len(falcon_data)})")
n = min(len(llama_data), len(falcon_data))

llama_latencies = []
falcon_latencies = []
llama_lengths = []
falcon_lengths = []
llama_d2 = []
falcon_d2 = []
trait_diffs = []

for i in range(n):
    l = llama_data[i]
    f = falcon_data[i]
    
    # latency
    llama_latencies.append(l.get("processing_time", 0.0))
    falcon_latencies.append(f.get("processing_time", 0.0))
    
    # response length
    llama_lengths.append(len(l.get("response", "").split()))
    falcon_lengths.append(len(f.get("response", "").split()))
    
    # diversity
    llama_d2.append(distinct_2(l.get("response", "")))
    falcon_d2.append(distinct_2(f.get("response", "")))
    
    # trait MAE
    l_traits = np.array([l["traits"].get(t, 0.5) for t in ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]])
    f_traits = np.array([f["traits"].get(t, 0.5) for t in ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]])
    mae = np.mean(np.abs(l_traits - f_traits))
    trait_diffs.append(mae)
    
    # optional side-by-side preview
    print(f"=== Query {i+1} ===")
    print(f"User: {l['transcript']}")
    print(f"--- LLaMA Response: {l['response'][:100]}...")
    print(f"--- Falcon Response: {f['response'][:100]}...\n")

print("\n=== MODEL COMPARISON RESULTS ===\n")
print(f"Total queries compared: {n}\n")

print("Average latency (seconds):")
print(f"  LLaMA  : {np.mean(llama_latencies):.2f}")
print(f"  Falcon : {np.mean(falcon_latencies):.2f}\n")

print("Average response length (words):")
print(f"  LLaMA  : {np.mean(llama_lengths):.2f}")
print(f"  Falcon : {np.mean(falcon_lengths):.2f}\n")

print("Average distinct-2 diversity:")
print(f"  LLaMA  : {np.mean(llama_d2):.3f}")
print(f"  Falcon : {np.mean(falcon_d2):.3f}\n")

print("Average mean absolute difference in Big Five traits:")
print(f"  {np.mean(trait_diffs):.3f}")

print("\n✅ Comparison complete!")
