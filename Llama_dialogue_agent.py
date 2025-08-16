# import os
# import re
# import json
# import torch
# import numpy as np
# import logging
# import time
# from typing import Optional, Dict, Any, List
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics.pairwise import cosine_similarity

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# os.environ["HUGGINGFACE_TOKEN"] = "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"

# # ---------------- Configuration ----------------
# class ModelConfig:
#     """Configuration class for model parameters"""
#     def __init__(self):
#         self.model_name       = "meta-llama/Llama-3.2-1B-Instruct"
#         self.max_length       = 512
#         self.max_new_tokens   = 128
#         self.temperature      = 0.7
#         self.top_p            = 0.9
#         self.emotions         = [
#             "Happy", "Sad", "Angry", "Anxious", "Surprised",
#             "Disgusted", "Confused", "Calm", "Excited",
#             "Embarrassed", "Guilty", "Neutral"
#         ]
#         self.big_five_traits  = [
#             "openness", "conscientiousness", "extraversion",
#             "agreeableness", "neuroticism"
#         ]

# # ---------------- Metadata Extraction ----------------
# class MetadataExtractor:
#     """Automatically extracts metadata from text and context"""
#     def __init__(self):
#         self.quick_indicators = ['quick', 'fast', 'immediately', 'instantly']
#         self.slow_indicators  = ['slow', 'think', '...', 'hmm', 'well']
#         self.emotion_body_map = {
#             'Happy': 'smiling, relaxed posture',
#             'Sad': 'slouched, head down',
#             'Angry': 'tense, crossed arms',
#             'Anxious': 'fidgeting, restless',
#             'Surprised': 'raised eyebrows, open mouth',
#             'Disgusted': 'wrinkled nose, turned away',
#             'Confused': 'tilted head, furrowed brow',
#             'Calm': 'relaxed, steady posture',
#             'Excited': 'animated gestures',
#             'Embarrassed': 'blushing, looking away',
#             'Guilty': 'avoiding eye contact, withdrawn',
#             'Neutral': 'normal posture'
#         }

#     def extract_response_time(self, text: str) -> str:
#         tl = text.lower()
#         if any(ind in tl for ind in self.quick_indicators):
#             return "fast"
#         if any(ind in tl for ind in self.slow_indicators):
#             return "slow"
#         return "normal"

#     def extract_body_language(self, emotion: str) -> str:
#         return self.emotion_body_map.get(emotion, "normal")

#     def extract_speech_attributes(self, text: str) -> str:
#         if text.isupper():
#             return "loud"
#         if text.count('!') > 1:
#             return "excited"
#         if '...' in text or text.count('?') > 1:
#             return "hesitant"
#         return "clear"

# # ---------------- Perception Module ----------------
# class LlamaEmotionClassifier:
#     """Emotion classification using Llama model"""

#     def __init__(self, config: ModelConfig, hf_token: Optional[str] = None):
#         self.config = config
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
#         if token:
#             try:
#                 login(token=token)
#                 logger.info("Authenticated with HuggingFace")
#             except Exception as e:
#                 logger.warning(f"Login failed: {e}")

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             config.model_name, trust_remote_code=True, use_fast=True
#         )
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         self.model = AutoModelForCausalLM.from_pretrained(
#             config.model_name,
#             device_map={"": self.device},
#             trust_remote_code=True,
#             low_cpu_mem_usage=True,
#             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
#         )
#         self.model.eval()

#     def _create_prompt(self, text, rt, bl, sa) -> str:
#         emo_list = ", ".join(self.config.emotions)
#         return (
#             f"Classify the emotion from the following options: {emo_list}\n\n"
#             f"Text: \"{text}\"\n"
#             f"Response time: {rt}\n"
#             f"Body language: {bl}\n"
#             f"Speech attributes: {sa}\n\n"
#             f"Based on the text and context, the emotion is:"
#         )

#     def classify_single(self, text, rt, bl, sa) -> str:
#         prompt = self._create_prompt(text, rt, bl, sa)
#         inputs = self.tokenizer(
#             prompt, return_tensors="pt", truncation=True,
#             max_length=self.config.max_length
#         ).to(self.device)
#         with torch.inference_mode():
#             out = self.model.generate(
#                 **inputs,
#                 max_new_tokens=10,
#                 do_sample=False,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id
#             )
#         resp = self.tokenizer.decode(
#             out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True
#         ).strip()
#         pattern = r"\b(" + "|".join(self.config.emotions) + r")\b"
#         m = re.search(pattern, resp, re.IGNORECASE)
#         return m.group(1).capitalize() if m else "Neutral"

# # ---------------- Inference Module ----------------
# class LlamaInferenceAgent:
#     """Big Five personality trait inference using Llama model"""

#     def __init__(self, config: ModelConfig, hf_token: Optional[str] = None):
#         self.config = config
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
#         if token:
#             try:
#                 login(token=token)
#             except:
#                 pass

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             config.model_name, trust_remote_code=True, use_fast=False
#         )
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         self.model = AutoModelForCausalLM.from_pretrained(
#             config.model_name,
#             device_map={"": self.device},
#             torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
#             trust_remote_code=True,
#             low_cpu_mem_usage=True
#         )
#         self.model.eval()

#     def _build_prompt(self, text, rt, bl, sa, emo) -> str:
#         desc = {
#             'openness': 'creativity and openness to new experiences',
#             'conscientiousness': 'organization and self-discipline',
#             'extraversion': 'sociability and assertiveness',
#             'agreeableness': 'cooperation and trust in others',
#             'neuroticism': 'emotional instability and anxiety'
#         }
#         return (
#             f"<|system|>\nYou are a JSON generator. Output EXACTLY one JSON object—no explanation, no extra text.\n\n"
#             f"<|user|>\nBased on this text, provide Big Five personality scores (0.0 to 1.0) in JSON:\n\n"
#             f"Text: \"{text}\"\n"
#             f"Emotion: {emo}\n"
#             f"Context: response time {rt}, body language {bl}, speech {sa}\n\n"
#             f"- Openness ({desc['openness']}):\n"
#             f"- Conscientiousness ({desc['conscientiousness']}):\n"
#             f"- Extraversion ({desc['extraversion']}):\n"
#             f"- Agreeableness ({desc['agreeableness']}):\n"
#             f"- Neuroticism ({desc['neuroticism']}):\n\n"
#             f"Return ONLY a JSON object, e.g. {{\"openness\":0.7,\"conscientiousness\":0.6,\"extraversion\":0.5,\"agreeableness\":0.8,\"neuroticism\":0.3}}\n\n"
#             f"<|assistant|>\n"
#         )

#     def _extract_balanced_json(self, text: str) -> Optional[Dict]:
#         """Extract first balanced JSON object using regex"""
#         # Pattern to match balanced braces - simplified version
#         pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
#         match = re.search(pattern, text)
#         if match:
#             try:
#                 return json.loads(match.group(0))
#             except json.JSONDecodeError:
#                 return None
#         return None

#     def _heuristic_scoring(self, text, emo, rt, bl, sa) -> Dict[str, float]:
#         tl = text.lower()
#         scores = {t: 0.5 for t in self.config.big_five_traits}
        
#         # Basic heuristics based on text content and emotion
#         if emo in ['Happy', 'Excited']:
#             scores['extraversion'] = min(0.8, scores['extraversion'] + 0.2)
#             scores['neuroticism'] = max(0.2, scores['neuroticism'] - 0.2)
#         elif emo in ['Sad', 'Anxious']:
#             scores['neuroticism'] = min(0.8, scores['neuroticism'] + 0.2)
#             scores['extraversion'] = max(0.2, scores['extraversion'] - 0.1)
#         elif emo == 'Angry':
#             scores['neuroticism'] = min(0.9, scores['neuroticism'] + 0.3)
#             scores['agreeableness'] = max(0.1, scores['agreeableness'] - 0.3)
        
#         # Response time heuristics
#         if rt == 'fast':
#             scores['extraversion'] = min(0.9, scores['extraversion'] + 0.1)
#         elif rt == 'slow':
#             scores['conscientiousness'] = min(0.8, scores['conscientiousness'] + 0.1)
#             scores['openness'] = min(0.8, scores['openness'] + 0.1)
        
#         # Creative/analytical words
#         creative_words = ['creative', 'art', 'music', 'imagine', 'dream']
#         if any(word in tl for word in creative_words):
#             scores['openness'] = min(0.9, scores['openness'] + 0.2)
        
#         return scores

#     def infer_traits(self, text, rt, bl, sa, emo) -> Dict[str, float]:
#         try:
#             heur = self._heuristic_scoring(text, emo, rt, bl, sa)

#             prompt = self._build_prompt(text, rt, bl, sa, emo)
#             inputs = self.tokenizer(
#                 prompt, return_tensors="pt", truncation=True, max_length=2048
#             ).to(self.device)
            
#             with torch.no_grad():
#                 out = self.model.generate(
#                     **inputs,
#                     max_new_tokens=80,  # Reduced for JSON-only output
#                     do_sample=False,    # Deterministic decoding
#                     temperature=0.0,    # No randomness
#                     pad_token_id=self.tokenizer.pad_token_id,
#                     eos_token_id=self.tokenizer.eos_token_id
#                 )
            
#             gen = self.tokenizer.decode(
#                 out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True
#             ).strip()
            
#             logger.info(f"LLM trait response:\n{gen}")

#             # Use robust JSON extraction
#             llm = self._extract_balanced_json(gen)
#             if llm is None:
#                 logger.warning("JSON extraction failed, falling back to heuristic")
#                 llm = {}

#             final = {}
#             for t in self.config.big_five_traits:
#                 h = heur.get(t, 0.5)
#                 l = float(llm.get(t, h)) if llm else h
#                 # Ensure valid range and blend heuristic with LLM
#                 final[t] = max(0.0, min(1.0, 0.6 * h + 0.4 * l))
            
#             return final

#         except Exception as e:
#             logger.error(f"Trait inference error: {e}")
#             return {t: 0.5 for t in self.config.big_five_traits}

# # ---------------- Retrieval Module ----------------
# class Retriever:
#     """Retrieval-augmented memory for similar interactions"""

#     def __init__(self, data_path: str, emb_path: str):
#         self.data_path = Path(data_path)
#         self.emb_path  = Path(emb_path)
#         self.data      = []
#         self.full_emb  = np.empty((0,0))
#         self.vectorizer= None
#         self.encoder   = None

#         if self.data_path.exists() and self.emb_path.exists():
#             try:
#                 self.data     = json.loads(self.data_path.read_text(encoding='utf-8'))
#                 self.full_emb = np.load(self.emb_path)
#                 texts = [e['transcript'] for e in self.data]
#                 self.vectorizer = TfidfVectorizer(max_features=1000).fit(texts)
#                 emos   = [[e['emotion']] for e in self.data]
#                 self.encoder    = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(emos)
#                 logger.info(f"Loaded {len(self.data)} interactions")
#             except Exception as e:
#                 logger.error(f"Load error: {e}")
#                 self._reset_data()
#         else:
#             logger.info("No existing data—starting fresh")
#             self._reset_data()

#     def _reset_data(self):
#         self.data       = []
#         self.full_emb   = np.empty((0,0))
#         self.vectorizer = None
#         self.encoder    = None

#     def embed_query(self, text: str, emotion: str, traits: List[float]) -> np.ndarray:
#         if self.vectorizer:
#             te = self.vectorizer.transform([text]).toarray()
#         else:
#             te = np.zeros((1,1000))
#         if self.encoder:
#             ee = self.encoder.transform([[emotion]])
#         else:
#             ee = np.zeros((1,len(ModelConfig().emotions)))
#         tr = np.array([traits])
#         emb = np.hstack([te, ee, tr])[0]
#         if self.full_emb.size>0:
#             D = self.full_emb.shape[1]
#             if len(emb)>D: emb = emb[:D]
#             elif len(emb)<D: emb = np.concatenate([emb, np.zeros(D-len(emb))])
#         return emb

#     def get_top_k(self, query_emb: np.ndarray, k: int=3) -> List[Dict[str,Any]]:
#         if not self.data:
#             logger.info("No past interactions—skipping retrieval")
#             return []
#         if query_emb.shape[0] != self.full_emb.shape[1]:
#             logger.warning("Dim mismatch—recomputing embeddings")
#             self._update_embeddings()
#             if query_emb.shape[0] != self.full_emb.shape[1]:
#                 logger.error("Still mismatch—aborting retrieval")
#                 return []
#         sims = cosine_similarity([query_emb], self.full_emb)[0]
#         idxs= sims.argsort()[-k:][::-1]
#         results = [self.data[i] for i in idxs if sims[i]>0.01]
#         logger.info(f"Retrieved {len(results)} examples (scores {[round(sims[i],2) for i in idxs[:len(results)]]})")
#         return results

#     def add_interaction(self, text, emotion, traits, response):
#         self.data.append({
#             "transcript": text,
#             "emotion": emotion,
#             "traits": traits,
#             "response": response,
#             "timestamp": time.time()
#         })
#         self._update_embeddings()
#         self.save_data()

#     def _update_embeddings(self):
#         texts = [e['transcript'] for e in self.data]
#         emos  = [[e['emotion']] for e in self.data]
#         self.vectorizer = TfidfVectorizer(max_features=1000).fit(texts)
#         self.encoder    = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(emos)
#         embs = [ self.embed_query(d['transcript'], d['emotion'], list(d['traits'].values()))
#                  for d in self.data ]
#         self.full_emb = np.vstack(embs) if embs else np.empty((0,0))
#         logger.info(f"Embeddings updated: {self.full_emb.shape}")

#     def save_data(self):
#         try:
#             self.data_path.write_text(json.dumps(self.data, indent=2), encoding='utf-8')
#             if self.full_emb.size>0:
#                 np.save(self.emb_path, self.full_emb)
#             logger.info("Retrieval data saved")
#         except Exception as e:
#             logger.error(f"Save error: {e}")

# # ---------------- Dialogue Module & Main Loop ----------------
# class LlamaDialogueAgent:
#     def __init__(self, data_path='aligned_data_with_traits.json', emb_path='full_embeddings.npy',
#                  hf_token: Optional[str]=None, auto_metadata: bool=True):
#         self.config            = ModelConfig()
#         self.device            = "cuda" if torch.cuda.is_available() else "cpu"
#         self.auto_metadata     = auto_metadata
#         self.metadata_extractor= MetadataExtractor()
#         self.perceptor         = LlamaEmotionClassifier(self.config, hf_token)
#         self.inferencer        = LlamaInferenceAgent(self.config, hf_token)
#         self.retriever         = Retriever(data_path, emb_path)
#         token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
#         if token:
#             try: login(token=token)
#             except: pass
#         self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.model_name,
#             device_map={"": self.device},
#             torch_dtype=torch.bfloat16 if self.device=="cuda" else torch.float32,
#             trust_remote_code=True,
#             low_cpu_mem_usage=True
#         )
#         self.model.eval()

#     def chat(self, user_input: str, response_time: Optional[str]=None,
#              body_language: Optional[str]=None, speech_attributes: Optional[str]=None,
#              top_k: int=3) -> Dict[str,Any]:

#         start = time.time()
#         if self.auto_metadata:
#             response_time    = response_time    or self.metadata_extractor.extract_response_time(user_input)
#             speech_attributes= speech_attributes or self.metadata_extractor.extract_speech_attributes(user_input)
#         else:
#             response_time    = response_time    or "normal"
#             speech_attributes= speech_attributes or "clear"

#         emotion = self.perceptor.classify_single(
#             user_input, response_time, body_language or "normal", speech_attributes
#         )
#         if self.auto_metadata and not body_language:
#             body_language = self.metadata_extractor.extract_body_language(emotion)
#         elif not body_language:
#             body_language = "normal"

#         traits = self.inferencer.infer_traits(
#             user_input, response_time, body_language, speech_attributes, emotion
#         )

#         q_emb    = self.retriever.embed_query(user_input, emotion, list(traits.values()))
#         examples = self.retriever.get_top_k(q_emb, k=top_k)

#         context = ""
#         if examples:
#             context = "Similar past interactions:\n" + "\n".join(
#                 f"- {ex['transcript']} (Emotion: {ex['emotion']})"
#                 for ex in examples
#             )

#         system_p = "You are an empathetic assistant. Respond based on the user's emotion and traits."
#         prompt = (
#             f"<|system|>\n{system_p}\n"
#             f"Emotion: {emotion}\nTraits: {traits}\n{context}\n"
#             f"<|user|>\n{user_input}\n<|assistant|>"
#         )

#         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
#         with torch.no_grad():
#             out = self.model.generate(
#                 **inputs,
#                 max_new_tokens=self.config.max_new_tokens,
#                 do_sample=True,
#                 temperature=self.config.temperature,
#                 top_p=self.config.top_p,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id
#             )
#         response = self.tokenizer.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

#         self.retriever.add_interaction(user_input, emotion, traits, response)
#         proc_time = round(time.time() - start, 2)

#         return {
#             'response': response,
#             'emotion': emotion,
#             'traits': traits,
#             'metadata': {
#                 'response_time': response_time,
#                 'body_language': body_language,
#                 'speech_attributes': speech_attributes,
#                 'auto_metadata': self.auto_metadata,
#                 'processing_time': proc_time
#             },
#             'similar_examples': len(examples)
#         }

#     def save_session(self):
#         self.retriever.save_data()

# def main():
#     print("🤖 Dialogue System Ready!")
#     agent = LlamaDialogueAgent(auto_metadata=True)
#     print("Type 'exit' to quit.\n")

#     while True:
#         ui = input("You: ").strip()
#         if ui.lower() in ('exit','quit'):
#             agent.save_session()
#             break
#         rt = input("Response time (Enter to auto): ").strip() or None
#         bl = input("Body language (Enter to auto): ").strip() or None
#         sa = input("Speech attrs (Enter to auto): ").strip() or None
#         print("Processing...")
#         res = agent.chat(ui, rt, bl, sa)
#         print(f"\nAssistant: {res['response']}")
#         print(f"Emotion: {res['emotion']}, Traits: {res['traits']}, Sim examples: {res['similar_examples']}")
#         print(f"Time: {res['metadata']['processing_time']}s\n")

# if __name__ == "__main__":
#     main()





import os
import re
import json
import torch
import numpy as np
import logging
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- Configuration ----------------
class ModelConfig:
    """Configuration class for model parameters"""
    def __init__(self):
        self.model_name       = "meta-llama/Llama-3.2-1B-Instruct"
        self.max_length       = 512
        self.max_new_tokens   = 128
        self.temperature      = 0.7
        self.top_p            = 0.9
        self.emotions         = [
            "Happy", "Sad", "Angry", "Anxious", "Surprised",
            "Disgusted", "Confused", "Calm", "Excited",
            "Embarrassed", "Guilty", "Neutral"
        ]
        self.big_five_traits  = [
            "openness", "conscientiousness", "extraversion",
            "agreeableness", "neuroticism"
        ]

# ---------------- Metadata Extraction ----------------
class MetadataExtractor:
    """Automatically extracts metadata from text and context"""
    def __init__(self):
        self.quick_indicators = ['quick', 'fast', 'immediately', 'instantly']
        self.slow_indicators  = ['slow', 'think', '...', 'hmm', 'well']
        self.emotion_body_map = {
            'Happy': 'smiling, relaxed posture',
            'Sad': 'slouched, head down',
            'Angry': 'tense, crossed arms',
            'Anxious': 'fidgeting, restless',
            'Surprised': 'raised eyebrows, open mouth',
            'Disgusted': 'wrinkled nose, turned away',
            'Confused': 'tilted head, furrowed brow',
            'Calm': 'relaxed, steady posture',
            'Excited': 'animated gestures',
            'Embarrassed': 'blushing, looking away',
            'Guilty': 'avoiding eye contact, withdrawn',
            'Neutral': 'normal posture'
        }

    def extract_response_time(self, text: str) -> str:
        tl = text.lower()
        if any(ind in tl for ind in self.quick_indicators):
            return "fast"
        if any(ind in tl for ind in self.slow_indicators):
            return "slow"
        return "normal"

    def extract_body_language(self, emotion: str) -> str:
        return self.emotion_body_map.get(emotion, "normal")

    def extract_speech_attributes(self, text: str) -> str:
        if text.isupper():
            return "loud"
        if text.count('!') > 1:
            return "excited"
        if '...' in text or text.count('?') > 1:
            return "hesitant"
        return "clear"

class LlamaEmotionClassifier:
    """Simplified emotion classification using Llama model"""

    EMOTIONS = [
        "Happy", "Sad", "Angry", "Anxious", "Surprised",
        "Disgusted", "Confused", "Calm", "Excited", "Embarrassed", "Guilty", "Neutral"
    ]

    def __init__(self, config: ModelConfig, hf_token: Optional[str] = None):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model(hf_token)

    def _load_model(self, hf_token: Optional[str]):
        """Load tokenizer and model"""
        token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        if token:
            try:
                login(token=token)
                logger.info("Authenticated with HuggingFace")
            except Exception as e:
                logger.warning(f"Login failed: {e}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map={"": self.device},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.eval()

    def _create_prompt(self, text: str, rt: str, bl: str, sa: str) -> str:
        """Create classification prompt with better formatting"""
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are an emotion classifier. Respond with exactly ONE word from this list: "
            f"{', '.join(self.EMOTIONS)}.\n"
            f"Do not explain or add extra text. Just return the emotion word.\n"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"Text: \"{text}\"\n"
            f"Response time: {rt}\n"
            f"Body language: {bl}\n"
            f"Speech: {sa}\n"
            f"Classify this emotion:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    def _llm_classify(self, text: str, rt: str, bl: str, sa: str) -> Optional[str]:
        """Try LLM-based classification with improved parsing"""
        try:
            prompt = self._create_prompt(text, rt, bl, sa)
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=self.config.max_length
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,  # Reduced for single word response
                    do_sample=False,
                    temperature=0.0,   # More deterministic
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True
            ).strip()
            
            logger.info(f"Raw LLM response: '{response}'")
            
            # Clean the response
            response = response.replace("Emotion:", "").strip()
            response = re.sub(r'[^\w\s]', '', response)  # Remove punctuation
            response = response.split()[0] if response.split() else ""  # Take first word
            
            logger.info(f"Cleaned LLM response: '{response}'")
            
            # Check if response matches any emotion (case-insensitive)
            for emotion in self.EMOTIONS:
                if response.lower() == emotion.lower():
                    logger.info(f"LLM matched emotion: {emotion}")
                    return emotion
            
            # Fallback: partial matching
            for emotion in self.EMOTIONS:
                if emotion.lower() in response.lower() or response.lower() in emotion.lower():
                    logger.info(f"LLM partial match: {emotion}")
                    return emotion
                    
            logger.warning(f"LLM response '{response}' didn't match any emotion")
            return None
            
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return None

    def _rule_based_classify(self, text: str, rt: str, bl: str, sa: str) -> str:
        """Rule-based fallback classification"""
        text_lower = text.lower()
        bl_lower = bl.lower()
        sa_lower = sa.lower()
        rt_lower = rt.lower()
        
        scores = {emotion: 0 for emotion in self.EMOTIONS}
        
        # Keyword mapping
        emotion_keywords = {
            'Happy': ['happy', 'joy', 'excited', 'cheerful', 'glad', 'smile', 'laugh'],
            'Sad': ['sad', 'depressed', 'down', 'upset', 'cry', 'disappointed'],
            'Angry': ['angry', 'mad', 'furious', 'hate', 'frustrated'],
            'Anxious': ['anxious', 'worried', 'nervous', 'stressed', 'concerned'],
            'Surprised': ['surprised', 'shocked', 'amazed', 'unexpected'],
            'Confused': ['confused', 'puzzled', 'don\'t understand', 'unclear'],
            'Calm': ['calm', 'peaceful', 'relaxed', 'composed'],
            'Excited': ['thrilled', 'pumped', 'can\'t wait'],
        }
        
        # Score based on keywords
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[emotion] += 3
        
        # Body language cues
        if any(word in bl_lower for word in ['smile', 'eye contact']):
            scores['Happy'] += 2
        if any(word in bl_lower for word in ['slumped', 'hesitant']):
            scores['Sad'] += 2
        
        # Speech and response time cues
        if any(word in sa_lower for word in ['loud', 'raised']):
            scores['Angry'] += 2
        if 'fast' in rt_lower:
            scores['Excited'] += 1
        
        # Return highest scoring emotion or Neutral
        top_emotion = max(scores, key=scores.get)
        return top_emotion if scores[top_emotion] > 0 else 'Neutral'

    def classify_single(self, text: str, rt: str, bl: str, sa: str) -> str:
        """Classify emotion for single input"""
        # Try LLM first, fallback to rules
        emotion = self._llm_classify(text, rt, bl, sa)
        if emotion:
            logger.info(f"LLM classified emotion: {emotion}")
            return emotion
        
        emotion = self._rule_based_classify(text, rt, bl, sa)
        logger.info(f"Rule-based emotion: {emotion}")
        return emotion

    # Alias for compatibility
    infer_emotion = classify_single

# ---------------- Complete Fixed Inference Module ----------------
class LlamaInferenceAgent:
    """Big Five personality trait inference using Llama model with improved heuristic + LLM blending."""

    def __init__(self, config: ModelConfig, hf_token: Optional[str] = None):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        if token:
            try:
                login(token=token)
                logger.info("Authenticated with HuggingFace for inference")
            except Exception as e:
                logger.warning(f"HF login failed for inference: {e}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True, use_fast=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map={"": self.device},
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model.eval()

    def _build_prompt(self, text, rt, bl, sa, emo) -> str:
        """Improved prompt with clear scoring guidelines and examples"""
        return (
            "<|system|>\n"
            "You are a personality assessment expert. Analyze the given text and context to determine Big Five personality trait scores.\n"
            "Provide scores from 0.0 (very low) to 1.0 (very high) for each trait.\n"
            "Return ONLY a valid JSON object with no additional text.\n\n"
            "<|user|>\n"
            f"Text: \"{text}\"\n"
            f"Emotion: {emo}\n"
            f"Context: Response time: {rt}, Body language: {bl}, Speech: {sa}\n\n"
            "Scoring guidelines:\n"
            "- Openness (0.0-1.0): Creativity, curiosity, openness to new experiences\n"
            "  * Simple greetings: 0.4-0.6 (neutral)\n"
            "  * Creative/artistic content: 0.7-0.9\n"
            "  * Routine/conventional: 0.2-0.4\n\n"
            "- Conscientiousness (0.0-1.0): Organization, self-discipline, reliability\n"
            "  * Well-structured responses: 0.6-0.8\n"
            "  * Casual/informal: 0.3-0.5\n"
            "  * Detailed/methodical: 0.7-0.9\n\n"
            "- Extraversion (0.0-1.0): Sociability, assertiveness, energy\n"
            "  * Friendly greetings: 0.6-0.8\n"
            "  * Confident/assertive tone: 0.7-0.9\n"
            "  * Withdrawn/quiet: 0.2-0.4\n\n"
            "- Agreeableness (0.0-1.0): Cooperation, trust, empathy\n"
            "  * Polite greetings: 0.6-0.8\n"
            "  * Helpful/considerate: 0.7-0.9\n"
            "  * Confrontational: 0.1-0.3\n\n"
            "- Neuroticism (0.0-1.0): Emotional instability, anxiety, stress\n"
            "  * Calm/confident: 0.1-0.3\n"
            "  * Anxious/worried: 0.7-0.9\n"
            "  * Neutral/stable: 0.3-0.5\n\n"
            "Example JSON format:\n"
            "{\"openness\":0.5,\"conscientiousness\":0.6,\"extraversion\":0.7,\"agreeableness\":0.8,\"neuroticism\":0.2}\n\n"
            "<|assistant|>\n"
        )

    def _extract_balanced_json(self, text: str) -> Optional[Dict]:
        """Extract JSON with better validation and multiple fallback patterns"""
        # Clean the text first
        text = text.strip()
        
        # Try multiple patterns
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Original nested pattern
            r'\{[^\}]+\}',  # Simple pattern
            r'({[^{}]*"[^"]*"[^{}]*:[^{}]*[0-9.]+[^{}]*})',  # Pattern with quotes and numbers
            r'(\{.*?\})'  # Greedy pattern
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    # Validate that it has all expected keys and valid values
                    if (isinstance(data, dict) and 
                        all(trait in data for trait in self.config.big_five_traits) and
                        all(isinstance(data[trait], (int, float)) and 0.0 <= data[trait] <= 1.0 
                            for trait in self.config.big_five_traits)):
                        logger.info(f"Successfully extracted JSON: {data}")
                        return data
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.debug(f"JSON parse failed for pattern {pattern}: {e}")
                    continue
        
        # If no valid JSON found, try to extract scores manually
        logger.warning("Attempting manual score extraction from LLM response")
        scores = {}
        
        for trait in self.config.big_five_traits:
            # Look for various patterns like "openness": 0.7, "openness":0.7, openness: 0.7
            patterns_for_trait = [
                rf'"{trait}":\s*([0-9.]+)',
                rf"'{trait}':\s*([0-9.]+)",
                rf'{trait}:\s*([0-9.]+)',
                rf'{trait}.*?([0-9]\.[0-9]+)',
                rf'{trait}.*?([01]\.[0-9]+)'
            ]
            
            for pattern in patterns_for_trait:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        if 0.0 <= score <= 1.0:
                            scores[trait] = score
                            break
                    except (ValueError, IndexError):
                        continue
        
        if len(scores) == len(self.config.big_five_traits):
            logger.info(f"Manual extraction successful: {scores}")
            return scores
        
        logger.error(f"Failed to extract valid scores. Found: {scores}")
        return None

    def _heuristic_scoring(self, text, emo, rt, bl, sa) -> Dict[str, float]:
        """Enhanced heuristic scoring with more accurate psychological baselines"""
        tl = text.lower().strip()
        
        # Start with neutral baselines
        scores = {
            'openness': 0.5,
            'conscientiousness': 0.5, 
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.3  # Lower baseline for neuroticism (most people are stable)
        }
        
        # === TEXT CONTENT ANALYSIS ===
        # Greeting patterns suggest social engagement
        greeting_words = ['hey', 'hello', 'hi', 'good morning', 'good afternoon', 'good evening', 'how are you']
        if any(word in tl for word in greeting_words):
            scores['extraversion'] = min(0.8, scores['extraversion'] + 0.2)
            scores['agreeableness'] = min(0.8, scores['agreeableness'] + 0.2)
            scores['neuroticism'] = max(0.1, scores['neuroticism'] - 0.1)
        
        # Positive language
        positive_words = ['great', 'awesome', 'fantastic', 'wonderful', 'amazing', 'love', 'enjoy']
        if any(word in tl for word in positive_words):
            scores['extraversion'] = min(0.9, scores['extraversion'] + 0.2)
            scores['agreeableness'] = min(0.8, scores['agreeableness'] + 0.1)
            scores['neuroticism'] = max(0.1, scores['neuroticism'] - 0.2)
        
        # Creative/artistic content
        creative_words = ['creative', 'art', 'music', 'imagine', 'dream', 'poetry', 'design', 'paint', 'write']
        if any(w in tl for w in creative_words):
            scores['openness'] = min(0.9, scores['openness'] + 0.3)
        
        # Analytical/structured language
        analytical_words = ['analyze', 'structure', 'plan', 'organize', 'systematic', 'logical']
        if any(w in tl for w in analytical_words):
            scores['conscientiousness'] = min(0.8, scores['conscientiousness'] + 0.2)
            scores['openness'] = min(0.8, scores['openness'] + 0.1)
        
        # Uncertainty/confusion markers
        uncertainty_words = ['confused', 'unsure', 'maybe', 'perhaps', 'i think', 'not sure']
        if any(w in tl for w in uncertainty_words):
            scores['neuroticism'] = min(0.7, scores['neuroticism'] + 0.2)
            scores['conscientiousness'] = max(0.3, scores['conscientiousness'] - 0.1)
        
        # === EMOTION-BASED ADJUSTMENTS ===
        if emo in ['Happy', 'Excited']:
            scores['extraversion'] = min(0.9, scores['extraversion'] + 0.3)
            scores['agreeableness'] = min(0.8, scores['agreeableness'] + 0.2)
            scores['neuroticism'] = max(0.1, scores['neuroticism'] - 0.2)
        elif emo in ['Sad', 'Anxious']:
            scores['neuroticism'] = min(0.8, scores['neuroticism'] + 0.4)
            scores['extraversion'] = max(0.2, scores['extraversion'] - 0.2)
        elif emo == 'Angry':
            scores['neuroticism'] = min(0.9, scores['neuroticism'] + 0.4)
            scores['agreeableness'] = max(0.1, scores['agreeableness'] - 0.4)
        elif emo == 'Calm':
            scores['neuroticism'] = max(0.1, scores['neuroticism'] - 0.3)
            scores['conscientiousness'] = min(0.8, scores['conscientiousness'] + 0.2)
        elif emo == 'Confused':
            scores['neuroticism'] = min(0.7, scores['neuroticism'] + 0.2)
            scores['openness'] = max(0.3, scores['openness'] - 0.1)  # Confusion may indicate less openness
        
        # === RESPONSE TIME ANALYSIS ===
        if rt == 'fast':
            scores['extraversion'] = min(0.9, scores['extraversion'] + 0.2)
            scores['neuroticism'] = max(0.1, scores['neuroticism'] - 0.1)
        elif rt == 'slow':
            scores['conscientiousness'] = min(0.8, scores['conscientiousness'] + 0.2)
            scores['openness'] = min(0.8, scores['openness'] + 0.1)  # Thoughtfulness
        
        # === SPEECH ATTRIBUTES ===
        if sa == 'confident':
            scores['extraversion'] = min(0.9, scores['extraversion'] + 0.3)
            scores['neuroticism'] = max(0.1, scores['neuroticism'] - 0.2)
        elif sa == 'hesitant':
            scores['neuroticism'] = min(0.8, scores['neuroticism'] + 0.3)
            scores['extraversion'] = max(0.2, scores['extraversion'] - 0.2)
        elif sa == 'excited':
            scores['extraversion'] = min(0.9, scores['extraversion'] + 0.2)
            scores['openness'] = min(0.8, scores['openness'] + 0.1)
        
        # === BODY LANGUAGE ===
        if bl:
            bl_lower = bl.lower()
            if any(indicator in bl_lower for indicator in ['confident', 'steady', 'eye contact', 'leaning forward']):
                scores['extraversion'] = min(0.9, scores['extraversion'] + 0.2)
                scores['neuroticism'] = max(0.1, scores['neuroticism'] - 0.1)
            elif any(indicator in bl_lower for indicator in ['fidgeting', 'nervous', 'avoiding', 'withdrawn']):
                scores['neuroticism'] = min(0.8, scores['neuroticism'] + 0.3)
                scores['extraversion'] = max(0.2, scores['extraversion'] - 0.2)
            elif any(indicator in bl_lower for indicator in ['relaxed', 'calm', 'composed']):
                scores['neuroticism'] = max(0.1, scores['neuroticism'] - 0.2)
                scores['agreeableness'] = min(0.8, scores['agreeableness'] + 0.1)
        
        # === TEXT LENGTH ANALYSIS ===
        if len(text.split()) > 20:  # Longer responses
            scores['openness'] = min(0.8, scores['openness'] + 0.1)
            scores['extraversion'] = min(0.8, scores['extraversion'] + 0.1)
        elif len(text.split()) < 5:  # Very short responses
            scores['extraversion'] = max(0.3, scores['extraversion'] - 0.1)
        
        # Ensure all scores are within valid range
        for trait in scores:
            scores[trait] = max(0.0, min(1.0, scores[trait]))
        
        logger.info(f"Heuristic scores: {scores}")
        return scores

    def infer_traits(self, text, rt, bl, sa, emo) -> Dict[str, float]:
        """Improved trait inference with better error handling and validation"""
        try:
            # 1) Get heuristic baseline
            heur_scores = self._heuristic_scoring(text, emo, rt, bl, sa)
            
            # 2) Generate LLM prompt and get response
            prompt = self._build_prompt(text, rt, bl, sa, emo)
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.device)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=120,  # Increased for better JSON generation
                    do_sample=False,
                    temperature=0.0,  # Deterministic for consistency
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1  # Prevent repetitive output
                )

            gen_text = self.tokenizer.decode(
                out[0][inputs['input_ids'].shape[-1]:],
                skip_special_tokens=True
            ).strip()
            
            logger.info(f"LLM trait response: {gen_text}")

            # 3) Extract and validate JSON from LLM response
            llm_scores = self._extract_balanced_json(gen_text)
            
            if not llm_scores:
                logger.warning("Failed to extract valid JSON from LLM response, using heuristic scores only")
                return heur_scores
            
            # 4) Blend heuristic and LLM scores
            final_scores = {}
            for trait in self.config.big_five_traits:
                h_score = heur_scores[trait]
                l_score = llm_scores.get(trait, h_score)
                
                # Additional validation for LLM score
                if not isinstance(l_score, (int, float)) or not (0.0 <= l_score <= 1.0):
                    logger.warning(f"Invalid LLM score for {trait}: {l_score}, using heuristic score")
                    l_score = h_score
                
                # Blend: 50% heuristic + 50% LLM (balanced approach)
                # For very confident heuristics (extreme values), give them more weight
                heuristic_weight = 0.6 if (h_score < 0.2 or h_score > 0.8) else 0.5
                llm_weight = 1.0 - heuristic_weight
                
                blended_score = heuristic_weight * h_score + llm_weight * float(l_score)
                final_scores[trait] = max(0.0, min(1.0, blended_score))

            logger.info(f"Final blended scores: {final_scores}")
            return final_scores

        except Exception as e:
            logger.error(f"Trait inference error: {e}")
            # Return neutral scores as fallback
            fallback_scores = {trait: 0.5 for trait in self.config.big_five_traits}
            fallback_scores['neuroticism'] = 0.3  # Slightly lower baseline for neuroticism
            return fallback_scores


# ---------------- Fixed Retrieval Module ----------------
class Retriever:
    """Retrieval-augmented memory for similar interactions"""

    def __init__(self, original_data_path: str, original_emb_path: str, 
                 new_data_path: str = 'llama_aligned_traits.json', 
                 new_emb_path: str = 'llama_embedding.npy'):
        # Original files for existing data
        self.original_data_path = Path(original_data_path)
        self.original_emb_path  = Path(original_emb_path)
        
        # New files for storing new interactions
        self.new_data_path = Path(new_data_path)
        self.new_emb_path  = Path(new_emb_path)
        
        self.original_data = []
        self.new_data      = []
        self.original_emb  = np.empty((0,0))
        self.new_emb       = np.empty((0,0))
        self.vectorizer    = None
        self.encoder       = None

        # Load original data for retrieval context
        self._load_original_data()
        
        # Load new data if exists
        self._load_new_data()

    def _load_original_data(self):
        """Load original data for retrieval context only"""
        if self.original_data_path.exists() and self.original_emb_path.exists():
            try:
                with open(self.original_data_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # Debug: Print the structure and check for empty transcripts
                if raw_data:
                    logger.info(f"Total entries in original data: {len(raw_data)}")
                    empty_transcripts = sum(1 for item in raw_data if not item.get('transcript', '').strip())
                    logger.info(f"Entries with empty transcripts: {empty_transcripts}")
                    
                    # Show first few entries with their transcript status
                    for i, item in enumerate(raw_data[:3]):
                        transcript = item.get('transcript', '')
                        logger.info(f"Entry {i}: transcript='{transcript[:50]}...' (length: {len(transcript)})")
                
                # Filter out entries with empty transcripts
                self.original_data = [
                    item for item in raw_data 
                    if item.get('transcript', '').strip()
                ]
                
                logger.info(f"Loaded {len(self.original_data)} original interactions with valid transcripts")
                
                # Load embeddings - but we need to align them with filtered data
                all_embeddings = np.load(self.original_emb_path)
                
                # If we filtered out some entries, we need to filter embeddings too
                if len(self.original_data) != len(raw_data):
                    logger.warning(f"Filtered {len(raw_data) - len(self.original_data)} entries with empty transcripts")
                    # Create a mapping of valid indices
                    valid_indices = [
                        i for i, item in enumerate(raw_data)
                        if item.get('transcript', '').strip()
                    ]
                    if len(valid_indices) <= all_embeddings.shape[0]:
                        self.original_emb = all_embeddings[valid_indices]
                        logger.info(f"Filtered embeddings to match valid data: {self.original_emb.shape}")
                    else:
                        logger.error("Mismatch between data and embeddings - using all embeddings")
                        self.original_emb = all_embeddings
                else:
                    self.original_emb = all_embeddings
                    
            except Exception as e:
                logger.error(f"Original data load error: {e}")
                self.original_data = []
                self.original_emb  = np.empty((0,0))
        else:
            logger.info("No original data found")
            self.original_data = []
            self.original_emb  = np.empty((0,0))



    def _load_new_data(self):
        """Load new interaction data"""
        if self.new_data_path.exists() and self.new_emb_path.exists():
            try:
                with open(self.new_data_path, 'r', encoding='utf-8') as f:
                    self.new_data = json.load(f)
                self.new_emb  = np.load(self.new_emb_path)
                logger.info(f"Loaded {len(self.new_data)} new interactions")
            except Exception as e:
                logger.error(f"New data load error: {e}")
                self._reset_new_data()
        else:
            logger.info("No existing new data—starting fresh")
            self._reset_new_data()

        # Initialize vectorizer and encoder with all available data
        self._initialize_vectorizer_encoder()

    def _reset_new_data(self):
        self.new_data = []
        self.new_emb  = np.empty((0,0))

    def _initialize_vectorizer_encoder(self):
        """Initialize vectorizer and encoder with all available data"""
        all_data = self.original_data + self.new_data
        if all_data:
            # Extract texts and emotions, ensuring we only use valid entries
            texts = []
            emotions = []
            for item in all_data:
                text = item.get('transcript', '').strip()
                if text:  # Only include non-empty texts
                    texts.append(text)
                    emotions.append([item.get('emotion', 'Neutral')])
            
            if texts:
                self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english').fit(texts)
                self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(emotions)
                logger.info(f"Vectorizer initialized with {len(texts)} valid texts")
            else:
                logger.warning("No valid texts found for vectorizer initialization")
                self.vectorizer = None
                self.encoder = None
        else:
            logger.warning("No data available for vectorizer initialization")
            self.vectorizer = None
            self.encoder = None

    def embed_query(self, text: str, emotion: str, traits: List[float]) -> np.ndarray:
        if self.vectorizer and text.strip():
            te = self.vectorizer.transform([text]).toarray()
        else:
            te = np.zeros((1, 1000))
            
        if self.encoder:
            ee = self.encoder.transform([[emotion]])
        else:
            ee = np.zeros((1, len(ModelConfig().emotions)))
            
        tr = np.array([traits])
        emb = np.hstack([te, ee, tr])[0]
        
        # Get combined embeddings for dimension checking
        combined_emb = self._get_combined_embeddings()
        if combined_emb.size > 0:
            D = combined_emb.shape[1]
            if len(emb) > D: 
                emb = emb[:D]
            elif len(emb) < D: 
                emb = np.concatenate([emb, np.zeros(D-len(emb))])
        return emb

    def _get_combined_embeddings(self) -> np.ndarray:
        """Combine original and new embeddings for retrieval"""
        if self.original_emb.size > 0 and self.new_emb.size > 0:
            return np.vstack([self.original_emb, self.new_emb])
        elif self.original_emb.size > 0:
            return self.original_emb
        elif self.new_emb.size > 0:
            return self.new_emb
        else:
            return np.empty((0,0))

    def get_top_k(self, query_emb: np.ndarray, k: int=3) -> List[Dict[str,Any]]:
        combined_data = self.original_data + self.new_data
        combined_emb = self._get_combined_embeddings()
        
        if not combined_data:
            logger.info("No past interactions—skipping retrieval")
            return []
        
        # Since we already filtered during loading, all data should have valid transcripts
        logger.info(f"Total available interactions: {len(combined_data)}")
        
        if combined_emb.size == 0:
            logger.warning("No embeddings available—skipping retrieval")
            return []
            
        if query_emb.shape[0] != combined_emb.shape[1]:
            logger.warning(f"Dimension mismatch: query {query_emb.shape[0]} vs embeddings {combined_emb.shape[1]}")
            logger.warning("Recomputing embeddings...")
            self._update_new_embeddings()
            combined_emb = self._get_combined_embeddings()
            if query_emb.shape[0] != combined_emb.shape[1]:
                logger.error(f"Still mismatch after recomputing: {query_emb.shape[0]} vs {combined_emb.shape[1]}")
                return []
        
        # Compute similarities
        sims = cosine_similarity([query_emb], combined_emb)[0]
        idxs = sims.argsort()[-k:][::-1]
        
        # Filter by similarity threshold and return results
        results = []
        for idx in idxs:
            if idx < len(combined_data) and sims[idx] > 0.01:
                item = combined_data[idx]
                # Ensure transcript exists and is not empty
                if item.get('transcript', '').strip():
                    results.append(item)
        
        logger.info(f"Retrieved {len(results)} examples with similarities: {[round(sims[i],3) for i in idxs[:len(results)]]}")
        
        # Debug: Print retrieved transcripts
        for i, result in enumerate(results):
            transcript = result.get('transcript', '')
            logger.info(f"Retrieved {i+1}: '{transcript[:100]}...' (emotion: {result.get('emotion', 'Unknown')})")
            
        return results

    def add_interaction(self, text, emotion, traits, response, processing_time):
        """Add new interaction to separate storage files"""
        self.new_data.append({
            "transcript": text,
            "emotion": emotion,
            "traits": traits,
            "response": response,
            "timestamp": time.time(),
            "processing_time": processing_time
        })
        self._update_new_embeddings()
        self.save_new_data()

    def _update_new_embeddings(self):
        """Update embeddings for new data only"""
        if not self.new_data:
            return
            
        # Re-initialize vectorizer and encoder with all data
        self._initialize_vectorizer_encoder()
        
        # Create embeddings for new data
        new_embs = []
        for d in self.new_data:
            transcript = d.get('transcript', '')
            if transcript.strip():  # Only process non-empty transcripts
                emb = self.embed_query(transcript, d.get('emotion', 'Neutral'), list(d.get('traits', {}).values()))
                new_embs.append(emb)
        
        self.new_emb = np.vstack(new_embs) if new_embs else np.empty((0,0))
        logger.info(f"New embeddings updated: {self.new_emb.shape}")

    def save_new_data(self):
        """Save only the new interaction data"""
        try:
            with open(self.new_data_path, 'w', encoding='utf-8') as f:
                json.dump(self.new_data, f, indent=2, ensure_ascii=False)
            if self.new_emb.size > 0:
                np.save(self.new_emb_path, self.new_emb)
            logger.info(f"New interaction data saved to {self.new_data_path} and {self.new_emb_path}")
        except Exception as e:
            logger.error(f"Save error: {e}")

# ---------------- Dialogue Module & Main Loop ----------------
class LlamaDialogueAgent:
    def __init__(self, original_data_path='aligned_data_with_traits.json', 
                 original_emb_path='full_embeddings.npy',
                 new_data_path='llama_aligned_traits.json',
                 new_emb_path='llama_embedding.npy',
                 hf_token: Optional[str]=None, auto_metadata: bool=True):
        self.config            = ModelConfig()
        self.device            = "cuda" if torch.cuda.is_available() else "cpu"
        self.auto_metadata     = auto_metadata
        self.metadata_extractor= MetadataExtractor()
        self.perceptor         = LlamaEmotionClassifier(self.config, hf_token)
        self.inferencer        = LlamaInferenceAgent(self.config, hf_token)
        self.retriever         = Retriever(original_data_path, original_emb_path, 
                                         new_data_path, new_emb_path)
        token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        if token:
            try: login(token=token)
            except: pass
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map={"": self.device},
            torch_dtype=torch.bfloat16 if self.device=="cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model.eval()

    def chat(self, user_input: str, response_time: Optional[str]=None,
             body_language: Optional[str]=None, speech_attributes: Optional[str]=None,
             top_k: int=3) -> Dict[str,Any]:

        start = time.time()
        proc_time = 0.0

        if self.auto_metadata:
            response_time    = response_time    or self.metadata_extractor.extract_response_time(user_input)
            speech_attributes= speech_attributes or self.metadata_extractor.extract_speech_attributes(user_input)
        else:
            response_time    = response_time    or "normal"
            speech_attributes= speech_attributes or "clear"

        emotion = self.perceptor.classify_single(
            user_input, response_time, body_language or "normal", speech_attributes
        )
        if self.auto_metadata and not body_language:
            body_language = self.metadata_extractor.extract_body_language(emotion)
        elif not body_language:
            body_language = "normal"

        traits = self.inferencer.infer_traits(
            user_input, response_time, body_language, speech_attributes, emotion
        )

        q_emb    = self.retriever.embed_query(user_input, emotion, list(traits.values()))
        examples = self.retriever.get_top_k(q_emb, k=top_k)

        context = ""
        if examples:
            context = "Similar past interactions:\n" + "\n".join(
                f"- {ex['transcript']} (Emotion: {ex['emotion']})"
                for ex in examples
            )

        system_p = "You are an empathetic assistant. Respond based on the user's emotion and traits."
        prompt = (
            f"<|system|>\n{system_p}\n"
            f"Emotion: {emotion}\nTraits: {traits}\n{context}\n"
            f"<|user|>\n{user_input}\n<|assistant|>"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

        # Store in new files instead of original files
        proc_time = round(time.time() - start, 2)
        self.retriever.add_interaction(user_input, emotion, traits, response, proc_time)

        return {
            'response': response,
            'emotion': emotion,
            'traits': traits,
            'metadata': {
                'response_time': response_time,
                'body_language': body_language,
                'speech_attributes': speech_attributes,
                'auto_metadata': self.auto_metadata,
                'processing_time': proc_time
            },
            'similar_examples': len(examples)
        }

    def save_session(self):
        """Save new interaction data to separate files"""
        self.retriever.save_new_data()

# def main():
#     print("🤖 Dialogue System Ready!")
#     print("📁 New interactions will be saved to 'llama_aligned_traits.json' and 'llama_embedding.npy'")
#     agent = LlamaDialogueAgent(auto_metadata=True)
#     print("Type 'exit' to quit.\n")

#     while True:
#         ui = input("You: ").strip()
#         if ui.lower() in ('exit','quit'):
#             agent.save_session()
#             break
#         rt = input("Response time (Enter to auto): ").strip() or None
#         bl = input("Body language (Enter to auto): ").strip() or None
#         sa = input("Speech attrs (Enter to auto): ").strip() or None
#         print("Processing...")
#         res = agent.chat(ui, rt, bl, sa)
#         print(f"\nAssistant: {res['response']}")
#         print(f"Emotion: {res['emotion']}, Traits: {res['traits']}, Sim examples: {res['similar_examples']}")
#         print(f"Time: {res['metadata']['processing_time']}s\n")

# if __name__ == "__main__":
#     main()

def main():
    """Batch-run LlamaDialogueAgent with test queries and store via existing agent.save_session()"""
    print("🦙 Improved LLaMA Dialogue System Ready for Batch Test!")
    print("⚠️  Make sure to set your HUGGINGFACE_TOKEN environment variable")
    print("💡 Batch mode: results will be saved in your existing LLaMA JSON files\n")
    
    try:
        agent = LlamaDialogueAgent(auto_metadata=True)
        
        # Load test queries from user_queries.json
        with open("user_queries.json", "r", encoding="utf-8") as f:
            test_queries = json.load(f)
        
        for i, query in enumerate(test_queries, 1):
            print(f"📝 Running query {i}/{len(test_queries)}: {query['transcript'][:60]}...")
            try:
                start_time = time.time()
                
                result = agent.chat(
                    user_input=query['transcript'],
                    response_time=query.get('response_time'),
                    body_language=query.get('body_language'),
                    speech_attributes=query.get('speech_attributes')
                )
                
                end_time = time.time()
                
                print(f"✅ Done. Assistant: {result['response'][:80]}...")
                print(f"📊 Emotion: {result['emotion']}, Traits: {result['traits']}")
                print(f"⏱️  Wall time: {round(end_time - start_time, 2)}s\n")
                
            except Exception as e:
                print(f"⚠️  Error processing query {i}: {e}")
                logger.error(f"Error processing query {i}: {e}")
        
        # trigger existing new-data save
        agent.save_session()
        print("🎉 All test queries processed and stored in your existing LLaMA JSON. Goodbye!")

    except Exception as e:
        print(f"Failed to initialize: {e}")
        logger.error(f"Initialization error: {e}")

if __name__ == "__main__":
    main()
