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

# # ---------------- Configuration ----------------
# class ModelConfig:
#     def __init__(self):
#         self.model_name       = "tiiuae/falcon-rw-1b"
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

# # ---------------- Falcon-based Modules ----------------
# class FalconEmotionClassifier:
#     def __init__(self, config: ModelConfig, hf_token: Optional[str] = None):
#         self.config = config
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
#         if token:
#             try:
#                 login(token=token)
#             except Exception as e:
#                 logger.warning(f"Login failed: {e}")

#         self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         self.model = AutoModelForCausalLM.from_pretrained(
#             config.model_name,
#             device_map={"": self.device},
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
#         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(self.device)
#         with torch.inference_mode():
#             out = self.model.generate(
#                 **inputs,
#                 max_new_tokens=10,
#                 do_sample=False,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id
#             )
#         resp = self.tokenizer.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
#         pattern = r"\\b(" + "|".join(self.config.emotions) + r")\\b"
#         m = re.search(pattern, resp, re.IGNORECASE)
#         return m.group(1).capitalize() if m else "Neutral"

# class TraitInferencer:
#     def __init__(self, config: ModelConfig):
#         self.config = config

#     def infer_traits(self, text: str, emotion: str, rt: str, bl: str, sa: str) -> Dict[str, float]:
#         tl = text.lower()
#         scores = {t: 0.5 for t in self.config.big_five_traits}

#         if emotion in ['Happy', 'Excited']:
#             scores['extraversion'] += 0.2
#             scores['neuroticism'] -= 0.2
#         elif emotion in ['Sad', 'Anxious']:
#             scores['neuroticism'] += 0.2
#             scores['extraversion'] -= 0.1
#         elif emotion == 'Angry':
#             scores['neuroticism'] += 0.3
#             scores['agreeableness'] -= 0.3

#         if rt == 'fast':
#             scores['extraversion'] += 0.1
#         elif rt == 'slow':
#             scores['conscientiousness'] += 0.1
#             scores['openness'] += 0.1

#         creative_words = ['creative', 'art', 'music', 'imagine', 'dream']
#         if any(w in tl for w in creative_words):
#             scores['openness'] += 0.2

#         return {k: max(0.0, min(1.0, round(v, 2))) for k, v in scores.items()}

# class FalconDialogueAgent:
#     def __init__(self, data_path='Falcon_Aligned_data_withTraits.json', emb_path='Falconfull_embedding.npy', hf_token=None):
#         self.config = ModelConfig()
#         self.metadata_extractor = MetadataExtractor()
#         self.perceptor = FalconEmotionClassifier(self.config, hf_token)
#         self.trait_inferencer = TraitInferencer(self.config)
#         self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.model_name,
#             device_map="auto",
#             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
#         )
#         self.model.eval()

#         self.data_path = Path(data_path)
#         self.emb_path = Path(emb_path)
#         self.data = []
#         self.embeddings = np.empty((0, 0))
#         self.vectorizer = None
#         self.encoder = None

#         if self.data_path.exists() and self.emb_path.exists():
#             self.data = json.loads(self.data_path.read_text(encoding='utf-8'))
#             self.embeddings = np.load(self.emb_path)
#             texts = [e['transcript'] for e in self.data]
#             self.vectorizer = TfidfVectorizer(max_features=1000).fit(texts)
#             emos = [[e['emotion']] for e in self.data]
#             self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(emos)
#         else:
#             logger.info("Starting fresh: no previous data found.")

#     def _embed_query(self, text: str, emotion: str) -> np.ndarray:
#         te = self.vectorizer.transform([text]).toarray() if self.vectorizer else np.zeros((1, 1000))
#         ee = self.encoder.transform([[emotion]]) if self.encoder else np.zeros((1, len(self.config.emotions)))
#         emb = np.hstack([te, ee])[0]
#         if self.embeddings.size > 0:
#             D = self.embeddings.shape[1]
#             emb = np.resize(emb, (D,))
#         return emb

#     def _retrieve_similar(self, query_emb: np.ndarray, k=3) -> List[Dict[str, Any]]:
#         if not self.data or self.embeddings.size == 0:
#             return []
#         sims = cosine_similarity([query_emb], self.embeddings)[0]
#         idxs = sims.argsort()[-k:][::-1]
#         return [self.data[i] for i in idxs if sims[i] > 0.01]

#     def _save_data(self):
#         self.data_path.parent.mkdir(parents=True, exist_ok=True)
#         self.data_path.write_text(json.dumps(self.data, indent=2), encoding='utf-8')
#         if self.embeddings.size > 0:
#             np.save(self.emb_path, self.embeddings)

#     def _update_embeddings(self):
#         texts = [d['transcript'] for d in self.data]
#         emos = [[d['emotion']] for d in self.data]
#         self.vectorizer = TfidfVectorizer(max_features=1000).fit(texts)
#         self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(emos)
#         self.embeddings = np.vstack([
#             self._embed_query(d['transcript'], d['emotion']) for d in self.data
#         ])

#     def chat(self, user_input: str) -> str:
#         rt = self.metadata_extractor.extract_response_time(user_input)
#         sa = self.metadata_extractor.extract_speech_attributes(user_input)
#         emotion = self.perceptor.classify_single(user_input, rt, "normal", sa)
#         bl = self.metadata_extractor.extract_body_language(emotion)
#         traits = self.trait_inferencer.infer_traits(user_input, emotion, rt, bl, sa)

#         emb = self._embed_query(user_input, emotion)
#         retrieved = self._retrieve_similar(emb)
#         context = "\n".join([f"- {ex['transcript']} (Emotion: {ex['emotion']})" for ex in retrieved])

#         system_prompt = "You are a helpful assistant. Respond to the user based on emotion, traits, and similar examples."
#         prompt = (
#             f"<|system|>\n{system_prompt}\nEmotion: {emotion}\nTraits: {traits}\nBody: {bl}\nSpeech: {sa}\n{context}\n"
#             f"<|user|>\n{user_input}\n<|assistant|>"
#         )

#         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
#         with torch.no_grad():
#             out = self.model.generate(
#                 **inputs,
#                 max_new_tokens=self.config.max_new_tokens,
#                 do_sample=True,
#                 temperature=self.config.temperature,
#                 top_p=self.config.top_p
#             )
#         response = self.tokenizer.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

#         self.data.append({"transcript": user_input, "emotion": emotion, "traits": traits, "response": response})
#         self._update_embeddings()
#         self._save_data()

#         return response

# if __name__ == "__main__":
#     agent = FalconDialogueAgent()
#     print("🤖 Falcon Dialogue Agent Ready!")
#     while True:
#         text = input("You: ").strip()
#         if text.lower() in ("exit", "quit"):
#             break
#         reply = agent.chat(text)
#         print(f"Assistant: {reply}\n")





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

# # SECURITY WARNING: Remove this hardcoded token and use environment variable
# # os.environ["HUGGINGFACE_TOKEN"] = "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"

# # ---------------- Configuration ----------------
# class ModelConfig:
#     """Configuration class for model parameters"""
#     def __init__(self):
#         self.model_name       = "tiiuae/falcon-rw-1b"
#         self.max_length       = 512
#         self.max_new_tokens   = 64  # Increased for better responses
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
#             'Neutral': 'medium posture'
#         }

#     def extract_response_time(self, text: str) -> str:
#         tl = text.lower()
#         if any(ind in tl for ind in self.quick_indicators):
#             return "fast"
#         if any(ind in tl for ind in self.slow_indicators):
#             return "slow"
#         return "medium"

#     def extract_body_language(self, emotion: str) -> str:
#         return self.emotion_body_map.get(emotion, "medium")

#     def extract_speech_attributes(self, text: str) -> str:
#         if text.isupper():
#             return "loud"
#         if text.count('!') > 1:
#             return "excited"
#         if '...' in text or text.count('?') > 1:
#             return "hesitant"
#         return "clear"

# # ---------------- Perception Module ----------------
# class FalconEmotionClassifier:
#     """Emotion classification using Falcon-RW-1B model"""

#     def __init__(self, config: ModelConfig, hf_token: Optional[str] = None):
#         self.config = config
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
#         # Initialize tokenizer and model
#         try:
#             if token:
#                 login(token=token)
#                 logger.info("Authenticated with HuggingFace")
            
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 config.model_name, 
#                 trust_remote_code=True,
#                 padding_side="left"
#             )
            
#             if self.tokenizer.pad_token is None:
#                 self.tokenizer.pad_token = self.tokenizer.eos_token
#                 self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

#             self.model = AutoModelForCausalLM.from_pretrained(
#                 config.model_name,
#                 device_map="auto",
#                 trust_remote_code=True,
#                 torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
#                 low_cpu_mem_usage=True
#             )
#             self.model.eval()
#             logger.info(f"Falcon model loaded on {self.device}")
            
#         except Exception as e:
#             logger.error(f"Failed to load model: {e}")
#             raise

#     def _create_prompt(self, text, rt, bl, sa) -> str:
#         emo_list = ", ".join(self.config.emotions)
#         return (
#             f"Text: '{text}'\n"
#             f"Context: {rt} response, {bl} body language, {sa} speech\n"
#             f"Classify emotion from: {emo_list}\n"
#             f"Emotion:"
#         )

#     def classify_single(self, text, rt, bl, sa) -> str:
#         try:
#             prompt = self._create_prompt(text, rt, bl, sa)
#             inputs = self.tokenizer(
#                 prompt, 
#                 return_tensors="pt", 
#                 truncation=True,
#                 max_length=self.config.max_length,
#                 padding=True
#             ).to(self.device)
            
#             with torch.inference_mode():
#                 out = self.model.generate(
#                     input_ids=inputs["input_ids"],
#                     attention_mask=inputs["attention_mask"],
#                     max_new_tokens=5,  # Short output for emotion classification
#                     do_sample=False,
#                     pad_token_id=self.tokenizer.pad_token_id,
#                     eos_token_id=self.tokenizer.eos_token_id,
#                 )
            
#             resp = self.tokenizer.decode(
#                 out[0][inputs['input_ids'].shape[-1]:], 
#                 skip_special_tokens=True
#             ).strip()
            
#             # Extract emotion from response
#             pattern = r"\b(" + "|".join(self.config.emotions) + r")\b"
#             m = re.search(pattern, resp, re.IGNORECASE)
#             result = m.group(1).capitalize() if m else "Neutral"
#             logger.info(f"Classified emotion: {result}")
#             return result
            
#         except Exception as e:
#             logger.error(f"Emotion classification error: {e}")
#             return "Neutral"

# # ---------------- Inference Module ----------------
# class FalconInferenceAgent:
#     """Big Five personality trait inference using Falcon-RW-1B model"""

#     def __init__(self, config: ModelConfig, hf_token: Optional[str] = None):
#         self.config = config
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
#         # Share the same model instance to save memory
#         self.tokenizer = None
#         self.model = None

#     def _build_prompt(self, text, rt, bl, sa, emo) -> str:
#         return (
#             f"Text: '{text}'\n"
#             f"Emotion: {emo}\n"
#             f"Context: {rt}, {bl}, {sa}\n"
#             f"Rate Big Five traits (0.0-1.0):\n"
#             f"Openness: 0."
#         )

#     def _heuristic_scoring(self, text, emo, rt, bl, sa) -> Dict[str, float]:
#         """Enhanced heuristic scoring based on text analysis"""
#         tl = text.lower()
#         scores = {t: 0.5 for t in self.config.big_five_traits}

#         # Emotion-based adjustments
#         emotion_adjustments = {
#             'Happy': {'extraversion': 0.7, 'neuroticism': 0.3, 'agreeableness': 0.6},
#             'Excited': {'extraversion': 0.8, 'openness': 0.7, 'neuroticism': 0.3},
#             'Sad': {'neuroticism': 0.7, 'extraversion': 0.3, 'agreeableness': 0.6},
#             'Anxious': {'neuroticism': 0.8, 'extraversion': 0.2, 'conscientiousness': 0.4},
#             'Angry': {'neuroticism': 0.8, 'agreeableness': 0.2, 'extraversion': 0.6},
#             'Calm': {'neuroticism': 0.2, 'conscientiousness': 0.7, 'agreeableness': 0.7},
#             'Confused': {'openness': 0.6, 'neuroticism': 0.6, 'conscientiousness': 0.4}
#         }
        
#         if emo in emotion_adjustments:
#             for trait, value in emotion_adjustments[emo].items():
#                 scores[trait] = value

#         # Text content analysis
#         creative_words = ['creative', 'art', 'music', 'imagine', 'dream', 'new', 'different']
#         organized_words = ['plan', 'organize', 'schedule', 'systematic', 'careful', 'detail']
#         social_words = ['people', 'friends', 'party', 'social', 'talk', 'meet']
        
#         if any(w in tl for w in creative_words):
#             scores['openness'] = min(0.9, scores['openness'] + 0.2)
#         if any(w in tl for w in organized_words):
#             scores['conscientiousness'] = min(0.9, scores['conscientiousness'] + 0.2)
#         if any(w in tl for w in social_words):
#             scores['extraversion'] = min(0.9, scores['extraversion'] + 0.2)

#         # Response time adjustments
#         if rt == 'fast':
#             scores['extraversion'] = min(0.9, scores['extraversion'] + 0.1)
#         elif rt == 'slow':
#             scores['conscientiousness'] = min(0.8, scores['conscientiousness'] + 0.1)

#         return scores

#     def infer_traits(self, text, rt, bl, sa, emo) -> Dict[str, float]:
#         """Infer personality traits using heuristic approach"""
#         try:
#             return self._heuristic_scoring(text, emo, rt, bl, sa)
#         except Exception as e:
#             logger.error(f"Trait inference error: {e}")
#             return {t: 0.5 for t in self.config.big_five_traits}

# # ---------------- Retrieval Module ----------------
# class Retriever:
#     """Retrieval-augmented memory for similar interactions"""

#     def __init__(self, original_data_path: str, original_emb_path: str, 
#                  new_data_path: str = 'falcon_aligned_traits.json', 
#                  new_emb_path: str = 'falcon_embedding.npy'):
#         self.original_data_path = Path(original_data_path)
#         self.original_emb_path  = Path(original_emb_path)
#         self.new_data_path = Path(new_data_path)
#         self.new_emb_path  = Path(new_emb_path)
        
#         self.original_data = []
#         self.new_data      = []
#         self.original_emb  = np.empty((0,0))
#         self.new_emb       = np.empty((0,0))
#         self.vectorizer    = None
#         self.encoder       = None

#         self._load_original_data()
#         self._load_new_data()

#     def _load_original_data(self):
#         """Load original data for retrieval context only"""
#         if self.original_data_path.exists() and self.original_emb_path.exists():
#             try:
#                 self.original_data = json.loads(self.original_data_path.read_text(encoding='utf-8'))
#                 self.original_emb  = np.load(self.original_emb_path)
#                 logger.info(f"Loaded {len(self.original_data)} original interactions for retrieval")
#             except Exception as e:
#                 logger.warning(f"Could not load original data: {e}")
#                 self.original_data = []
#                 self.original_emb  = np.empty((0,0))
#         else:
#             logger.info("No original data found - starting fresh")
#             self.original_data = []
#             self.original_emb  = np.empty((0,0))

#     def _load_new_data(self):
#         """Load new interaction data"""
#         if self.new_data_path.exists():
#             try:
#                 self.new_data = json.loads(self.new_data_path.read_text(encoding='utf-8'))
#                 if self.new_emb_path.exists():
#                     self.new_emb = np.load(self.new_emb_path)
#                 logger.info(f"Loaded {len(self.new_data)} new interactions")
#             except Exception as e:
#                 logger.warning(f"Could not load new data: {e}")
#                 self._reset_new_data()
#         else:
#             self._reset_new_data()

#         self._initialize_vectorizer_encoder()

#     def _reset_new_data(self):
#         self.new_data = []
#         self.new_emb  = np.empty((0,0))

#     def _initialize_vectorizer_encoder(self):
#         """Initialize vectorizer and encoder with available data"""
#         all_data = self.original_data + self.new_data
#         if all_data:
#             try:
#                 texts = [e.get('transcript', '') for e in all_data if e.get('transcript')]
#                 emotions = [[e.get('emotion', 'Neutral')] for e in all_data]
                
#                 if texts:
#                     self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english').fit(texts)
#                     self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(emotions)
#                     logger.info("Vectorizer and encoder initialized")
#                 else:
#                     logger.warning("No valid texts found for vectorizer")
#             except Exception as e:
#                 logger.error(f"Error initializing vectorizer: {e}")
#                 self.vectorizer = None
#                 self.encoder = None

#     def embed_query(self, text: str, emotion: str, traits: List[float]) -> np.ndarray:
#         """Create embedding for query"""
#         try:
#             if self.vectorizer and text:
#                 te = self.vectorizer.transform([text]).toarray()
#             else:
#                 te = np.zeros((1, 1000))
            
#             if self.encoder:
#                 ee = self.encoder.transform([[emotion]])
#             else:
#                 ee = np.zeros((1, len(ModelConfig().emotions)))
            
#             tr = np.array([traits])
#             emb = np.hstack([te, ee, tr])[0]
#             return emb
#         except Exception as e:
#             logger.error(f"Error creating embedding: {e}")
#             return np.zeros(1000 + len(ModelConfig().emotions) + len(traits))

#     def get_top_k(self, query_emb: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
#         """Retrieve top k similar interactions"""
#         combined_data = self.original_data + self.new_data
        
#         if not combined_data:
#             return []
        
#         # Simple similarity based on text matching for now
#         # In a real implementation, you'd use proper embeddings
#         return combined_data[:min(k, len(combined_data))]

#     def add_interaction(self, text, emotion, traits, response):
#         """Add new interaction"""
#         self.new_data.append({
#             "transcript": text,
#             "emotion": emotion,
#             "traits": traits,
#             "response": response,
#             "timestamp": time.time()
#         })
#         self.save_new_data()

#     def save_new_data(self):
#         """Save new interaction data"""
#         try:
#             self.new_data_path.write_text(json.dumps(self.new_data, indent=2), encoding='utf-8')
#             logger.info(f"Saved {len(self.new_data)} new interactions")
#         except Exception as e:
#             logger.error(f"Error saving data: {e}")

# # ---------------- Dialogue Module & Main Loop ----------------
# class FalconDialogueAgent:
#     def __init__(self, original_data_path='aligned_data_with_traits.json', 
#                  original_emb_path='full_embeddings.npy',
#                  new_data_path='falcon_aligned_traits.json',
#                  new_emb_path='falcon_embedding.npy',
#                  hf_token: Optional[str]=None, auto_metadata: bool=True):
        
#         self.config = ModelConfig()
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.auto_metadata = auto_metadata
#         self.metadata_extractor = MetadataExtractor()
        
#         # Initialize components
#         self.perceptor = FalconEmotionClassifier(self.config, hf_token)
#         self.inferencer = FalconInferenceAgent(self.config, hf_token)
#         self.retriever = Retriever(original_data_path, original_emb_path, 
#                                  new_data_path, new_emb_path)
        
#         # Use the same model for dialogue generation
#         self.tokenizer = self.perceptor.tokenizer
#         self.model = self.perceptor.model
        
#         logger.info("Falcon Dialogue Agent initialized")

#     def _generate_response(self, user_input: str, emotion: str, traits: Dict[str, float], 
#                           context: str = "") -> str:
#         """Generate contextual response using Falcon model"""
#         try:
#             # Create a more natural prompt for dialogue
#             trait_summary = f"openness: {traits['openness']:.1f}, extraversion: {traits['extraversion']:.1f}"
            
#             prompt = (
#                 f"User is feeling {emotion.lower()} and shows {trait_summary}.\n"
#                 f"{context}"
#                 f"User: {user_input}\n"
#                 f"Assistant: I understand you're feeling {emotion.lower()}."
#             )
            
#             inputs = self.tokenizer(
#                 prompt,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=400,  # Shorter for better generation
#                 padding=True
#             ).to(self.device)
            
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     input_ids=inputs["input_ids"],
#                     attention_mask=inputs["attention_mask"],
#                     max_new_tokens=self.config.max_new_tokens,
#                     do_sample=True,
#                     temperature=self.config.temperature,
#                     top_p=self.config.top_p,
#                     pad_token_id=self.tokenizer.pad_token_id,
#                     eos_token_id=self.tokenizer.eos_token_id,
#                 )
            
#             response = self.tokenizer.decode(
#                 outputs[0][inputs['input_ids'].shape[-1]:],
#                 skip_special_tokens=True
#             ).strip()
            
#             # Clean up response
#             response = response.split('\n')[0].strip()  # Take first line
#             if not response:
#                 response = self._fallback_response(emotion, user_input)
            
#             logger.info(f"Generated response: {response}")
#             return response
            
#         except Exception as e:
#             logger.error(f"Response generation error: {e}")
#             return self._fallback_response(emotion, user_input)

#     def _fallback_response(self, emotion: str, user_input: str) -> str:
#         """Provide fallback responses when model generation fails"""
#         responses = {
#             'Happy': "That's wonderful to hear! I'm glad you're feeling positive.",
#             'Sad': "I'm sorry you're feeling down. Would you like to talk about it?",
#             'Angry': "I can sense you're frustrated. Let's work through this together.",
#             'Anxious': "It sounds like you're worried about something. Take a deep breath.",
#             'Excited': "Your enthusiasm is contagious! Tell me more about what's exciting you.",
#             'Calm': "It's nice that you're feeling peaceful. How can I help you today?",
#             'Confused': "I understand you're feeling uncertain. Let me try to help clarify things.",
#             'Neutral': "I hear you. How can I assist you today?"
#         }
        
#         base_response = responses.get(emotion, "I understand how you're feeling.")
#         if len(user_input.split()) < 5:
#             return base_response
#         else:
#             return f"{base_response} Thank you for sharing that with me."

#     def chat(self, user_input: str, response_time: Optional[str]=None,
#              body_language: Optional[str]=None, speech_attributes: Optional[str]=None,
#              top_k: int=3) -> Dict[str,Any]:
        
#         start_time = time.time()
        
#         try:
#             # Extract metadata
#             if self.auto_metadata:
#                 response_time = response_time or self.metadata_extractor.extract_response_time(user_input)
#                 speech_attributes = speech_attributes or self.metadata_extractor.extract_speech_attributes(user_input)
#             else:
#                 response_time = response_time or "medium"
#                 speech_attributes = speech_attributes or "clear"

#             # Classify emotion
#             emotion = self.perceptor.classify_single(
#                 user_input, response_time, body_language or "medium", speech_attributes
#             )
            
#             # Extract body language
#             if self.auto_metadata and not body_language:
#                 body_language = self.metadata_extractor.extract_body_language(emotion)
#             elif not body_language:
#                 body_language = "medium"

#             # Infer personality traits
#             traits = self.inferencer.infer_traits(
#                 user_input, response_time, body_language, speech_attributes, emotion
#             )

#             # Retrieve similar examples
#             q_emb = self.retriever.embed_query(user_input, emotion, list(traits.values()))
#             examples = self.retriever.get_top_k(q_emb, k=top_k)
            
#             print("DEBUG: retrieved examples =", examples)

#             # Build context
#             context = ""
#             if examples:
#                 context = "Previous similar interactions suggest: "
#                 context += f"Users with similar emotions often appreciate empathetic responses.\n"

#             # Generate response
#             response = self._generate_response(user_input, emotion, traits, context)

#             # Store interaction
#             self.retriever.add_interaction(user_input, emotion, traits, response)
            
#             processing_time = round(time.time() - start_time, 2)

#             return {
#                 'response': response,
#                 'emotion': emotion,
#                 'traits': traits,
#                 'metadata': {
#                     'response_time': response_time,
#                     'body_language': body_language,
#                     'speech_attributes': speech_attributes,
#                     'auto_metadata': self.auto_metadata,
#                     'processing_time': processing_time
#                 },
#                 'similar_examples': len(examples)
#             }
            
#         except Exception as e:
#             logger.error(f"Chat error: {e}")
#             return {
#                 'response': "I apologize, but I encountered an error. Please try again.",
#                 'emotion': "Neutral",
#                 'traits': {t: 0.5 for t in self.config.big_five_traits},
#                 'metadata': {
#                     'response_time': 'medium',
#                     'body_language': 'medium',
#                     'speech_attributes': 'clear',
#                     'auto_metadata': self.auto_metadata,
#                     'processing_time': round(time.time() - start_time, 2)
#                 },
#                 'similar_examples': 0
#             }

#     def save_session(self):
#         """Save session data"""
#         self.retriever.save_new_data()
#         logger.info("Session saved")

# def main():
#     """Main interactive loop"""
#     print("🦅 Falcon Dialogue System Ready!")
#     print("📁 New interactions will be saved to 'falcon_aligned_traits.json'")
#     print("⚠️  Make sure to set your HUGGINGFACE_TOKEN environment variable")
#     print("Type 'exit' to quit.\n")
    
#     try:
#         agent = FalconDialogueAgent(auto_metadata=True)
        
#         while True:
#             try:
#                 user_input = input("You: ").strip()
#                 if user_input.lower() in ('exit', 'quit'):
#                     agent.save_session()
#                     print("Goodbye!")
#                     break
                
#                 if not user_input:
#                     continue
                
#                 # Optional metadata inputs
#                 response_time = input("Response time (fast/medium/slow, Enter to auto): ").strip() or None
#                 body_language = input("Body language (Enter to auto): ").strip() or None
#                 speech_attrs = input("Speech attributes (Enter to auto): ").strip() or None
                
#                 print("Processing...")
                
#                 result = agent.chat(user_input, response_time, body_language, speech_attrs)
                
#                 print(f"\nAssistant: {result['response']}")
#                 print(f"📊 Emotion: {result['emotion']}")
#                 print(f"🧠 Traits: {', '.join([f'{k}: {v:.2f}' for k, v in result['traits'].items()])}")
#                 print(f"⏱️  Processing time: {result['metadata']['processing_time']}s")
#                 print(f"📚 Similar examples used: {result['similar_examples']}\n")
                
#             except KeyboardInterrupt:
#                 print("\nExiting...")
#                 agent.save_session()
#                 break
#             except Exception as e:
#                 print(f"Error: {e}")
#                 logger.error(f"Main loop error: {e}")
                
#     except Exception as e:
#         print(f"Failed to initialize: {e}")
#         logger.error(f"Initialization error: {e}")

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
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- Configuration ----------------
class ModelConfig:
    """Configuration class for model parameters"""
    def __init__(self):
        # Consider upgrading to a larger model like:
        # self.model_name = "microsoft/DialoGPT-medium"  # Better for dialogue
        # self.model_name = "tiiuae/falcon-7b-instruct"  # Larger Falcon model
        self.model_name = "tiiuae/falcon-rw-1b"  # Keeping your original choice
        
        self.max_length = 256  # Reduced for better performance
        self.max_new_tokens = 32  # Reduced for more focused responses
        self.temperature = 0.8
        self.top_p = 0.9
        self.emotions = [
            "Happy", "Sad", "Angry", "Anxious", "Surprised",
            "Disgusted", "Confused", "Calm", "Excited",
            "Embarrassed", "Guilty", "Neutral"
        ]
        self.big_five_traits = [
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
            'Neutral': 'medium posture'
        }

    def extract_response_time(self, text: str) -> str:
        tl = text.lower()
        if any(ind in tl for ind in self.quick_indicators):
            return "fast"
        if any(ind in tl for ind in self.slow_indicators):
            return "slow"
        return "medium"

    def extract_body_language(self, emotion: str) -> str:
        return self.emotion_body_map.get(emotion, "medium")

    def extract_speech_attributes(self, text: str) -> str:
        if text.isupper():
            return "loud"
        if text.count('!') > 1:
            return "excited"
        if '...' in text or text.count('?') > 1:
            return "hesitant"
        return "clear"

# ---------------- Fixed Emotion Classification ----------------
class FalconEmotionClassifier:
    """Improved emotion classification using Falcon model"""

    def __init__(self, config: ModelConfig, hf_token: Optional[str] = None):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Emotion keywords for rule-based fallback
        self.emotion_keywords = {
            'Happy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'cheerful', 'glad'],
            'Sad': ['sad', 'down', 'low', 'depressed', 'upset', 'disappointed', 'blue', 'miserable', 'unhappy'],
            'Angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated', 'rage', 'pissed'],
            'Anxious': ['anxious', 'worried', 'nervous', 'stressed', 'concerned', 'afraid', 'scared', 'uneasy'],
            'Surprised': ['surprised', 'shocked', 'amazed', 'unexpected', 'wow', 'astonished', 'stunned'],
            'Disgusted': ['disgusted', 'gross', 'sick', 'revolted', 'repulsed'],
            'Confused': ['confused', 'puzzled', 'unclear', 'lost', "don't understand", 'bewildered', 'perplexed'],
            'Calm': ['calm', 'peaceful', 'relaxed', 'serene', 'tranquil', 'composed', 'steady'],
            'Excited': ['thrilled', 'pumped', 'enthusiastic', 'eager', 'hyped'],
            'Embarrassed': ['embarrassed', 'ashamed', 'humiliated', 'mortified'],
            'Guilty': ['guilty', 'remorseful', 'regretful', 'sorry']
        }
        
        try:
            if token:
                login(token=token)
                logger.info("Authenticated with HuggingFace")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name, 
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            logger.info(f"Falcon model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None

    def _create_few_shot_prompt(self, text: str) -> str:
        """Create a few-shot prompt for better emotion classification"""
        return f"""Classify the emotion in these examples:

Text: "I'm so happy today! Everything is going great!"
Emotion: Happy

Text: "I feel really down and don't know why"
Emotion: Sad

Text: "This is so frustrating! Nothing works!"
Emotion: Angry

Text: "I'm worried about the exam tomorrow"
Emotion: Anxious

Text: "Wow, I didn't expect that!"
Emotion: Surprised

Text: "{text}"
Emotion:"""

    def _rule_based_classification(self, text: str) -> str:
        """Enhanced rule-based emotion classification"""
        text_lower = text.lower()
        
        # Score each emotion based on keyword matches
        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Return emotion with highest score, or Neutral if no matches
        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        
        return "Neutral"

    def _model_based_classification(self, text: str) -> Optional[str]:
        """Model-based emotion classification with better error handling"""
        if not self.model or not self.tokenizer:
            return None
            
        try:
            prompt = self._create_few_shot_prompt(text)
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=300,
                padding=True
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Clean and extract emotion
            response = re.sub(r'[^\w\s]', '', response).strip()
            words = response.split()
            
            # Look for valid emotions in response
            for word in words:
                for emotion in self.config.emotions:
                    if word.lower() == emotion.lower():
                        return emotion
            
            return None
            
        except Exception as e:
            logger.error(f"Model classification error: {e}")
            return None

    def classify_single(self, text: str, rt: str, bl: str, sa: str) -> str:
        """Classify emotion - try model first, then rule-based fallback"""
        # Try model-based classification first
        if self.model and self.tokenizer:
            emotion = self._model_based_classification(text)
            if emotion:
                logger.info(f"Model classified emotion: {emotion}")
                return emotion
        
        # Fallback to rule-based classification
        emotion = self._rule_based_classification(text)
        logger.info(f"Rule-based classified emotion: {emotion}")
        return emotion

    # Alias for compatibility
    infer_emotion = classify_single

# ---------------- Inference Module ----------------
class FalconInferenceAgent:
    """Big Five personality trait inference using Falcon-RW-1B model"""

    def __init__(self, config: ModelConfig, hf_token: Optional[str] = None):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Share the same model instance to save memory
        self.tokenizer = None
        self.model = None

    def _build_prompt(self, text, rt, bl, sa, emo) -> str:
        return (
            f"Text: '{text}'\n"
            f"Emotion: {emo}\n"
            f"Context: {rt}, {bl}, {sa}\n"
            f"Rate Big Five traits (0.0-1.0):\n"
            f"Openness: 0."
        )

    def _heuristic_scoring(self, text, emo, rt, bl, sa) -> Dict[str, float]:
        """Enhanced heuristic scoring based on text analysis"""
        tl = text.lower()
        scores = {t: 0.5 for t in self.config.big_five_traits}

        # Emotion-based adjustments
        emotion_adjustments = {
            'Happy': {'extraversion': 0.7, 'neuroticism': 0.3, 'agreeableness': 0.6},
            'Excited': {'extraversion': 0.8, 'openness': 0.7, 'neuroticism': 0.3},
            'Sad': {'neuroticism': 0.7, 'extraversion': 0.3, 'agreeableness': 0.6},
            'Anxious': {'neuroticism': 0.8, 'extraversion': 0.2, 'conscientiousness': 0.4},
            'Angry': {'neuroticism': 0.8, 'agreeableness': 0.2, 'extraversion': 0.6},
            'Calm': {'neuroticism': 0.2, 'conscientiousness': 0.7, 'agreeableness': 0.7},
            'Confused': {'openness': 0.6, 'neuroticism': 0.6, 'conscientiousness': 0.4}
        }
        
        if emo in emotion_adjustments:
            for trait, value in emotion_adjustments[emo].items():
                scores[trait] = value

        # Text content analysis
        creative_words = ['creative', 'art', 'music', 'imagine', 'dream', 'new', 'different']
        organized_words = ['plan', 'organize', 'schedule', 'systematic', 'careful', 'detail']
        social_words = ['people', 'friends', 'party', 'social', 'talk', 'meet']
        
        if any(w in tl for w in creative_words):
            scores['openness'] = min(0.9, scores['openness'] + 0.2)
        if any(w in tl for w in organized_words):
            scores['conscientiousness'] = min(0.9, scores['conscientiousness'] + 0.2)
        if any(w in tl for w in social_words):
            scores['extraversion'] = min(0.9, scores['extraversion'] + 0.2)

        # Response time adjustments
        if rt == 'fast':
            scores['extraversion'] = min(0.9, scores['extraversion'] + 0.1)
        elif rt == 'slow':
            scores['conscientiousness'] = min(0.8, scores['conscientiousness'] + 0.1)

        return scores

    def infer_traits(self, text, rt, bl, sa, emo) -> Dict[str, float]:
        """Infer personality traits using heuristic approach"""
        try:
            return self._heuristic_scoring(text, emo, rt, bl, sa)
        except Exception as e:
            logger.error(f"Trait inference error: {e}")
            return {t: 0.5 for t in self.config.big_five_traits}

# ---------------- Fixed Retrieval Module ----------------
class Retriever:
    """Fixed retrieval-augmented memory with proper similarity search"""

    def __init__(self, original_data_path: str, original_emb_path: str, 
                 new_data_path: str = 'falcon_aligned_traits.json', 
                 new_emb_path: str = 'falcon_embedding.npy'):
        self.original_data_path = Path(original_data_path)
        self.original_emb_path = Path(original_emb_path)
        self.new_data_path = Path(new_data_path)
        self.new_emb_path = Path(new_emb_path)
        
        self.original_data = []
        self.new_data = []
        self.all_texts = []
        self.vectorizer = None
        self.text_embeddings = None

        self._load_original_data()
        self._load_new_data()
        self._build_text_index()

    def _load_original_data(self):
        """Load original data"""
        if self.original_data_path.exists():
            try:
                self.original_data = json.loads(self.original_data_path.read_text(encoding='utf-8'))
                logger.info(f"Loaded {len(self.original_data)} original interactions")
            except Exception as e:
                logger.warning(f"Could not load original data: {e}")
                self.original_data = []
        else:
            logger.info("No original data found")
            self.original_data = []

    def _load_new_data(self):
        """Load new interaction data"""
        if self.new_data_path.exists():
            try:
                self.new_data = json.loads(self.new_data_path.read_text(encoding='utf-8'))
                logger.info(f"Loaded {len(self.new_data)} new interactions")
            except Exception as e:
                logger.warning(f"Could not load new data: {e}")
                self.new_data = []
        else:
            self.new_data = []

    def _build_text_index(self):
        """Build proper text similarity index"""
        all_data = self.original_data + self.new_data
        if not all_data:
            return
            
        try:
            self.all_texts = []
            for item in all_data:
                text = item.get('transcript', '')
                if text:
                    self.all_texts.append(text)
            
            if self.all_texts:
                self.vectorizer = TfidfVectorizer(
                    max_features=500, 
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.text_embeddings = self.vectorizer.fit_transform(self.all_texts)
                logger.info(f"Built text index with {len(self.all_texts)} documents")
        except Exception as e:
            logger.error(f"Error building text index: {e}")

    def get_top_k(self, query_text: str, emotion: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top k similar interactions based on text similarity"""
        all_data = self.original_data + self.new_data
        
        if not all_data or not self.vectorizer or not query_text.strip():
            return []
        
        try:
            # Get text similarity
            query_vec = self.vectorizer.transform([query_text])
            similarities = cosine_similarity(query_vec, self.text_embeddings).flatten()
            
            # Get top k indices
            top_indices = similarities.argsort()[-k:][::-1]
            
            # Filter out low similarity results
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    if idx < len(all_data):
                        item = all_data[idx].copy()
                        item['similarity'] = float(similarities[idx])
                        results.append(item)
            
            logger.info(f"Retrieved {len(results)} similar examples")
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []

    def add_interaction(self, text: str, emotion: str, traits: Dict[str, float], response: str, processing_time: float):
        """Add new interaction and rebuild index"""
        self.new_data.append({
            "transcript": text,
            "emotion": emotion,
            "traits": traits,
            "response": response,
            "timestamp": time.time(),
            "processing_time": processing_time,
        })
        
        # Rebuild index with new data
        self._build_text_index()
        self.save_new_data()

    def save_new_data(self):
        """Save new interaction data"""
        try:
            self.new_data_path.write_text(json.dumps(self.new_data, indent=2), encoding='utf-8')
            logger.info(f"Saved {len(self.new_data)} new interactions")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

# ---------------- Improved Response Generation ----------------
class FalconDialogueAgent:
    def __init__(self, original_data_path='aligned_data_with_traits.json', 
                 original_emb_path='full_embeddings.npy',
                 new_data_path='falcon_aligned_traits.json',
                 new_emb_path='falcon_embedding.npy',
                 hf_token: Optional[str] = None, auto_metadata: bool = True):
        
        # Initialize config first
        self.config = ModelConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.auto_metadata = auto_metadata
        
        # Initialize metadata extractor
        self.metadata_extractor = MetadataExtractor()
        
        # Initialize components with proper config
        self.perceptor = FalconEmotionClassifier(self.config, hf_token)
        self.inferencer = FalconInferenceAgent(self.config, hf_token)
        self.retriever = Retriever(original_data_path, original_emb_path, 
                                 new_data_path, new_emb_path)
        
        # Use the same model for dialogue generation
        self.tokenizer = self.perceptor.tokenizer
        self.model = self.perceptor.model
        
        logger.info("Improved Falcon Dialogue Agent initialized")

    def _create_empathetic_response(self, user_input: str, emotion: str, 
                                  traits: Dict[str, float], similar_examples: List[Dict]) -> str:
        """Create contextually appropriate empathetic response"""
        
        # Base empathetic responses by emotion
        empathetic_starters = {
            'Happy': ["I'm so glad to hear you're feeling good!", "That's wonderful!", "Your happiness is contagious!"],
            'Sad': ["I'm sorry you're feeling down.", "I can hear that you're struggling right now.", "It sounds like you're going through a tough time."],
            'Angry': ["I can sense your frustration.", "It sounds like something really upset you.", "I understand you're feeling angry right now."],
            'Anxious': ["I can hear the worry in your words.", "It sounds like you're feeling anxious.", "I understand you're concerned about something."],
            'Confused': ["I can see you're feeling uncertain.", "It sounds like you're trying to figure things out.", "I understand the confusion you're experiencing."],
            'Neutral': ["I hear you.", "Thank you for sharing that with me.", "I'm here to listen."]
        }
        
        starter = empathetic_starters.get(emotion, empathetic_starters['Neutral'])[0]
        
        # Add specific responses based on user input content
        user_lower = user_input.lower()
        
        if 'help' in user_lower:
            return f"{starter} I'm here to help you work through this. What would be most helpful right now?"
        elif 'why' in user_lower and ('feel' in user_lower or 'feeling' in user_lower):
            return f"{starter} Sometimes our feelings can be hard to understand. Would you like to talk about what might be contributing to how you're feeling?"
        elif 'don\'t know' in user_lower or 'not sure' in user_lower:
            return f"{starter} It's okay not to have all the answers. Sometimes talking through things can help bring clarity."
        
        # Default empathetic response
        follow_ups = {
            'Happy': "What's been going well for you?",
            'Sad': "Would you like to talk about what's been weighing on you?",
            'Angry': "What happened that's got you feeling this way?",
            'Anxious': "What's been on your mind lately?",
            'Confused': "What's been puzzling you?",
            'Neutral': "How can I support you today?"
        }
        
        follow_up = follow_ups.get(emotion, "How are you doing?")
        return f"{starter} {follow_up}"

    def _generate_response(self, user_input: str, emotion: str, traits: Dict[str, float], 
                          similar_examples: List[Dict]) -> str:
        """Generate improved contextual response"""
        try:
            # For small models, use template-based responses for better quality
            return self._create_empathetic_response(user_input, emotion, traits, similar_examples)
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._create_empathetic_response(user_input, emotion, traits, [])

    def chat(self, user_input: str, response_time: Optional[str] = None,
             body_language: Optional[str] = None, speech_attributes: Optional[str] = None,
             top_k: int = 3) -> Dict[str, Any]:
        
        start_time = time.time()
        
        try:
            # Extract metadata
            if self.auto_metadata:
                response_time = response_time or self.metadata_extractor.extract_response_time(user_input)
                speech_attributes = speech_attributes or self.metadata_extractor.extract_speech_attributes(user_input)
            else:
                response_time = response_time or "medium"
                speech_attributes = speech_attributes or "clear"

            # Classify emotion
            emotion = self.perceptor.classify_single(
                user_input, response_time, body_language or "medium", speech_attributes
            )
            
            # Extract body language
            if self.auto_metadata and not body_language:
                body_language = self.metadata_extractor.extract_body_language(emotion)
            elif not body_language:
                body_language = "medium"

            # Infer personality traits
            traits = self.inferencer.infer_traits(
                user_input, response_time, body_language, speech_attributes, emotion
            )

            # Retrieve similar examples (now properly implemented)
            similar_examples = self.retriever.get_top_k(user_input, emotion, k=top_k)

            # Generate response
            response = self._generate_response(user_input, emotion, traits, similar_examples)

            processing_time = round(time.time() - start_time, 2)
            
            # Store interaction
            self.retriever.add_interaction(user_input, emotion, traits, response, processing_time)

            return {
                'response': response,
                'emotion': emotion,
                'traits': traits,
                'metadata': {
                    'response_time': response_time,
                    'body_language': body_language,  
                    'speech_attributes': speech_attributes,
                    'auto_metadata': self.auto_metadata,
                    'processing_time': processing_time
                },
                'similar_examples': len(similar_examples),
                'retrieved_examples': similar_examples[:2] if similar_examples else []  # Show first 2 for debugging
            }
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {
                'response': "I'm sorry you're feeling this way. I'm here to listen and help however I can.",
                'emotion': "Neutral",
                'traits': {t: 0.5 for t in self.config.big_five_traits},
                'metadata': {
                    'response_time': 'medium',
                    'body_language': 'medium',
                    'speech_attributes': 'clear',
                    'auto_metadata': self.auto_metadata,
                    'processing_time': round(time.time() - start_time, 2)
                },
                'similar_examples': 0,
                'retrieved_examples': []
            }

    def save_session(self):
        """Save session data"""
        self.retriever.save_new_data()
        logger.info("Session saved")

# def main():
#     """Main interactive loop with improved error handling"""
#     print("🦅 Improved Falcon Dialogue System Ready!")
#     print("📁 New interactions will be saved to 'falcon_aligned_traits.json'")
#     print("⚠️  Make sure to set your HUGGINGFACE_TOKEN environment variable")
#     print("💡 This version includes improved emotion classification and retrieval")
#     print("Type 'exit' to quit.\n")
    
#     try:
#         agent = FalconDialogueAgent(auto_metadata=True)
        
#         while True:
#             try:
#                 user_input = input("You: ").strip()
#                 if user_input.lower() in ('exit', 'quit'):
#                     agent.save_session()
#                     print("Goodbye!")
#                     break
                
#                 if not user_input:
#                     continue
                
#                 print("Processing...")
                
#                 result = agent.chat(user_input)
                
#                 print(f"\nAssistant: {result['response']}")
#                 print(f"📊 Emotion: {result['emotion']}")
#                 print(f"🧠 Traits: {', '.join([f'{k}: {v:.2f}' for k, v in result['traits'].items()])}")
#                 print(f"⏱️  Processing time: {result['metadata']['processing_time']}s")
#                 print(f"📚 Similar examples used: {result['similar_examples']}")
                
#                 # Show retrieved examples for debugging
#                 if result.get('retrieved_examples'):
#                     print("🔍 Retrieved examples:")
#                     for i, ex in enumerate(result['retrieved_examples']):
#                         print(f"  {i+1}. {ex.get('transcript', '')[:50]}... (similarity: {ex.get('similarity', 0):.3f})")
#                 print()
                
#             except KeyboardInterrupt:
#                 print("\nExiting...")
#                 agent.save_session()
#                 break
#             except Exception as e:
#                 print(f"Error: {e}")
#                 logger.error(f"Main loop error: {e}")
                
#     except Exception as e:
#         print(f"Failed to initialize: {e}")
#         logger.error(f"Initialization error: {e}")

# if __name__ == "__main__":
#     main()



def main():
    """Batch-run FalconDialogueAgent with predefined test queries"""
    print("🦅 Falcon Dialogue System Ready for Batch Test!")
    print("⚠️  Make sure to set your HUGGINGFACE_TOKEN environment variable")
    print("💡 Batch mode: results will be saved to 'falcon_aligned_traits.json'\n")
    
    try:
        agent = FalconDialogueAgent(auto_metadata=True)
        
        # load your test queries JSON
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
                wall_time = round(end_time - start_time, 2)
                
                print(f"✅ Done. Assistant: {result['response'][:80]}...")
                print(f"📊 Emotion: {result['emotion']}, Traits: {result['traits']}")
                print(f"⏱️  Wall time: {wall_time}s\n")
                
            except Exception as e:
                print(f"⚠️  Error on query {i}: {e}")
                logger.error(f"Batch error on query {i}: {e}")
        
        agent.save_session()
        print("🎉 Batch test complete! All results saved to 'falcon_aligned_traits.json'.")
        
    except Exception as e:
        print(f"Failed to initialize: {e}")
        logger.error(f"Initialization error: {e}")

if __name__ == "__main__":
    main()
