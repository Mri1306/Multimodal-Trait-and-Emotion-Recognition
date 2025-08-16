# import os
# import re
# import json
# import numpy as np
# import logging
# import time
# from typing import Optional, Dict, Any, List
# from pathlib import Path
# from openai import OpenAI
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics.pairwise import cosine_similarity

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # ---------------- Configuration ----------------
# # Optional configuration adjustments for better API efficiency
# class ModelConfig:
#     """Configuration class for model parameters"""
#     def __init__(self):
#         # Use the cheaper model if quota is limited
#         self.model_name       = "gpt-3.5-turbo"  # Keep this - it's cheaper than gpt-4
        
#         # Reduce token usage to save on API costs
#         self.max_tokens       = 100  # Reduced from 150 to save tokens
        
#         # Slightly lower temperature for more consistent responses
#         self.temperature      = 0.5  # Reduced from 0.7 for more predictable output
        
#         # Keep emotions the same
#         self.emotions         = [
#             "Happy", "Sad", "Angry", "Anxious", "Surprised",
#             "Disgusted", "Confused", "Calm", "Excited",
#             "Embarrassed", "Guilty", "Neutral"
#         ]
        
#         # Keep traits the same
#         self.big_five_traits  = [
#             "openness", "conscientiousness", "extraversion",
#             "agreeableness", "neuroticism"
#         ]
        
#         # NEW: Add API usage controls
#         self.max_retries      = 2    # Limit retries to avoid burning through quota
#         self.retry_delay      = 2.0  # Seconds to wait between retries
#         self.enable_fallback  = True # Always use fallback when API fails
#         self.requests_per_minute = 50
#         self.min_interval        = 2.0    # ⬅️ at least 1 second between ANY two calls
#         self.last_request_times  = []     # timestamps of calls in the past 60s
#         self._last_call_ts       = 0.0    # timestamp of the very last call

#         # NEW: Add rate limiting
#         self.requests_per_minute = 50  # Conservative rate limit
#         self.last_request_times = []   # Track request timing

# def throttle(config: ModelConfig):
#     now = time.time()

#     # 1) ENFORCE per-second/QPS limit:
#     elapsed = now - config._last_call_ts
#     if elapsed < config.min_interval:
#         to_sleep = config.min_interval - elapsed
#         logger.info(f"[throttle] sleeping {to_sleep:.2f}s for QPS limit")
#         time.sleep(to_sleep)
#         now = time.time()

#     # 2) ENFORCE per-minute limit:
#     #    drop entries older than 60s, then if we’ve already made >= RPM requests, wait
#     cutoff = now - 60
#     config.last_request_times = [t for t in config.last_request_times if t >= cutoff]
#     if len(config.last_request_times) >= config.requests_per_minute:
#         # how long until the oldest timestamp falls out of the 60s window?
#         oldest = config.last_request_times[0]
#         wait = 60 - (now - oldest)
#         logger.info(f"[throttle] sleeping {wait:.2f}s for RPM limit")
#         time.sleep(wait)
#         now = time.time()
#         # prune again
#         config.last_request_times = [t for t in config.last_request_times if t >= now - 60]

#     # record this call
#     config._last_call_ts = now
#     config.last_request_times.append(now)

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
# # 1. Enhanced OpenAI client initialization with better error handling
# class OpenAIEmotionClassifier:
#     def __init__(self, config: ModelConfig, openai_api_key: Optional[str] = None):
#         self.config = config
#         try:
#             self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
#             # Test the API key with a minimal request
#             self._test_api_connection()
#         except Exception as e:
#             logger.error(f"OpenAI client initialization failed: {e}")
#             self.client = None

#     def _test_api_connection(self):
#         """Test if API key is valid and has quota"""
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[{"role": "user", "content": "Hi"}],
#                 max_tokens=1
#             )
#             logger.info("OpenAI API connection successful")
#         except Exception as e:
#             logger.warning(f"API test failed: {e}")
#             if "insufficient_quota" in str(e) or "429" in str(e):
#                 logger.error("OpenAI quota exceeded - falling back to local processing")

#     def _create_prompt(self, text, rt, bl, sa) -> str:
#         emo_list = ", ".join(self.config.emotions)
#         return f"""Classify the emotion from the following options: {emo_list}

# Text: "{text}"
# Response time: {rt}
# Body language: {bl}
# Speech attributes: {sa}

# Based on the text and context, respond with ONLY the emotion name from the list above."""
    
#     def classify_single(self, text, rt, bl, sa) -> str:
#         # Fallback emotion classification without API
#         if not self.client:
#             return self._fallback_emotion_classification(text, rt, bl, sa)
            
#         try:
#             prompt = self._create_prompt(text, rt, bl, sa)
#             throttle(self.config)

#             response = self.client.chat.completions.create(
#                 model=self.config.model_name,
#                 messages=[
#                     {"role": "system", "content": "You are an emotion classifier. Respond with only the emotion name."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=10,
#                 temperature=0.1
#             )
            
#             result = response.choices[0].message.content.strip()
#             pattern = r"\b(" + "|".join(self.config.emotions) + r")\b"
#             match = re.search(pattern, result, re.IGNORECASE)
#             return match.group(1).capitalize() if match else "Neutral"
            
#         except Exception as e:
#             if "insufficient_quota" in str(e) or "429" in str(e):
#                 logger.warning("API quota exceeded, using fallback emotion classification")
#                 return self._fallback_emotion_classification(text, rt, bl, sa)
#             else:
#                 logger.error(f"Emotion classification error: {e}")
#                 return "Neutral"

#     def _fallback_emotion_classification(self, text, rt, bl, sa) -> str:
#         """Local emotion classification without API"""
#         text_lower = text.lower()
        
#         # Keyword-based emotion detection
#         emotion_keywords = {
#             'Happy': ['happy', 'joy', 'great', 'awesome', 'excited', 'wonderful', 'good', 'fantastic'],
#             'Sad': ['sad', 'down', 'depressed', 'blue', 'unhappy', 'upset', 'disappointed'],
#             'Angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated'],
#             'Anxious': ['anxious', 'worried', 'nervous', 'stressed', 'panic', 'afraid'],
#             'Surprised': ['surprised', 'shocked', 'amazed', 'wow', 'incredible'],
#             'Confused': ['confused', 'lost', 'unclear', 'puzzled', "don't understand"],
#             'Excited': ['excited', 'thrilled', 'pumped', 'eager', 'enthusiastic']
#         }
        
#         # Check for emotion keywords
#         for emotion, keywords in emotion_keywords.items():
#             if any(keyword in text_lower for keyword in keywords):
#                 return emotion
        
#         # Context-based classification
#         if rt == 'fast' and ('!' in text or text.isupper()):
#             return 'Excited'
#         elif rt == 'slow' and ('...' in text or '?' in text):
#             return 'Confused'
#         elif bl and 'down' in bl.lower():
#             return 'Sad'
        
#         return 'Neutral'
    
# # ---------------- Inference Module ----------------
# # 2. Enhanced OpenAIInferenceAgent with fallback
# class OpenAIInferenceAgent:
#     def __init__(self, config: ModelConfig, openai_api_key: Optional[str] = None):
#         self.config = config
#         try:
#             self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
#         except Exception as e:
#             logger.error(f"OpenAI inference client initialization failed: {e}")
#             self.client = None

#     def _build_prompt(self, text, rt, bl, sa, emo) -> str:
#         desc = {
#             'openness': 'creativity and openness to new experiences',
#             'conscientiousness': 'organization and self-discipline',
#             'extraversion': 'sociabil and assertiveness',
#             'agreeableness': 'cooperation and trust in others',
#             'neuroticism': 'emotional instability and anxiety'
#         }
#         return f"""Based on this text, provide Big Five personality scores (0.0 to 1.0) in JSON format:

# Text: "{text}"
# Emotion: {emo}
# Context: response time {rt}, body language {bl}, speech {sa}

# Personality traits to score:
# - Openness ({desc['openness']})
# - Conscientiousness ({desc['conscientiousness']})
# - Extraversion ({desc['extraversion']})
# - Agreeableness ({desc['agreeableness']})
# - Neuroticism ({desc['neuroticism']})

# Return ONLY a JSON object with the scores, like:
# {{"openness":0.7,"conscientiousness":0.6,"extraversion":0.5,"agreeableness":0.8,"neuroticism":0.3}}"""

#     def _extract_json(self, text: str) -> Optional[Dict]:
#         """Extract JSON object from response"""
#         # Try to find JSON in the response
#         json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
#         if json_match:
#             try:
#                 return json.loads(json_match.group(0))
#             except json.JSONDecodeError:
#                 pass
#         return None

#     def _heuristic_scoring(self, text, emo, rt, bl, sa) -> Dict[str, float]:
#         """Fallback heuristic scoring if API fails"""
#         tl = text.lower()
#         scores = {t: 0.5 for t in self.config.big_five_traits}

#         # Emotion-based adjustments
#         if emo in ['Happy', 'Excited']:
#             scores['extraversion'] = min(0.8, scores['extraversion'] + 0.2)
#             scores['neuroticism'] = max(0.2, scores['neuroticism'] - 0.2)
#         elif emo in ['Sad', 'Anxious']:
#             scores['neuroticism'] = min(0.8, scores['neuroticism'] + 0.2)
#             scores['extraversion'] = max(0.2, scores['extraversion'] - 0.1)
#         elif emo == 'Angry':
#             scores['neuroticism'] = min(0.9, scores['neuroticism'] + 0.3)
#             scores['agreeableness'] = max(0.1, scores['agreeableness'] - 0.3)

#         # Response-time heuristics
#         if rt == 'fast':
#             scores['extraversion'] = min(0.9, scores['extraversion'] + 0.1)
#         elif rt == 'slow':
#             scores['conscientiousness'] = min(0.8, scores['conscientiousness'] + 0.1)
#             scores['openness'] = min(0.8, scores['openness'] + 0.1)

#         # Creativity keywords
#         creative_words = ['creative', 'art', 'music', 'imagine', 'dream']
#         if any(w in tl for w in creative_words):
#             scores['openness'] = min(0.9, scores['openness'] + 0.2)

#         return scores
    
#     def infer_traits(self, text, rt, bl, sa, emo) -> Dict[str, float]:
#         # Always compute heuristic baseline
#         heuristic_scores = self._heuristic_scoring(text, emo, rt, bl, sa)
        
#         # If no API client, return heuristic scores
#         if not self.client:
#             logger.info("Using heuristic-only trait scoring (no API)")
#             return heuristic_scores
        
#         try:
#             prompt = self._build_prompt(text, rt, bl, sa, emo)
#             throttle(self.config)

#             response = self.client.chat.completions.create(
#                 model=self.config.model_name,
#                 messages=[
#                     {"role": "system", "content": "You are a personality assessment expert. Return only valid JSON."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 max_tokens=100,
#                 temperature=0.1
#             )
            
#             result = response.choices[0].message.content.strip()
#             api_scores = self._extract_json(result)
            
#             if not api_scores:
#                 logger.warning("Failed to extract JSON from API response, using heuristic scores")
#                 return heuristic_scores
            
#             # Blend heuristic (40%) with API scores (60%)
#             final_scores = {}
#             for trait in self.config.big_five_traits:
#                 h_score = heuristic_scores[trait]
#                 a_score = float(api_scores.get(trait, h_score))
#                 a_score = max(0.0, min(1.0, a_score))
#                 final_scores[trait] = 0.4 * h_score + 0.6 * a_score
            
#             return final_scores
            
#         except Exception as e:
#             if "insufficient_quota" in str(e) or "429" in str(e):
#                 logger.warning("API quota exceeded, using heuristic-only trait scoring")
#             else:
#                 logger.error(f"Trait inference error: {e}")
#             return heuristic_scores

# # ---------------- Retrieval Module ----------------
# class Retriever:
#     """Retrieval-augmented memory for similar interactions"""

#     def __init__(self, original_data_path: str, original_emb_path: str, 
#                  new_data_path: str = 'openai_aligned_traits.json', 
#                  new_emb_path: str = 'openai_embedding.npy'):
#         # Original files for existing data
#         self.original_data_path = Path(original_data_path)
#         self.original_emb_path  = Path(original_emb_path)
        
#         # New files for storing new interactions
#         self.new_data_path = Path(new_data_path)
#         self.new_emb_path  = Path(new_emb_path)
        
#         self.original_data = []
#         self.new_data      = []
#         self.original_emb  = np.empty((0,0))
#         self.new_emb       = np.empty((0,0))
#         self.vectorizer    = None
#         self.encoder       = None

#         # Load original data for retrieval context
#         self._load_original_data()
        
#         # Load new data if exists
#         self._load_new_data()

#     def _load_original_data(self):
#         """Load original data for retrieval context only"""
#         if self.original_data_path.exists() and self.original_emb_path.exists():
#             try:
#                 with open(self.original_data_path, 'r', encoding='utf-8') as f:
#                     raw_data = json.load(f)
                
#                 # Filter out entries with empty transcripts
#                 self.original_data = [
#                     item for item in raw_data 
#                     if item.get('transcript', '').strip()
#                 ]
                
#                 logger.info(f"Loaded {len(self.original_data)} original interactions with valid transcripts")
                
#                 # Load embeddings and align with filtered data
#                 all_embeddings = np.load(self.original_emb_path)
                
#                 if len(self.original_data) != len(raw_data):
#                     logger.warning(f"Filtered {len(raw_data) - len(self.original_data)} entries with empty transcripts")
#                     # Create mapping of valid indices
#                     valid_indices = [
#                         i for i, item in enumerate(raw_data)
#                         if item.get('transcript', '').strip()
#                     ]
#                     if len(valid_indices) <= all_embeddings.shape[0]:
#                         self.original_emb = all_embeddings[valid_indices]
#                         logger.info(f"Filtered embeddings to match valid data: {self.original_emb.shape}")
#                     else:
#                         logger.error("Mismatch between data and embeddings - using all embeddings")
#                         self.original_emb = all_embeddings
#                 else:
#                     self.original_emb = all_embeddings
                    
#             except Exception as e:
#                 logger.error(f"Original data load error: {e}")
#                 self.original_data = []
#                 self.original_emb  = np.empty((0,0))
#         else:
#             logger.info("No original data found")
#             self.original_data = []
#             self.original_emb  = np.empty((0,0))

#     def _load_new_data(self):
#         """Load new interaction data"""
#         if self.new_data_path.exists() and self.new_emb_path.exists():
#             try:
#                 with open(self.new_data_path, 'r', encoding='utf-8') as f:
#                     self.new_data = json.load(f)
#                 self.new_emb  = np.load(self.new_emb_path)
#                 logger.info(f"Loaded {len(self.new_data)} new interactions")
#             except Exception as e:
#                 logger.error(f"New data load error: {e}")
#                 self._reset_new_data()
#         else:
#             logger.info("No existing new data—starting fresh")
#             self._reset_new_data()

#         # Initialize vectorizer and encoder with all available data
#         self._initialize_vectorizer_encoder()

#     def _reset_new_data(self):
#         self.new_data = []
#         self.new_emb  = np.empty((0,0))

#     def _initialize_vectorizer_encoder(self):
#         """Initialize vectorizer and encoder with all available data"""
#         all_data = self.original_data + self.new_data
#         if all_data:
#             # Extract texts and emotions, ensuring we only use valid entries
#             texts = []
#             emotions = []
#             for item in all_data:
#                 text = item.get('transcript', '').strip()
#                 if text:  # Only include non-empty texts
#                     texts.append(text)
#                     emotions.append([item.get('emotion', 'Neutral')])
            
#             if texts:
#                 self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english').fit(texts)
#                 self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(emotions)
#                 logger.info(f"Vectorizer initialized with {len(texts)} valid texts")
#             else:
#                 logger.warning("No valid texts found for vectorizer initialization")
#                 self.vectorizer = None
#                 self.encoder = None
#         else:
#             logger.warning("No data available for vectorizer initialization")
#             self.vectorizer = None
#             self.encoder = None

#     def embed_query(self, text: str, emotion: str, traits: List[float]) -> np.ndarray:
#         if self.vectorizer and text.strip():
#             te = self.vectorizer.transform([text]).toarray()
#         else:
#             te = np.zeros((1, 1000))
            
#         if self.encoder:
#             ee = self.encoder.transform([[emotion]])
#         else:
#             ee = np.zeros((1, len(ModelConfig().emotions)))
            
#         tr = np.array([traits])
#         emb = np.hstack([te, ee, tr])[0]
        
#         # Get combined embeddings for dimension checking
#         combined_emb = self._get_combined_embeddings()
#         if combined_emb.size > 0:
#             D = combined_emb.shape[1]
#             if len(emb) > D: 
#                 emb = emb[:D]
#             elif len(emb) < D: 
#                 emb = np.concatenate([emb, np.zeros(D-len(emb))])
#         return emb

#     def _get_combined_embeddings(self) -> np.ndarray:
#         """Combine original and new embeddings for retrieval"""
#         if self.original_emb.size > 0 and self.new_emb.size > 0:
#             return np.vstack([self.original_emb, self.new_emb])
#         elif self.original_emb.size > 0:
#             return self.original_emb
#         elif self.new_emb.size > 0:
#             return self.new_emb
#         else:
#             return np.empty((0,0))

#     def get_top_k(self, query_emb: np.ndarray, k: int=3) -> List[Dict[str,Any]]:
#         combined_data = self.original_data + self.new_data
#         combined_emb = self._get_combined_embeddings()
        
#         if not combined_data:
#             logger.info("No past interactions—skipping retrieval")
#             return []
        
#         logger.info(f"Total available interactions: {len(combined_data)}")
        
#         if combined_emb.size == 0:
#             logger.warning("No embeddings available—skipping retrieval")
#             return []
            
#         if query_emb.shape[0] != combined_emb.shape[1]:
#             logger.warning(f"Dimension mismatch: query {query_emb.shape[0]} vs embeddings {combined_emb.shape[1]}")
#             logger.warning("Recomputing embeddings...")
#             self._update_new_embeddings()
#             combined_emb = self._get_combined_embeddings()
#             if query_emb.shape[0] != combined_emb.shape[1]:
#                 logger.error(f"Still mismatch after recomputing: {query_emb.shape[0]} vs {combined_emb.shape[1]}")
#                 return []
        
#         # Compute similarities
#         sims = cosine_similarity([query_emb], combined_emb)[0]
#         idxs = sims.argsort()[-k:][::-1]
        
#         # Filter by similarity threshold and return results
#         results = []
#         for idx in idxs:
#             if idx < len(combined_data) and sims[idx] > 0.01:
#                 item = combined_data[idx]
#                 # Ensure transcript exists and is not empty
#                 if item.get('transcript', '').strip():
#                     results.append(item)
        
#         logger.info(f"Retrieved {len(results)} examples with similarities: {[round(sims[i],3) for i in idxs[:len(results)]]}")
        
#         return results

#     def add_interaction(self, text, emotion, traits, response):
#         """Add new interaction to separate storage files"""
#         self.new_data.append({
#             "transcript": text,
#             "emotion": emotion,
#             "traits": traits,
#             "response": response,
#             "timestamp": time.time()
#         })
#         self._update_new_embeddings()
#         self.save_new_data()

#     def _update_new_embeddings(self):
#         """Update embeddings for new data only"""
#         if not self.new_data:
#             return
            
#         # Re-initialize vectorizer and encoder with all data
#         self._initialize_vectorizer_encoder()
        
#         # Create embeddings for new data
#         new_embs = []
#         for d in self.new_data:
#             transcript = d.get('transcript', '')
#             if transcript.strip():  # Only process non-empty transcripts
#                 emb = self.embed_query(transcript, d.get('emotion', 'Neutral'), list(d.get('traits', {}).values()))
#                 new_embs.append(emb)
        
#         self.new_emb = np.vstack(new_embs) if new_embs else np.empty((0,0))
#         logger.info(f"New embeddings updated: {self.new_emb.shape}")

#     def save_new_data(self):
#         """Save only the new interaction data"""
#         try:
#             with open(self.new_data_path, 'w', encoding='utf-8') as f:
#                 json.dump(self.new_data, f, indent=2, ensure_ascii=False)
#             if self.new_emb.size > 0:
#                 np.save(self.new_emb_path, self.new_emb)
#             logger.info(f"New interaction data saved to {self.new_data_path} and {self.new_emb_path}")
#         except Exception as e:
#             logger.error(f"Save error: {e}")

# # ---------------- Dialogue Module & Main Loop ----------------
# class OpenAIDialogueAgent:
#     def __init__(self, original_data_path='aligned_data_with_traits.json', 
#                  original_emb_path='full_embeddings.npy',
#                  new_data_path='openai_aligned_traits.json',
#                  new_emb_path='openai_embedding.npy',
#                  openai_api_key: Optional[str]=None, auto_metadata: bool=True):
        
#         self.config = ModelConfig()
#         self.auto_metadata = auto_metadata
#         self.metadata_extractor = MetadataExtractor()
#         self.perceptor = OpenAIEmotionClassifier(self.config, openai_api_key)
#         self.inferencer = OpenAIInferenceAgent(self.config, openai_api_key)
#         self.retriever = Retriever(original_data_path, original_emb_path, 
#                                  new_data_path, new_emb_path)
        
#         # Initialize OpenAI client with error handling
#         try:
#             self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
#         except Exception as e:
#             logger.error(f"OpenAI dialogue client initialization failed: {e}")
#             self.client = None

#     def _generate_fallback_response(self, user_input: str, emotion: str, traits: Dict[str, float], examples: List[Dict] = None) -> str:
#         """Generate response without OpenAI API"""
        
#         # Emotion-based responses
#         emotion_responses = {
#             'Sad': [
#                 "I hear that you're feeling down. It's completely normal to have these feelings sometimes.",
#                 "I'm sorry you're going through a tough time. Would you like to talk about what's bothering you?",
#                 "It sounds like you're having a difficult moment. Remember that it's okay to feel sad sometimes.",
#                 "I can sense you're not feeling your best right now. Sometimes just acknowledging our feelings can help."
#             ],
#             'Happy': [
#                 "I'm so glad to hear you're feeling good! What's bringing you joy today?",
#                 "Your positive energy is wonderful! I'd love to hear more about what's making you happy.",
#                 "It's great to see you in good spirits! Care to share what's going well?"
#             ],
#             'Anxious': [
#                 "I understand you're feeling anxious. Take a deep breath - you're not alone in this.",
#                 "Anxiety can be overwhelming. Would some calming techniques be helpful right now?",
#                 "I can sense your worry. Sometimes it helps to talk through what's making you anxious."
#             ],
#             'Angry': [
#                 "I can sense your frustration. It's important to acknowledge these feelings.",
#                 "It sounds like something has really upset you. Would you like to talk through it?",
#                 "I hear the anger in your words. It's okay to feel this way - let's work through it together."
#             ],
#             'Confused': [
#                 "I can see you're trying to work through something. Let's take it step by step.",
#                 "It's okay to feel uncertain. Would you like help sorting through your thoughts?",
#                 "Confusion is natural when we're processing complex situations. I'm here to help clarify things."
#             ],
#             'Excited': [
#                 "I love your enthusiasm! Tell me more about what's got you so excited.",
#                 "Your energy is contagious! What's happening that's making you feel so great?"
#             ],
#             'Calm': [
#                 "I appreciate your calm approach. What's on your mind?",
#                 "It's nice to have a peaceful conversation. How can I help you today?"
#             ],
#             'Neutral': [
#                 "I'm here to listen and help. What's on your mind?",
#                 "How are you feeling today? I'm here to support you however I can."
#             ]
#         }
        
#         # Get appropriate response for emotion
#         import random
#         if emotion in emotion_responses:
#             base_response = random.choice(emotion_responses[emotion])
#         else:
#             base_response = "I'm here to listen and help. What's on your mind?"
        
#         # Add personality-aware additions based on Big Five traits
#         if traits.get('neuroticism', 0.5) > 0.7:
#             base_response += " Take your time - there's no pressure to figure everything out right now."
        
#         if traits.get('openness', 0.5) > 0.7:
#             base_response += " Sometimes exploring our feelings can lead to new insights about ourselves."
        
#         if traits.get('extraversion', 0.5) < 0.3:
#             base_response += " It's perfectly fine to process things quietly and at your own pace."
        
#         if traits.get('conscientiousness', 0.5) > 0.7:
#             base_response += " I know you like to think things through carefully, which is a real strength."
        
#         if traits.get('agreeableness', 0.5) > 0.7:
#             base_response += " Your caring nature is something I really appreciate."
        
#         # Add context from similar examples if available
#         if examples:
#             # Look for common patterns in similar interactions
#             similar_emotions = [ex.get('emotion', 'Neutral') for ex in examples]
#             if len(set(similar_emotions)) == 1 and similar_emotions[0] == emotion:
#                 base_response += f" I notice this {emotion.lower()} feeling has come up before - that's completely normal."
        
#         # Add specific advice based on emotion
#         if emotion == 'Sad':
#             base_response += " Sometimes it helps to do something small and kind for yourself."
#         elif emotion == 'Anxious':
#             base_response += " Try focusing on what you can control right now, rather than what you can't."
#         elif emotion == 'Angry':
#             base_response += " It might help to take a few moments to breathe before deciding how to respond."
#         elif emotion == 'Confused':
#             base_response += " Breaking things down into smaller pieces often makes them clearer."
        
#         return base_response

#     def chat(self, user_input: str, response_time: Optional[str]=None,
#          body_language: Optional[str]=None, speech_attributes: Optional[str]=None,
#          top_k: int=3) -> Dict[str,Any]:

#         start = time.time()
        
#         # Extract or use provided metadata
#         if self.auto_metadata:
#             response_time    = response_time    or self.metadata_extractor.extract_response_time(user_input)
#             speech_attributes= speech_attributes or self.metadata_extractor.extract_speech_attributes(user_input)
#         else:
#             response_time    = response_time    or "normal"
#             speech_attributes= speech_attributes or "clear"

#         # Classify emotion
#         emotion = self.perceptor.classify_single(
#             user_input, response_time, body_language or "normal", speech_attributes
#         )
        
#         if self.auto_metadata and not body_language:
#             body_language = self.metadata_extractor.extract_body_language(emotion)
#         elif not body_language:
#             body_language = "normal"

#         # Infer personality traits
#         traits = self.inferencer.infer_traits(
#             user_input, response_time, body_language, speech_attributes, emotion
#         )

#         # Retrieve similar examples
#         q_emb    = self.retriever.embed_query(user_input, emotion, list(traits.values()))
#         examples = self.retriever.get_top_k(q_emb, k=top_k)

#         # Build context from similar examples
#         context = ""
#         if examples:
#             context = "\n\nSimilar past interactions:\n" + "\n".join(
#                 f"- User: {ex['transcript']} (Emotion: {ex['emotion']})"
#                 for ex in examples
#             )

#         # Generate response - with fallback if API unavailable
#         if not self.client:
#             logger.info("Using fallback response generation (no API)")
#             assistant_response = self._generate_fallback_response(user_input, emotion, traits, examples)
#         else:
#             # Build system prompt
#             system_prompt = f"""You are an empathetic assistant. Respond based on the user's detected emotion and personality traits.

#     User's Current State:
#     - Emotion: {emotion}
#     - Personality Traits: {traits}
#     - Response Time: {response_time}
#     - Body Language: {body_language}
#     - Speech Attributes: {speech_attributes}

#     Respond in a way that acknowledges their emotional state and adapts to their personality. Be warm, understanding, and helpful.{context}"""

#             try:
#                 throttle(self.config)

#                 response = self.client.chat.completions.create(
#                     model=self.config.model_name,
#                     messages=[
#                         {"role": "system", "content": system_prompt},
#                         {"role": "user", "content": user_input}
#                     ],
#                     max_tokens=self.config.max_tokens,
#                     temperature=self.config.temperature
#                 )
                
#                 assistant_response = response.choices[0].message.content.strip()
                
#             except Exception as e:
#                 if "insufficient_quota" in str(e) or "429" in str(e):
#                     logger.warning("API quota exceeded, using fallback response")
#                     assistant_response = self._generate_fallback_response(user_input, emotion, traits, examples)
#                 elif "rate_limit" in str(e).lower():
#                     logger.warning("Rate limit hit, using fallback response")
#                     assistant_response = self._generate_fallback_response(user_input, emotion, traits, examples)
#                 else:
#                     logger.error(f"OpenAI API error: {e}")
#                     assistant_response = self._generate_fallback_response(user_input, emotion, traits, examples)

#         # Store interaction
#         self.retriever.add_interaction(user_input, emotion, traits, assistant_response)
        
#         processing_time = round(time.time() - start, 2)

#         return {
#             'response': assistant_response,
#             'emotion': emotion,
#             'traits': traits,
#             'metadata': {
#                 'response_time': response_time,
#                 'body_language': body_language,
#                 'speech_attributes': speech_attributes,
#                 'auto_metadata': self.auto_metadata,
#                 'processing_time': processing_time
#             },
#             'similar_examples': len(examples)
#         }

#     def save_session(self):
#         """Save new interaction data to separate files"""
#         self.retriever.save_new_data()

# def main():
#     print("🤖 OpenAI-Powered Dialogue System Ready!")
#     print("📁 New interactions will be saved to 'openai_aligned_traits.json' and 'openai_embedding.npy'")
#     print("🔑 Make sure your OPENAI_API_KEY environment variable is set!")
#     # Check API key
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         print("⚠️  Warning: No OPENAI_API_KEY found. System will use fallback responses.")
#         proceed = input("Continue anyway? (y/n): ").strip().lower()
#         if proceed != 'y':
#             return
#     else:
#         print("✅ OpenAI API key found")
    
#     try:
#         agent = OpenAIDialogueAgent(auto_metadata=True)
#         print("✅ System initialized successfully!")
        
#         if not agent.client:
#             print("⚠️  Running in fallback mode (local processing only)")
            
#     except Exception as e:
#         print(f"❌ Error initializing system: {e}")
#         return
    
#     print("Type 'exit' to quit.\n")

#     while True:
#         try:
#             ui = input("You: ").strip()
#             if ui.lower() in ('exit','quit'):
#                 agent.save_session()
#                 print("👋 Session saved. Goodbye!")
#                 break
                
#             rt = input("Response time (Enter to auto): ").strip() or None
#             bl = input("Body language (Enter to auto): ").strip() or None
#             sa = input("Speech attrs (Enter to auto): ").strip() or None
            
#             print("🤔 Processing...")
#             res = agent.chat(ui, rt, bl, sa)
            
#             print(f"\n🤖 Assistant: {res['response']}")
#             print(f"📊 Emotion: {res['emotion']}")
#             print(f"🧠 Traits: {', '.join([f'{k}: {v:.2f}' for k, v in res['traits'].items()])}")
#             print(f"🔍 Similar examples: {res['similar_examples']}")
#             print(f"⏱️  Processing time: {res['metadata']['processing_time']}s\n")
            
#         except KeyboardInterrupt:
#             print("\n👋 Interrupted. Saving session...")
#             agent.save_session()
#             break
#         except Exception as e:
#             print(f"❌ Error: {e}")
#             print("Please try again.\n")

# if __name__ == "__main__":
#     main()






import os
import re
import json
import numpy as np
import logging
import time
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- Improved Rate Limiting ----------------
class RateLimiter:
    """Thread-safe rate limiter for OpenAI API calls"""
    
    def __init__(self, requests_per_minute: int = 20, tokens_per_minute: int = 40000):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_times = []
        self.token_usage = []
        self.lock = threading.Lock()
        
        # More conservative timing
        self.min_request_interval = 60.0 / requests_per_minute  # seconds between requests
        self.last_request_time = 0
        
    def wait_if_needed(self, estimated_tokens: int = 200):
        """Wait if necessary to respect rate limits"""
        with self.lock:
            now = time.time()
            
            # 1. Enforce minimum interval between requests
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_request_interval:
                wait_time = self.min_request_interval - time_since_last
                logger.info(f"Rate limiting: waiting {wait_time:.2f}s between requests")
                time.sleep(wait_time)
                now = time.time()
            
            # 2. Clean old entries (older than 1 minute)
            cutoff_time = now - 60
            self.request_times = [t for t in self.request_times if t > cutoff_time]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]
            
            # 3. Check request rate limit
            if len(self.request_times) >= self.requests_per_minute:
                oldest_request = min(self.request_times)
                wait_time = 60 - (now - oldest_request) + 1  # +1 for buffer
                logger.info(f"Request rate limit: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                now = time.time()
                # Clean again after waiting
                cutoff_time = now - 60
                self.request_times = [t for t in self.request_times if t > cutoff_time]
            
            # 4. Check token rate limit
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            if current_tokens + estimated_tokens > self.tokens_per_minute:
                if self.token_usage:
                    oldest_token_time = min(t for t, _ in self.token_usage)
                    wait_time = 60 - (now - oldest_token_time) + 1
                    logger.info(f"Token rate limit: waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                    now = time.time()
            
            # 5. Record this request
            self.request_times.append(now)
            self.token_usage.append((now, estimated_tokens))
            self.last_request_time = now

# ---------------- Configuration ----------------
class ModelConfig:
    """Configuration class for model parameters"""
    def __init__(self):
        # Use more conservative settings
        self.model_name = "gpt-3.5-turbo"
        self.max_tokens = 80  # Reduced further
        self.temperature = 0.3  # More deterministic
        
        self.emotions = [
            "Happy", "Sad", "Angry", "Anxious", "Surprised",
            "Disgusted", "Confused", "Calm", "Excited",
            "Embarrassed", "Guilty", "Neutral"
        ]
        
        self.big_five_traits = [
            "openness", "conscientiousness", "extraversion",
            "agreeableness", "neuroticism"
        ]
        
        # Rate limiting configuration
        self.requests_per_minute = 15  # Very conservative
        self.tokens_per_minute = 30000  # Conservative token limit
        self.max_retries = 2
        self.base_retry_delay = 2.0
        self.enable_fallback = True

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

# ---------------- Enhanced OpenAI Client ----------------
class RobustOpenAIClient:
    """OpenAI client with robust error handling and rate limiting"""
    
    def __init__(self, config: ModelConfig, api_key: Optional[str] = None):
        self.config = config
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.requests_per_minute,
            tokens_per_minute=config.tokens_per_minute
        )
        
        try:
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self._test_connection()
            self.available = True
        except Exception as e:
            logger.error(f"OpenAI client initialization failed: {e}")
            self.client = None
            self.available = False
    
    def _test_connection(self):
        """Test API connection with minimal request"""
        try:
            # Wait for rate limiting
            self.rate_limiter.wait_if_needed(10)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                timeout=10
            )
            logger.info("✅ OpenAI API connection successful")
        except Exception as e:
            logger.warning(f"API connection test failed: {e}")
            if any(term in str(e).lower() for term in ['quota', '429', 'rate', 'limit']):
                logger.error("❌ Rate limit or quota issue detected")
                raise
    
    def make_request(self, messages: List[Dict], max_tokens: int = None, temperature: float = None):
        """Make a robust API request with retries and rate limiting"""
        if not self.available:
            raise Exception("OpenAI client not available")
        
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        # Estimate tokens for rate limiting
        estimated_tokens = sum(len(msg['content'].split()) * 1.3 for msg in messages) + max_tokens
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed(int(estimated_tokens))
                
                logger.info(f"Making API request (attempt {attempt + 1}/{self.config.max_retries + 1})")
                
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=30  # 30 second timeout
                )
                
                # Log actual token usage if available
                if hasattr(response, 'usage') and response.usage:
                    logger.info(f"Tokens used: {response.usage.total_tokens}")
                
                return response
                
            except Exception as e:
                error_str = str(e).lower()
                
                if any(term in error_str for term in ['rate', 'limit', '429']):
                    wait_time = (2 ** attempt) * self.config.base_retry_delay  # Exponential backoff
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                    
                elif any(term in error_str for term in ['quota', 'insufficient']):
                    logger.error("❌ API quota exceeded - no retries")
                    self.available = False
                    raise
                    
                elif 'timeout' in error_str:
                    logger.warning(f"Request timeout on attempt {attempt + 1}")
                    if attempt < self.config.max_retries:
                        time.sleep(self.config.base_retry_delay * (attempt + 1))
                        continue
                    
                else:
                    logger.error(f"API error on attempt {attempt + 1}: {e}")
                    if attempt < self.config.max_retries:
                        time.sleep(self.config.base_retry_delay)
                        continue
                
                # If we're here, we've exhausted retries
                if attempt == self.config.max_retries:
                    logger.error("❌ All retry attempts failed")
                    raise

# ---------------- Perception Module ----------------
class OpenAIEmotionClassifier:
    def __init__(self, config: ModelConfig, openai_api_key: Optional[str] = None):
        self.config = config
        self.openai_client = RobustOpenAIClient(config, openai_api_key)

    def _create_prompt(self, text, rt, bl, sa) -> str:
        emo_list = ", ".join(self.config.emotions)
        return f"""Classify the emotion from these options: {emo_list}

Text: "{text}"
Response time: {rt}
Body language: {bl}
Speech: {sa}

Reply with ONLY the emotion name."""
    
    def classify_single(self, text, rt, bl, sa) -> str:
        # Always have fallback ready
        fallback_emotion = self._fallback_emotion_classification(text, rt, bl, sa)
        
        if not self.openai_client.available:
            logger.info("Using fallback emotion classification (API unavailable)")
            return fallback_emotion
        
        try:
            messages = [
                {"role": "system", "content": "You are an emotion classifier. Respond with only the emotion name."},
                {"role": "user", "content": self._create_prompt(text, rt, bl, sa)}
            ]
            
            response = self.openai_client.make_request(messages, max_tokens=10, temperature=0.1)
            result = response.choices[0].message.content.strip()
            
            # Validate result
            pattern = r"\b(" + "|".join(self.config.emotions) + r")\b"
            match = re.search(pattern, result, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
            else:
                logger.warning(f"Invalid emotion response: {result}, using fallback")
                return fallback_emotion
            
        except Exception as e:
            logger.warning(f"Emotion classification failed: {e}, using fallback")
            return fallback_emotion

    def _fallback_emotion_classification(self, text, rt, bl, sa) -> str:
        """Enhanced local emotion classification"""
        text_lower = text.lower()
        
        # Enhanced keyword mapping
        emotion_keywords = {
            'Happy': ['happy', 'joy', 'great', 'awesome', 'excited', 'wonderful', 'good', 'fantastic', 'love', 'amazing', 'perfect'],
            'Sad': ['sad', 'down', 'depressed', 'blue', 'unhappy', 'upset', 'disappointed', 'hurt', 'crying', 'grief'],
            'Angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated', 'hate', 'rage', 'pissed'],
            'Anxious': ['anxious', 'worried', 'nervous', 'stressed', 'panic', 'afraid', 'scared', 'tense', 'uneasy'],
            'Surprised': ['surprised', 'shocked', 'amazed', 'wow', 'incredible', 'unbelievable', 'stunned'],
            'Confused': ['confused', 'lost', 'unclear', 'puzzled', "don't understand", 'bewildered', 'perplexed'],
            'Excited': ['excited', 'thrilled', 'pumped', 'eager', 'enthusiastic', 'energetic'],
            'Calm': ['calm', 'peaceful', 'relaxed', 'serene', 'tranquil', 'composed'],
            'Embarrassed': ['embarrassed', 'ashamed', 'humiliated', 'mortified', 'awkward'],
            'Guilty': ['guilty', 'sorry', 'regret', 'ashamed', 'remorse']
        }
        
        # Score each emotion based on keywords
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Return highest scoring emotion
        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        
        # Context-based fallback
        if rt == 'fast' and ('!' in text or text.isupper()):
            return 'Excited'
        elif rt == 'slow' and ('...' in text or '?' in text):
            return 'Confused'
        elif len(text.split()) < 3:
            return 'Neutral'
        
        return 'Neutral'

# ---------------- Enhanced Inference Module ----------------
class OpenAIInferenceAgent:
    def __init__(self, config: ModelConfig, openai_api_key: Optional[str] = None):
        self.config = config
        self.openai_client = RobustOpenAIClient(config, openai_api_key)

    def _build_prompt(self, text, rt, bl, sa, emo) -> str:
        return f"""Rate personality traits 0.0-1.0 based on this text. Return JSON only.

Text: "{text}"
Emotion: {emo}
Context: {rt} response, {bl} body language, {sa} speech

Rate these Big Five traits:
- openness: creativity, openness to experience
- conscientiousness: organization, self-discipline  
- extraversion: sociability, assertiveness
- agreeableness: cooperation, trust
- neuroticism: emotional instability, anxiety

Return only: {{"openness":0.7,"conscientiousness":0.6,"extraversion":0.5,"agreeableness":0.8,"neuroticism":0.3}}"""

    def infer_traits(self, text, rt, bl, sa, emo) -> Dict[str, float]:
        # Always compute heuristic baseline
        heuristic_scores = self._heuristic_scoring(text, emo, rt, bl, sa)
        
        if not self.openai_client.available:
            logger.info("Using heuristic-only trait scoring (API unavailable)")
            return heuristic_scores
        
        try:
            messages = [
                {"role": "system", "content": "You are a personality expert. Return only valid JSON with trait scores 0.0-1.0."},
                {"role": "user", "content": self._build_prompt(text, rt, bl, sa, emo)}
            ]
            
            response = self.openai_client.make_request(messages, max_tokens=60, temperature=0.1)
            result = response.choices[0].message.content.strip()
            
            # Extract and validate JSON
            api_scores = self._extract_json(result)
            if not api_scores:
                logger.warning("Failed to extract valid JSON, using heuristic scores")
                return heuristic_scores
            
            # Blend scores (30% heuristic, 70% API for more API influence when available)
            final_scores = {}
            for trait in self.config.big_five_traits:
                h_score = heuristic_scores[trait]
                a_score = float(api_scores.get(trait, h_score))
                a_score = max(0.0, min(1.0, a_score))  # Clamp to valid range
                final_scores[trait] = 0.3 * h_score + 0.7 * a_score
            
            return final_scores
            
        except Exception as e:
            logger.warning(f"Trait inference failed: {e}, using heuristic scores")
            return heuristic_scores

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Enhanced JSON extraction"""
        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        if json_match:
            try:
                json_obj = json.loads(json_match.group(0))
                # Validate that it contains expected keys
                if all(trait in json_obj for trait in self.config.big_five_traits):
                    return json_obj
            except json.JSONDecodeError:
                pass
        
        # Try to extract individual values if JSON format failed
        scores = {}
        for trait in self.config.big_five_traits:
            pattern = rf'"{trait}":\s*([0-9.]+)'
            match = re.search(pattern, text)
            if match:
                try:
                    scores[trait] = float(match.group(1))
                except ValueError:
                    continue
        
        return scores if len(scores) == len(self.config.big_five_traits) else None

    def _heuristic_scoring(self, text, emo, rt, bl, sa) -> Dict[str, float]:
        """Enhanced heuristic scoring"""
        tl = text.lower()
        scores = {trait: 0.5 for trait in self.config.big_five_traits}

        # Emotion-based adjustments (more nuanced)
        emotion_adjustments = {
            'Happy': {'extraversion': 0.15, 'neuroticism': -0.15, 'agreeableness': 0.1},
            'Excited': {'extraversion': 0.2, 'neuroticism': -0.1, 'openness': 0.1},
            'Sad': {'neuroticism': 0.2, 'extraversion': -0.1},
            'Anxious': {'neuroticism': 0.25, 'conscientiousness': -0.05},
            'Angry': {'neuroticism': 0.3, 'agreeableness': -0.25},
            'Calm': {'neuroticism': -0.2, 'conscientiousness': 0.1},
            'Confused': {'openness': 0.1, 'neuroticism': 0.05}
        }
        
        if emo in emotion_adjustments:
            for trait, adjustment in emotion_adjustments[emo].items():
                scores[trait] = max(0.0, min(1.0, scores[trait] + adjustment))

        # Text analysis
        word_count = len(text.split())
        if word_count > 20:  # Verbose
            scores['extraversion'] = min(0.9, scores['extraversion'] + 0.1)
        elif word_count < 5:  # Terse
            scores['extraversion'] = max(0.1, scores['extraversion'] - 0.1)

        # Keywords analysis
        keywords = {
            'openness': ['creative', 'art', 'music', 'imagine', 'dream', 'new', 'different', 'explore'],
            'conscientiousness': ['plan', 'organize', 'schedule', 'work', 'duty', 'responsible', 'careful'],
            'extraversion': ['friends', 'party', 'social', 'people', 'talk', 'meet', 'group'],
            'agreeableness': ['help', 'kind', 'please', 'thank', 'sorry', 'together', 'share'],
            'neuroticism': ['worry', 'stress', 'problem', 'difficult', 'hard', 'trouble', 'fear']
        }
        
        for trait, words in keywords.items():
            keyword_count = sum(1 for word in words if word in tl)
            if keyword_count > 0:
                adjustment = min(0.2, keyword_count * 0.05)
                scores[trait] = min(1.0, scores[trait] + adjustment)

        return scores

# ---------------- Continue with existing Retriever class... ----------------
# [The Retriever class remains the same as in your original code]

# ---------------- Enhanced Dialogue Agent ----------------
class OpenAIDialogueAgent:
    def __init__(self, original_data_path='aligned_data_with_traits.json', 
                 original_emb_path='full_embeddings.npy',
                 new_data_path='openai_aligned_traits.json',
                 new_emb_path='openai_embedding.npy',
                 openai_api_key: Optional[str]=None, auto_metadata: bool=True):
        
        self.config = ModelConfig()
        self.auto_metadata = auto_metadata
        self.metadata_extractor = MetadataExtractor()
        self.perceptor = OpenAIEmotionClassifier(self.config, openai_api_key)
        self.inferencer = OpenAIInferenceAgent(self.config, openai_api_key)
        
        # Initialize separate client for dialogue generation
        self.dialogue_client = RobustOpenAIClient(self.config, openai_api_key)
        
        # Note: Retriever class would go here - keeping your existing implementation

    def _generate_fallback_response(self, user_input: str, emotion: str, traits: Dict[str, float], examples: List[Dict] = None) -> str:
        """Enhanced fallback response generation"""
        
        # Your existing fallback logic here, but enhanced...
        emotion_responses = {
            'Sad': [
                "I can sense you're going through a difficult time. Your feelings are completely valid.",
                "It sounds like you're feeling down. Would you like to talk about what's weighing on your mind?",
                "I hear the sadness in your words. Sometimes acknowledging these feelings is the first step."
            ],
            'Happy': [
                "I love hearing the joy in your message! What's bringing you such happiness?",
                "Your positive energy really comes through. I'd love to hear more about what's going well.",
                "It's wonderful to connect with you when you're feeling so good!"
            ],
            'Anxious': [
                "I can feel the worry in your words. Take a deep breath - you're not facing this alone.",
                "Anxiety can feel overwhelming. Would it help to talk through what's making you feel this way?",
                "I understand you're feeling anxious. Sometimes sharing these feelings can provide relief."
            ],
            'Angry': [
                "I can sense your frustration. It's important to acknowledge these strong feelings.",
                "Your anger is coming through clearly. Would you like to talk about what's triggered this?",
                "I hear how upset you are. Let's work through this together."
            ],
            'Excited': [
                "Your excitement is contagious! I'd love to hear what has you feeling so energized.",
                "I can feel your enthusiasm! Tell me more about what's got you so pumped up."
            ],
            'Confused': [
                "I can sense you're trying to work through something complex. Let's break it down together.",
                "It's completely normal to feel confused sometimes. What aspects are you struggling with?"
            ],
            'Neutral': [
                "I'm here and ready to listen. What's on your mind today?",
                "How are you feeling? I'm here to support you however I can."
            ]
        }
        
        import random
        base_response = random.choice(emotion_responses.get(emotion, emotion_responses['Neutral']))
        
        # Add personality-aware elements
        if traits.get('neuroticism', 0.5) > 0.7:
            base_response += " Remember, it's okay to take things one step at a time."
        
        if traits.get('openness', 0.5) > 0.7:
            base_response += " Your openness to exploring your feelings shows real self-awareness."
        
        if traits.get('extraversion', 0.5) < 0.3:
            base_response += " I appreciate that you're sharing this with me, even if talking doesn't always come easily."
        
        return base_response

    def chat(self, user_input: str, response_time: Optional[str]=None,
             body_language: Optional[str]=None, speech_attributes: Optional[str]=None,
             top_k: int=3) -> Dict[str,Any]:

        start = time.time()
        
        # Extract metadata
        if self.auto_metadata:
            response_time = response_time or self.metadata_extractor.extract_response_time(user_input)
            speech_attributes = speech_attributes or self.metadata_extractor.extract_speech_attributes(user_input)
        else:
            response_time = response_time or "normal"
            speech_attributes = speech_attributes or "clear"

        # Classify emotion
        emotion = self.perceptor.classify_single(
            user_input, response_time, body_language or "normal", speech_attributes
        )
        
        if self.auto_metadata and not body_language:
            body_language = self.metadata_extractor.extract_body_language(emotion)
        elif not body_language:
            body_language = "normal"

        # Infer personality traits
        traits = self.inferencer.infer_traits(
            user_input, response_time, body_language, speech_attributes, emotion
        )

        # Generate response
        assistant_response = self._generate_response(user_input, emotion, traits, response_time, body_language, speech_attributes)
        
        processing_time = round(time.time() - start, 2)

        return {
            'response': assistant_response,
            'emotion': emotion,
            'traits': traits,
            'metadata': {
                'response_time': response_time,
                'body_language': body_language,
                'speech_attributes': speech_attributes,
                'api_available': self.dialogue_client.available,
                'processing_time': processing_time
            }
        }

    def _generate_response(self, user_input: str, emotion: str, traits: Dict[str, float], 
                          response_time: str, body_language: str, speech_attributes: str) -> str:
        """Generate response with API or fallback"""
        
        fallback_response = self._generate_fallback_response(user_input, emotion, traits)
        
        if not self.dialogue_client.available:
            logger.info("Using fallback response (API unavailable)")
            return fallback_response
        
        try:
            system_prompt = f"""You are an empathetic assistant. Respond warmly based on the user's emotional state and personality.

User Analysis:
- Emotion: {emotion}
- Personality: {traits}
- Response speed: {response_time}
- Body language: {body_language}  
- Speech style: {speech_attributes}

Adapt your response to their emotional needs and personality. Be understanding, supportive, and genuine."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            response = self.dialogue_client.make_request(messages)
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"Response generation failed: {e}, using fallback")
            return fallback_response

def main():
    print("🤖 Enhanced OpenAI Dialogue System")
    print("🔧 Features: Robust rate limiting, comprehensive fallbacks, better error handling")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found. System will use local fallbacks.")
        proceed = input("Continue? (y/n): ").strip().lower()
        if proceed != 'y':
            return
    else:
        print("✅ OpenAI API key detected")
    
    try:
        agent = OpenAIDialogueAgent(auto_metadata=True)
        print("✅ System initialized successfully!")
        
        # Check API availability
        api_status = "✅ Available" if agent.dialogue_client.available else "⚠️ Unavailable (using fallbacks)"
        print(f"🔗 API Status: {api_status}")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return
    
    print("\nType 'exit' to quit, 'status' for system info\n")

    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ('exit', 'quit'):
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'status':
                api_status = "Available" if agent.dialogue_client.available else "Unavailable"
                print(f"📊 System Status:")
                print(f"   API: {api_status}")
                print(f"   Rate Limit: {agent.dialogue_client.rate_limiter.requests_per_minute} req/min")
                print(f"   Recent requests: {len(agent.dialogue_client.rate_limiter.request_times)}")
                continue
                
            if not user_input:
                continue
            
            print("🤔 Processing...")
            result = agent.chat(user_input)
            
            print(f"\n🤖 Assistant: {result['response']}")
            print(f"📊 Emotion: {result['emotion']}")
            print(f"🧠 Traits: {', '.join([f'{k}: {v:.2f}' for k, v in result['traits'].items()])}")
            
            # Display metadata
            metadata = result['metadata']
            print(f"🔍 Metadata: {metadata['response_time']} response, {metadata['body_language']} body language, {metadata['speech_attributes']} speech")
            print(f"⏱️  Processing time: {metadata['processing_time']}s")
            
            # Show API status if relevant
            if not metadata['api_available']:
                print("ℹ️  Response generated using local fallbacks")
            
            print()  # Add spacing
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()