import os
import re
import json
import torch
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelConfig:
    """Configuration class for model parameters"""
    def __init__(self):
        self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
        self.max_length = 512
        self.max_new_tokens = 128
        self.temperature = 0.0
        self.emotions = [
            "Happy", "Sad", "Angry", "Anxious", "Surprised", 
            "Disgusted", "Confused", "Calm", "Excited", 
            "Embarrassed", "Guilty", "Neutral"
        ]
        self.big_five_traits = [
            "openness", "conscientiousness", "extraversion", 
            "agreeableness", "neuroticism"
        ]

# ---------------- Perception Module ----------------
class LlamaEmotionClassifier:
    """Emotion classifier using Llama model"""
    
    def __init__(self, config: ModelConfig, hf_token: Optional[str] = None):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._authenticate(hf_token)
        self._load_model()

    def _authenticate(self, hf_token: Optional[str]) -> None:
        """Authenticate with Hugging Face"""
        token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        if token:
            try:
                login(token=token)
                logger.info("Successfully authenticated with Hugging Face")
            except Exception as e:
                logger.warning(f"Failed to authenticate with Hugging Face: {e}")

    def _load_model(self) -> None:
        """Load the emotion classification model"""
        try:
            logger.info(f"Loading Perception Model: {self.config.model_name} on {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, 
                use_fast=True, 
                trust_remote_code=True
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map={"": self.device},
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            logger.info("Perception model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load perception model: {e}")
            raise

    def _create_prompt(self, text: str, response_time: str, body_language: str, speech_attributes: str) -> str:
        """Create a structured prompt for emotion classification"""
        emotions_list = ", ".join(self.config.emotions)
        return (
            f"Classify the emotion from the following inputs. "
            f"Choose exactly one emotion from: {emotions_list}.\n\n"
            f"Text: \"{text}\"\n"
            f"Response time: {response_time}\n"
            f"Body language: {body_language}\n"
            f"Speech attributes: {speech_attributes}\n\n"
            f"Emotion:"
        )

    def classify_single(self, text: str, response_time: str, body_language: str, speech_attributes: str) -> str:
        """Classify emotion from multimodal inputs"""
        try:
            prompt = self._create_prompt(text, response_time, body_language, speech_attributes)
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.config.max_length
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,  # Only need one word
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Extract emotion using regex
            emotion_pattern = r"\b(" + "|".join(self.config.emotions) + r")\b"
            match = re.search(emotion_pattern, response, re.IGNORECASE)
            
            return match.group(1).capitalize() if match else "Neutral"
            
        except Exception as e:
            logger.error(f"Error in emotion classification: {e}")
            return "Neutral"

# ---------------- Inference Module ----------------
class LlamaInferenceAgent:
    """Big Five personality traits inference agent"""
    
    def __init__(self, config: ModelConfig, hf_token: Optional[str] = None, device: Optional[str] = None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._authenticate(hf_token)
        self._load_model()

    def _authenticate(self, hf_token: Optional[str]) -> None:
        """Authenticate with Hugging Face"""
        token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        if token:
            try:
                login(token=token)
            except Exception as e:
                logger.warning(f"Failed to authenticate: {e}")

    def _load_model(self) -> None:
        """Load the inference model"""
        try:
            logger.info(f"Loading Inference Model: {self.config.model_name} on {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, 
                use_fast=False, 
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map={"": self.device},
                torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            logger.info("Inference model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load inference model: {e}")
            raise

    def _build_prompt(self, text: str, response_time: str, body_language: str, 
                     speech_attributes: str, emotion: str) -> str:
        """Build prompt for personality trait inference"""
        system_prompt = (
            "Analyze the following inputs and assign Big Five personality trait scores "
            "from 0.0 to 1.0. Output only valid JSON format with keys: "
            f"{', '.join(self.config.big_five_traits)}."
        )
        
        return (
            f"<|system|>\n{system_prompt}\n<|user|>\n"
            f"Transcript: {text}\n"
            f"Response Time: {response_time}\n"
            f"Body Language: {body_language}\n"
            f"Speech Attributes: {speech_attributes}\n"
            f"Emotion: {emotion}\n"
            f"Big Five Scores (JSON):"
        )

    def infer_traits(self, text: str, response_time: str, body_language: str, 
                    speech_attributes: str, emotion: str) -> Dict[str, float]:
        """Infer Big Five personality traits"""
        try:
            prompt = self._build_prompt(text, response_time, body_language, speech_attributes, emotion)
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4096
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    temperature=self.config.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                traits_dict = json.loads(json_match.group(0))
                # Validate and normalize scores
                return self._validate_traits(traits_dict)
            else:
                logger.warning("No valid JSON found in traits inference response")
                return self._default_traits()
                
        except Exception as e:
            logger.error(f"Error in trait inference: {e}")
            return self._default_traits()

    def _validate_traits(self, traits: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize trait scores"""
        validated = {}
        for trait in self.config.big_five_traits:
            score = traits.get(trait, 0.5)
            # Ensure score is between 0 and 1
            validated[trait] = max(0.0, min(1.0, float(score)))
        return validated

    def _default_traits(self) -> Dict[str, float]:
        """Return default trait scores"""
        return {trait: 0.5 for trait in self.config.big_five_traits}

# ---------------- Retrieval Module ----------------
class Retriever:
    """Retrieval system for finding similar examples"""
    
    def __init__(self, data_path: str, embeddings_path: str):
        self.data_path = Path(data_path)
        self.embeddings_path = Path(embeddings_path)
        self._load_data()
        self._setup_vectorizers()

    def _load_data(self) -> None:
        """Load training data and embeddings"""
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            if not self.embeddings_path.exists():
                raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")
            
            self.full_embeddings = np.load(self.embeddings_path)
            logger.info(f"Loaded {len(self.data)} examples and embeddings")
            
        except Exception as e:
            logger.error(f"Failed to load retrieval data: {e}")
            raise

    def _setup_vectorizers(self) -> None:
        """Setup TF-IDF vectorizer and one-hot encoder"""
        try:
            texts = [entry['transcript'] for entry in self.data]
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.vectorizer.fit(texts)
            
            emotions = [[entry['emotion']] for entry in self.data]
            self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.one_hot_encoder.fit(emotions)
            
        except Exception as e:
            logger.error(f"Failed to setup vectorizers: {e}")
            raise

    def embed_query(self, text: str, emotion: str, traits: List[float]) -> np.ndarray:
        """Create embedding for query"""
        try:
            # Text embedding
            text_embedding = self.vectorizer.transform([text]).toarray()
            
            # Emotion embedding
            emotion_embedding = self.one_hot_encoder.transform([[emotion]])
            
            # Traits embedding
            traits_embedding = np.array([traits])
            
            # Concatenate all embeddings
            query_embedding = np.hstack([text_embedding, emotion_embedding, traits_embedding])
            return query_embedding[0]
            
        except Exception as e:
            logger.error(f"Failed to create query embedding: {e}")
            # Return zero embedding as fallback
            return np.zeros(self.full_embeddings.shape[1])

    def get_top_k(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k similar examples"""
        try:
            similarities = cosine_similarity([query_embedding], self.full_embeddings)[0]
            top_indices = similarities.argsort()[-k:][::-1]
            
            return [self.data[i] for i in top_indices if i < len(self.data)]
            
        except Exception as e:
            logger.error(f"Failed to retrieve similar examples: {e}")
            return []

# ---------------- Dialogue Module ----------------
class LlamaDialogueAgent:
    """Main dialogue agent that orchestrates all components"""
    
    def __init__(self,
                 data_path: str = 'aligned_data_with_traits.json',
                 embeddings_path: str = 'full_embeddings.npy',
                 model_name: Optional[str] = None,
                 hf_token: Optional[str] = None):
        
        self.config = ModelConfig()
        if model_name:
            self.config.model_name = model_name
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self._initialize_components(data_path, embeddings_path, hf_token)

    def _initialize_components(self, data_path: str, embeddings_path: str, hf_token: Optional[str]) -> None:
        """Initialize all system components"""
        try:
            # Initialize perception module
            self.perception_agent = LlamaEmotionClassifier(self.config, hf_token)
            
            # Initialize inference module
            self.inference_agent = LlamaInferenceAgent(self.config, hf_token, self.device)
            
            # Initialize retrieval module
            if Path(data_path).exists() and Path(embeddings_path).exists():
                self.retriever = Retriever(data_path, embeddings_path)
            else:
                logger.warning("Retrieval data not found. Retrieval functionality disabled.")
                self.retriever = None
            
            # Initialize dialogue model
            self._load_dialogue_model(hf_token)
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def _load_dialogue_model(self, hf_token: Optional[str]) -> None:
        """Load the dialogue generation model"""
        try:
            token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
            if token:
                login(token=token)
            
            logger.info(f"Loading Dialogue Model: {self.config.model_name} on {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, 
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map={"": self.device},
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            logger.info("Dialogue model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load dialogue model: {e}")
            raise

    def chat(self, user_input: str, response_time: str = "normal", 
             body_language: str = "neutral", speech_attributes: str = "clear", 
             top_k: int = 3) -> Dict[str, Any]:
        """Main chat function that processes user input and generates response"""
        
        # Step 1: Emotion classification
        emotion = self.perception_agent.classify_single(
            user_input, response_time, body_language, speech_attributes
        )
        
        # Step 2: Personality trait inference
        traits = self.inference_agent.infer_traits(
            user_input, response_time, body_language, speech_attributes, emotion
        )
        
        # Step 3: Retrieve similar examples (if available)
        similar_examples = []
        if self.retriever:
            query_embedding = self.retriever.embed_query(
                user_input, emotion, list(traits.values())
            )
            similar_examples = self.retriever.get_top_k(query_embedding, k=top_k)
        
        # Step 4: Generate response
        response = self._generate_response(
            user_input, emotion, traits, similar_examples
        )
        
        return {
            'response': response,
            'emotion': emotion,
            'traits': traits,
            'similar_examples': len(similar_examples)
        }

    def _generate_response(self, user_input: str, emotion: str, 
                          traits: Dict[str, float], similar_examples: List[Dict]) -> str:
        """Generate contextual response"""
        try:
            # Build context from similar examples
            context = ""
            if similar_examples:
                context = "Similar user interactions:\n" + "\n".join([
                    f"- User: {ex['transcript']} (Emotion: {ex['emotion']})"
                    for ex in similar_examples[:3]
                ])
            
            # Create system prompt
            system_prompt = (
                "You are an empathetic assistant. Respond appropriately based on the user's "
                "emotional state and personality traits. Be supportive and understanding."
            )
            
            # Build full prompt
            prompt = (
                f"<|system|>\n{system_prompt}\n"
                f"User Emotion: {emotion}\n"
                f"User Traits: {traits}\n"
                f"{context}\n"
                f"<|user|>\n{user_input}\n<|assistant|>"
            )
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4096
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I'm sorry, I'm having trouble understanding right now. Could you please try again?"

# ---------------- Main Pipeline ----------------
def main():
    """Main function to run the dialogue system"""
    try:
        # Initialize the dialogue agent
        agent = LlamaDialogueAgent()
        
        print("🤖 Multi-Modal Dialogue System Ready!")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("-" * 50)
        
        while True:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            if user_input.lower() in ('exit', 'quit', 'bye'):
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Get optional metadata (with defaults)
            response_time = input("⏱️  Response time (press Enter for 'normal'): ").strip() or "normal"
            body_language = input("🤸 Body language (press Enter for 'neutral'): ").strip() or "neutral"
            speech_attributes = input("🗣️  Speech attributes (press Enter for 'clear'): ").strip() or "clear"
            
            # Process input and generate response
            print("🔄 Processing...")
            result = agent.chat(user_input, response_time, body_language, speech_attributes)
            
            # Display results
            print(f"\n🤖 Assistant: {result['response']}")
            print(f"😊 Detected emotion: {result['emotion']}")
            print(f"🧠 Personality traits: {result['traits']}")
            print(f"📚 Similar examples used: {result['similar_examples']}")
            print("-" * 50)
            
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()