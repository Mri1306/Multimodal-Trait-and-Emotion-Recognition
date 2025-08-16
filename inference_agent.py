import os
import re
import json
import torch
from typing import Optional, Dict
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

class LlamaInferenceAgent:
    """
    Loads the Llama-3.2-1b-Instruct model on GPU/CPU and infers
    Big-Five personality traits given transcript, metadata, and emotion.
    """
    def _init_(
        self,
        model_name: str = "meta-llama/Llama-3.2-1b-Instruct",
        hf_token: Optional[str] = None,
        device: Optional[str] = None
    ):
        # Authenticate to HF if token provided
        token = hf_token or os.getenv("HF_TOKEN")
        if token:
            login(token=token)

        # Device selection
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer & model
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=(torch.bfloat16 if "cuda" in self.device else torch.float32),
            trust_remote_code=True
        )
        self.model.eval()
        print("✅ Model loaded.")

    def _build_prompt(
        self,
        transcript: str,
        response_time: str,
        body_language: str,
        speech_attributes: str,
        perception_label: str
    ) -> str:
        system = (
            "You are an expert inference agent. Given the transcript, metadata, "
            "and perceived emotion label, think step-by-step to assign a score "
            "between 0.0 and 1.0 for each of the Big Five traits: "
            "Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism. "
            "Then output JSON only, for example: "
            "{'Openness':0.7,'Conscientiousness':0.5,'Extraversion':0.8,'Agreeableness':0.6,'Neuroticism':0.3}"
        )

        # Few‐shot examples
        examples = [
            {
                "transcript": "I love collaborating on new projects and brainstorming ideas.",
                "response_time": "Fast (1-2 seconds)",
                "body_language": "Leaning forward, animated gestures",
                "speech_attributes": "Energetic, clear",
                "emotion": "Happy",
                "scores": {
                    "Openness": 0.9,
                    "Conscientiousness": 0.6,
                    "Extraversion": 0.85,
                    "Agreeableness": 0.7,
                    "Neuroticism": 0.2
                }
            },
            {
                "transcript": "I missed the deadline and felt terrible, I always second-guess myself.",
                "response_time": "Slow (5-6 seconds)",
                "body_language": "Slumped shoulders, downcast eyes",
                "speech_attributes": "Soft, hesitant",
                "emotion": "Sad",
                "scores": {
                    "Openness": 0.4,
                    "Conscientiousness": 0.8,
                    "Extraversion": 0.3,
                    "Agreeableness": 0.6,
                    "Neuroticism": 0.75
                }
            }
        ]

        parts = [f"<|system|>\n{system}\n"]
        for ex in examples:
            parts.append(
                "<|user|>\n"
                f"Transcript: {ex['transcript']}\n"
                f"Response Time: {ex['response_time']}\n"
                f"Body Language: {ex['body_language']}\n"
                f"Speech Attributes: {ex['speech_attributes']}\n"
                f"Perceived Emotion: {ex['emotion']}\n"
                "Chain of Thought:\n"
                "Final Scores: " + json.dumps(ex['scores']) + "\n"
            )
        parts.append(
            "<|user|>\n"
            f"Transcript: {transcript}\n"
            f"Response Time: {response_time}\n"
            f"Body Language: {body_language}\n"
            f"Speech Attributes: {speech_attributes}\n"
            f"Perceived Emotion: {perception_label}\n"
            "Chain of Thought:\n"
            "Final Scores:"
        )
        return "\n".join(parts)

    def infer_traits(
        self,
        transcript: str,
        response_time: str,
        body_language: str,
        speech_attributes: str,
        perception_label: str
    ) -> Dict[str, any]:
        prompt = self._build_prompt(
            transcript, response_time, body_language, speech_attributes, perception_label
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        gen = self.tokenizer.decode(
            output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True
        ).strip()
        # Extract JSON substring
        m = re.search(r"\{.*\}", gen, re.DOTALL)
        if not m:
            raise ValueError(f"Failed to parse JSON from model output: {gen!r}")
        raw_scores = json.loads(m.group(0))
        # Build vector in fixed trait order
        trait_order = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        vector = [raw_scores.get(tr, 0.0) for tr in trait_order]
        return {"personality_traits": raw_scores, "traits_vector": vector}


def annotate_traits(
    input_path: str = "aligned_data_with_emotions.json",
    output_path: str = "aligned_data_with_traits.json"
):
    agent = LlamaInferenceAgent()
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, entry in enumerate(data, start=1):
        try:
            result = agent.infer_traits(
                entry["transcript"],
                entry.get("response_time", ""),
                entry.get("body_language", ""),
                entry.get("speech_attributes", ""),
                entry.get("emotion", "")
            )
            entry["personality_traits"] = result["personality_traits"]
            entry["traits_vector"] = result["traits_vector"]
        except Exception as e:
            print(f"⚠ Record {idx} inference failed: {e}")
            entry["personality_traits"] = None
            entry["traits_vector"] = None
        if idx % 50 == 0:
            print(f"  • Processed {idx}/{len(data)} records")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved data with traits to '{output_path}'")

if _name_ == "_main_":
    annotate_traits()