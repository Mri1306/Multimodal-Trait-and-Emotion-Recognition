import json
import os
import re
import torch
from typing import Optional, List, Dict
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

class Llama1BEmotionClassifier:
    """
    Emotion classifier using Llama-3.2-1B-Instruct. Provides single-label inference
    compatible with DialogueAgent via infer_emotion method.
    """
    EMOTIONS = [
        "Happy", "Sad", "Angry", "Anxious", "Surprised",
        "Disgusted", "Confused", "Calm", "Excited", "Embarrassed", "Guilty", "Neutral"
    ]

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        hf_token: Optional[str] = None
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.hf_token = hf_token
        self._load_model()

    def _load_model(self):
        if self.hf_token:
            login(token=self.hf_token)
        print(f"🔄 Loading emotion model: {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()
        print("✅ Emotion model loaded.")

    def _create_prompt(self, transcript: str, response_time: str,
                       body_language: str, speech_attributes: str) -> str:
        return (
            'Classify the emotion as exactly one of: ' +
            ', '.join(self.EMOTIONS) +
            f".\nText: \"{transcript}\"\n"
            f"Response time: {response_time}\n"
            f"Body language: {body_language}\n"
            f"Speech: {speech_attributes}\nEmotion:"
        )

    def _llm_classify(self, sample: Dict[str, str]) -> Optional[str]:
        prompt = self._create_prompt(
            sample["transcript"], sample["response_time"],
            sample["body_language"], sample["speech_attributes"]
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        gen = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
        match = re.search(
            r"\b(" + "|".join(self.EMOTIONS) + r")\b", gen, re.IGNORECASE
        )
        return match.group(1).capitalize() if match else None

    def _rule_based_classify(self, transcript: str, response_time: str,
                              body_language: str, speech_attributes: str) -> str:
        t = transcript.lower()
        bl = body_language.lower()
        sa = speech_attributes.lower()
        rt = response_time.lower()
        scores = {e: 0 for e in self.EMOTIONS}
        # Keyword heuristics
        word_map = {
            "Happy": ["happy", "laugh", "smile"],
            "Sad": ["sad", "cry", "lonely"],
            "Angry": ["angry", "hate", "mad"],
            "Anxious": ["anxious", "worried", "nervous"],
            # ... add more as needed
        }
        for emo, kws in word_map.items():
            for kw in kws:
                if kw in t:
                    scores[emo] += 3
        # Simple body-language cues
        if any(x in bl for x in ["smile", "eye contact"]): scores["Happy"] += 2
        if any(x in bl for x in ["slumped", "hesitant"]): scores["Sad"] += 2
        # Speech & response-time cues
        if any(x in sa for x in ["loud", "raised"]): scores["Angry"] += 2
        if "fast" in rt: scores["Excited"] += 1
        # Fallback to neutral
        top = max(scores, key=scores.get)
        return top if scores[top] > 0 else "Neutral"

    def classify_batch(self, samples: List[Dict[str, str]]) -> List[str]:
        results = []
        for samp in samples:
            emo = self._llm_classify(samp) or self._rule_based_classify(
                samp["transcript"], samp["response_time"],
                samp["body_language"], samp["speech_attributes"]
            )
            results.append(emo)
        return results

    def classify_single(self, transcript: str, response_time: str,
                        body_language: str, speech_attributes: str) -> str:
        return self.classify_batch([{"transcript": transcript,
                                     "response_time": response_time,
                                     "body_language": body_language,
                                     "speech_attributes": speech_attributes}])[0]

    # Alias for DialogueAgent compatibility
    infer_emotion = classify_single


def annotate_emotions(
    input_path: str = "aligned_data_normalized.json",
    output_path: str = "aligned_data_with_emotions.json"
):
    classifier = Llama1BEmotionClassifier(hf_token=os.getenv("HUGGINGFACE_TOKEN"))
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples, idxs = [], []
    for i, ent in enumerate(data):
        if all(k in ent for k in ["transcript", "response_time", "body_language", "speech_attributes"]):
            samples.append(ent)
            idxs.append(i)

    emotions = classifier.classify_batch(samples)
    for i, emo in zip(idxs, emotions):
        data[i]["emotion"] = emo
    for ent in data:
        ent.setdefault("emotion", "Neutral")

    counts = Counter(ent["emotion"] for ent in data)
    print("Emotion counts:", counts)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved emotions to {output_path}")

if __name__ == "__main__":
    annotate_emotions()