import json
import torch
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict
from google import genai
from google.genai import types
import os

# --- Model Definition ---
class ALSModel(torch.nn.Module):
    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.slp = torch.nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.slp(x))

# --- Helpers ---
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def calculate_temporal(d1, d2):
    delta = abs(d1 - d2)
    return 1.0 / (1.0 + np.log1p(delta))

# --- Evaluation Pipeline ---

class DeepRecallEvaluator:
    def __init__(self, 
                 stories_path: str = "v1/data/evaluation_stories.json",
                 model_path: str = "v1/artifacts/pretrained/als_unified_linear.pt"):
        self.stories_path = Path(stories_path)
        self.model = ALSModel(input_dim=4)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        
        with open(self.stories_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)["stories"]
        
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def run_evaluation(self):
        results = []

        for story in self.data:
            story_id = story["story_id"]
            persona = story["persona"]
            events = story["events"]
            
            print(f"\n>>> Evaluating Story {story_id}: {persona[:50]}...")
            
            # 1. Select Anchor (Highest intensity event)
            anchor = max(events, key=lambda x: x["emotional_intensity"])
            print(f"    Anchor Event: {anchor['event'][:60]}... (I: {anchor['emotional_intensity']})")

            # 2. Retrieval: Standard RAG (Top 5 Semantic)
            candidates = [e for e in events if e["id"] != anchor["id"]]
            for c in candidates:
                c["sim"] = cosine_similarity(anchor["semantic_vec"], c["semantic_vec"])
            
            rag_top5 = sorted(candidates, key=lambda x: x["sim"], reverse=True)[:5]
            
            # 3. Retrieval: Deep Recall (ALS)
            # a. Get top 15 by semantic first
            top15_semantic = sorted(candidates, key=lambda x: x["sim"], reverse=True)[:15]
            
            # b. Re-rank by ALS
            for c in top15_semantic:
                s = (cosine_similarity(anchor["semantic_vec"], c["semantic_vec"]) + 1) / 2
                e = (cosine_similarity(anchor["emotional_vec"], c["emotional_vec"]) + 1) / 2
                t = calculate_temporal(anchor["d_days"], c["d_days"])
                i_val = c["emotional_intensity"]
                
                feat = torch.tensor([[s, e, t, i_val]], dtype=torch.float32)
                with torch.no_grad():
                    c["als_score"] = self.model(feat).item()
            
            als_top5 = sorted(top15_semantic, key=lambda x: x["als_score"], reverse=True)[:5]

            # 4. Analyze Distributions
            def get_dist(node_list):
                factual = len([n for n in node_list if n["category"] == "factual"])
                emotional = len([n for n in node_list if n["category"] == "emotional"])
                return {"factual": factual, "emotional": emotional}

            rag_dist = get_dist(rag_top5)
            als_dist = get_dist(als_top5)

            print(f"    RAG Dist: {rag_dist}")
            print(f"    ALS Dist: {als_dist}")

            # 5. LLM Response Generation
            print(f"    Generating LLM responses for judging...")
            context_rag = "\n".join([f"- {n['event']}" for n in rag_top5])
            context_als = "\n".join([f"- {n['event']}" for n in als_top5])

            prompt_rag = f"Persona: {persona}\nContext (Retrieved Memories):\n{context_rag}\n\nAnchor Event (Current Context): {anchor['event']}\n\nWrite a short internal monologue (3-4 sentences) showing how the persona processes this anchor event given their memories."
            prompt_als = f"Persona: {persona}\nContext (Retrieved Memories):\n{context_als}\n\nAnchor Event (Current Context): {anchor['event']}\n\nWrite a short internal monologue (3-4 sentences) showing how the persona processes this anchor event given their memories."
            
            # Simple retry loop for rate limits
            def safe_generate(prompt):
                for _ in range(3):
                    try:
                        return self.client.models.generate_content(model="gemini-2.5-flash", contents=prompt).text
                    except Exception as e:
                        print(f"      Error generating content, retrying... {e}")
                        time.sleep(2)
                return "Error generating response."

            resp_rag = safe_generate(prompt_rag)
            resp_als = safe_generate(prompt_als)

            results.append({
                "story_id": story_id,
                "persona": persona,
                "anchor": anchor["event"],
                "rag_dist": rag_dist,
                "als_dist": als_dist,
                "rag_context": context_rag,
                "als_context": context_als,
                "rag_response": resp_rag,
                "als_response": resp_als
            })
            
            time.sleep(1) # Extra safety for rate limits

        # Save for Judging
        output_csv = "v1/artifacts/predictions/rag_vs_deep_recall_results.csv"
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"\nEvaluation complete. Results saved to {output_csv}")

if __name__ == "__main__":
    evaluator = DeepRecallEvaluator()
    evaluator.run_evaluation()
