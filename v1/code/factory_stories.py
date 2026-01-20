import os
import json
import time
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from google import genai
from google.genai import types

# --- Data Models ---

class EventNode(BaseModel):
    id: str
    event: str
    d_days: float
    emotional_state: str
    emotional_intensity: float
    category: str # "factual" or "emotional"
    semantic_vec: Optional[List[float]] = None
    emotional_vec: Optional[List[float]] = None

class Story(BaseModel):
    story_id: int
    persona: str
    events: List[EventNode]

class StoryDataset(BaseModel):
    stories: List[Story]

# --- Generator ---

class StoryFactory:
    def __init__(self, output_path: str = "v1/data/evaluation_stories.json"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        self.client = genai.Client(api_key=api_key)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_model = "text-embedding-004"

    def generate_stories(self, num_stories: int = 10):
        personas = [
            "A retired lighthouse keeper living alone on a stormy coast.",
            "An ambitious young architect designing a skyscraper in a crowded city.",
            "A high school teacher who was once a professional jazz musician.",
            "A dedicated environmental scientist working in the Amazon rainforest.",
            "A former investigative journalist now writing culinary reviews.",
            "A space technician maintaining a vessel on a long-haul journey to Mars.",
            "A competitive gardener preparing for the national championships.",
            "A history professor obsessed with the unsolved mysteries of the 19th century.",
            "A professional mediator who struggles to resolve conflicts in their own family.",
            "A talented glassblower in a small Italian village."
        ]

        all_stories = []

        for i, persona in enumerate(personas):
            print(f">>> [Story {i+1}/10] Generating for: {persona}")
            
            sys_instr = f"""
Persona: {persona}
Task: Generate a realistic episodic memory dataset for this persona.

CONSTRAINTS:
1. Total Events: 20
2. Split: 15 Factual, 5 Emotional.
3. Category Field: You MUST use the exact string "factual" or "emotional" for the 'category' field.
4. Factual Events: Mundane, day-to-day tasks (Intensity: 0.05-0.20).
5. Emotional Events: High or low stakes, high resonance (Intensity: 0.70-1.0).
5. d_days: 
   - 10 Factual events should be within the last 7 days (d_days between -7.0 and 0.0).
   - 5 Factual events can be 1-6 months ago (d_days between -180.0 and -30.0).
   - 5 Emotional events should be distant/varied past (d_days between -3000.0 and -30.0).

Output in JSON format matching the schema for a 'Story' object.
"""

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Generate the 20 events for story {i+1}.",
                config=types.GenerateContentConfig(
                    system_instruction=sys_instr,
                    response_mime_type="application/json",
                    response_schema=Story.model_json_schema(),
                )
            )
            
            story_data = json.loads(response.text)
            story_data["story_id"] = i + 1
            story_data["persona"] = persona
            
            story = Story(**story_data)
            
            # --- Embeddings ---
            print(f"    Embedding {len(story.events)} events...")
            
            semantic_texts = [e.event for e in story.events]
            emotional_texts = [e.emotional_state for e in story.events]
            
            # Semantic
            res_sem = self.client.models.embed_content(
                model=self.embedding_model,
                contents=semantic_texts,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            # Emotional
            res_emo = self.client.models.embed_content(
                model=self.embedding_model,
                contents=emotional_texts,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            
            for j, event in enumerate(story.events):
                event.semantic_vec = res_sem.embeddings[j].values
                event.emotional_vec = res_emo.embeddings[j].values
            
            all_stories.append(story)
            time.sleep(1) # Rate limit safety

        dataset = StoryDataset(stories=all_stories)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(dataset.model_dump_json(indent=4))
        print(f"\nSuccessfully saved {len(all_stories)} stories to {self.output_path}")

if __name__ == "__main__":
    factory = StoryFactory()
    factory.generate_stories(10)
