import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# The 10 CBD Techniques (Cognitive Behavioral Therapy inspired for grounding)
CBD_TECHNIQUES = [
    {
        "name": "Reframing",
        "description": "Identifying negative or catastrophic thoughts and replacing them with balanced, realistic alternatives."
    },
    {
        "name": "Grounding (5-4-3-2-1)",
        "description": "Engaging the senses to stay in the present moment. Observe 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste."
    },
    {
        "name": "Breathing (Box Breathing)",
        "description": "Inhale for 4 seconds, hold for 4 seconds, exhale for 4 seconds, hold for 4 seconds. Repeat to regulate the nervous system."
    },
    {
        "name": "Journaling",
        "description": "Writing down intrusive thoughts or feelings to externalize them, reducing their internal emotional weight."
    },
    {
        "name": "Thought Observation",
        "description": "Visualizing thoughts as leaves on a stream or clouds in the sky, observing them pass without judgment or attachment."
    },
    {
        "name": "Behavior Activation",
        "description": "Committing to small, manageable, positive actions (like drinking water or a 5-minute walk) to break the cycle of depression or anxiety."
    },
    {
        "name": "Self-compassion Prompts",
        "description": "Speaking to yourself with the same kindness and understanding you would offer a struggling friend."
    },
    {
        "name": "Distress Tolerance (TIPP)",
        "description": "Lowering emotional arousal through Temperature (cold water), Intense exercise, Paced breathing, or Paired muscle relaxation."
    },
    {
        "name": "Emotion Labeling",
        "description": "Simply naming the strong emotion out loud or in writing (e.g., 'I am feeling overwhelmed') to decrease the intensity of the feeling."
    },
    {
        "name": "Coping Statements",
        "description": "Using rehearsed, structured mantras like 'This feeling is uncomfortable, but it will pass,' or 'I have survived difficult times before.'"
    }
]

class CBDRetrievalLayer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the Retrieval-Augmented Generation layer for MindMate.
        Embeds the 10 CBD techniques into an ANN store (FAISS).
        """
        self.encoder = SentenceTransformer(model_name)
        # Prepare data
        self.documents = [f"{t['name']}: {t['description']}" for t in CBD_TECHNIQUES]
        
        # Build embeddings
        embeddings = self.encoder.encode(self.documents)
        self.dim = embeddings.shape[1]
        
        # Build FAISS Index
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(np.array(embeddings, dtype=np.float32))

    def retrieve(self, query: str, top_k: int = 2) -> list:
        """
        Takes sanitized user input and returns the Top-K most relevant CBD techniques to inject.
        """
        query_emb = self.encoder.encode([query]).astype(np.float32)
        distances, indices = self.index.search(query_emb, top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(CBD_TECHNIQUES[idx])
        return results

if __name__ == "__main__":
    cbd_store = CBDRetrievalLayer()
    q = "I can't stop my heart from racing, I'm panicking."
    print("User says:", q)
    print("Retrieving techniques...")
    res = cbd_store.retrieve(q)
    for r in res:
        print(f"- {r['name']}: {r['description']}")
