import os
from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

eval_data_dict = {
    "question": [
        "What RAM slots should I use for 2 DDR5 sticks on the motherboard?",
        "What does a solid yellow Q-LED light mean during POST?"
    ],
    "contexts": [
        ["Optimal 2-stick RAM configuration: Populate slots DIMM_A2 (slot 2) and DIMM_B2 (slot 4) first on ASUS X670E Hero."],
        ["DRAM LED (Yellow/Orange): Memory training failure, incompatible RAM, unseated sticks, or invalid slot order."]
    ],
    "answer": [
        "For a 2-stick configuration on the ASUS X670E Hero, populate slots DIMM_A2 (slot 2) and DIMM_B2 (slot 4) first.",
        "A solid yellow Q-LED indicates a DRAM memory training failure, unseated sticks, or incompatible RAM."
    ],
    "ground_truth": [
        "Populate slots DIMM_A2 and DIMM_B2 first.",
        "A yellow DRAM LED indicates a memory initialization error or bad RAM."
    ]
}

eval_dataset = Dataset.from_dict(eval_data_dict)

# 2. Evaluation Engines
eval_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Executing automated Ragas evaluation suite...")
results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevance, context_precision],
    llm=eval_llm,
    embeddings=eval_embeddings
)

# 3. Output Report
print("\n================== RAGAS EVALUATION METRICS REPORT ==================")
print(results.to_pandas()[["question", "faithfulness", "answer_relevance", "context_precision"]])