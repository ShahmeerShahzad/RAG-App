import os
from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Golden Evaluation Dataset for n8n Knowledge Base
test_dataset = {
    "question": [
        "How does pairedItem linking work in n8n?",
        "What is the difference between Test and Production Webhook URLs in n8n?",
        "How do you attach and use a Global Error Workflow?"
    ],
    "contexts": [
        ["Paired Item Context: n8n maintains an internal metadata link called pairedItem connecting every transformed output record back to its exact origin record in preceding nodes. Syntax $('Node Name').item.json.field uses paired item linking to resolve parent attributes."],
        ["Test URLs listen only during manual development when 'Listen for test event' is active and provide verbose debug data. Production URLs run continuously in the background when the workflow toggle is set to Active."],
        ["Global Error Workflows: Attached via Workflow Settings. Any unhandled exception halts the parent execution and triggers the Error Workflow, passing execution.id, workflow.id, and node.name to send alerts."]
    ],
    "answer": [
        "PairedItem linking maintains an internal metadata pointer mapping transformed output items back to their origin record in preceding nodes.",
        "Test webhook URLs only listen during active canvas testing and return debug data, while production URLs run continuously 24/7 when the workflow is activated.",
        "You attach an Error Workflow in Workflow Settings; it receives the execution ID, workflow ID, and offending node name to automate alerting when an error occurs."
    ],
    "ground_truth": [
        "pairedItem maps transformed records back to their exact origin record in preceding nodes.",
        "Test URLs work only during active canvas listening, while production URLs listen continuously when active.",
        "Attached in Workflow Settings to receive execution metadata when an unhandled error occurs."
    ]
}

eval_data = Dataset.from_dict(test_dataset)
eval_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Executing automated Ragas benchmark across 3 core metrics...")
results = evaluate(
    dataset=eval_data,
    metrics=[faithfulness, answer_relevance, context_precision],
    llm=eval_llm,
    embeddings=eval_embeddings
)

print("\n" + "=" * 65)
print("             RAGAS EVALUATION METRIC REPORT")
print("=" * 65)
df = results.to_pandas()
print(df[["question", "faithfulness", "answer_relevance", "context_precision"]].to_string(index=False))

df.to_csv("rag_evaluation_results.csv", index=False)
print("\n✅ Results exported to 'rag_evaluation_results.csv'")