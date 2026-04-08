"""
RAG Chain — connects vector search with LLM to answer questions about ad campaigns.

What this module does:
- Takes a natural language question
- Retrieves relevant campaign data from ChromaDB (via indexer.py)
- Builds a prompt with the question + retrieved context
- Sends to LLM (Llama 3.1 via HuggingFace Inference API)
- Returns a human-readable answer with real data

How RAG chain works (the full picture):
1. User asks: "Why did facebook_tier1 performance drop?"
2. RETRIEVE: indexer finds the 5 most relevant campaign rows
3. AUGMENT: we build a prompt = system instructions + found data + question
4. GENERATE: LLM reads the data and writes a smart answer

Without RAG, the LLM would hallucinate (make up data).
With RAG, it answers based on YOUR actual campaign data.

Input:  natural language question (string)
Output: answer with real campaign data (string)
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

from src.rag.indexer import query_index

# ============================================================
# CONFIG
# ============================================================

# Load environment variables from .env file
load_dotenv()

# HuggingFace Inference API (new router format, OpenAI-compatible)
# We use Llama 3.1 8B Instruct — a free, high-quality open-source LLM
HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# System prompt — instructions for the LLM
# This tells the model HOW to answer: use data, be specific, etc.
SYSTEM_PROMPT = """You are an expert ad campaign analyst AI assistant.
You analyze marketing campaign performance data and provide actionable insights.

Rules:
- Base your answers ONLY on the provided campaign data below.
- Always include specific numbers (ROAS, CTR, CPC, etc.) from the data.
- If the data doesn't contain enough information to answer, say so honestly.
- Keep answers concise but insightful (3-5 sentences).
- When comparing campaigns, use concrete metrics.
- If a campaign is flagged as an ANOMALY, explain what might be wrong.
"""


# ============================================================
# PROMPT BUILDING
# ============================================================

def build_prompt(question, context_docs):
    """
    Build a list of messages for the chat API.

    The new HuggingFace API uses OpenAI-compatible format:
    a list of messages with roles (system, user, assistant).

    Parameters:
        question: user's natural language question
        context_docs: list of relevant documents from ChromaDB

    Returns:
        list of message dicts ready for the API
    """
    # Join all retrieved documents into one context block
    context = "\n\n---\n\n".join(context_docs)

    # Build user message with context + question
    user_message = f"""Here is the relevant campaign data:

{context}

Based on the data above, answer this question:
{question}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    return messages


# ============================================================
# LLM CALL
# ============================================================

def call_llm(messages, max_tokens=500):
    """
    Send messages to the HuggingFace router API and get a response.

    What is an Inference API?
    Instead of running a huge LLM on your laptop (needs expensive GPU),
    HuggingFace runs it on their servers. You send text, get text back.
    Free tier has rate limits but works fine for our project.

    The new API uses OpenAI-compatible chat format:
    - Send a list of messages (system + user)
    - Get back a response with choices[0].message.content

    Parameters:
        messages: list of message dicts with role and content
        max_tokens: maximum length of the response

    Returns:
        string: the LLM's response text
    """
    if not HF_TOKEN:
        return ("Error: HF_TOKEN not found in .env file. "
                "Please add your HuggingFace API token.")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,  # low = more factual, less creative
        "top_p": 0.9,
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            # OpenAI-compatible format: choices[0].message.content
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            return "No response generated."

        elif response.status_code == 503:
            # Model is loading (cold start)
            return ("The model is currently loading on HuggingFace servers. "
                    "Please try again in 30-60 seconds.")

        else:
            return f"API error {response.status_code}: {response.text}"

    except requests.exceptions.Timeout:
        return "Request timed out. The model might be loading. Try again."
    except requests.exceptions.ConnectionError:
        return "Connection error. Check your internet connection."


# ============================================================
# MAIN ASK FUNCTION
# ============================================================

def ask(question, top_k=5):
    """
    Full RAG pipeline: retrieve context → build prompt → generate answer.

    This is the main function that other modules will call.
    It's the "public API" of the RAG system.

    Parameters:
        question: natural language question about campaigns
        top_k: how many relevant documents to retrieve

    Returns:
        dict with:
        - answer: LLM-generated response string
        - sources: list of retrieved documents (for transparency)
        - metadatas: metadata of retrieved documents
    """
    print(f"\nQuestion: {question}")

    # Step 1: RETRIEVE — find relevant campaign data
    print("Searching for relevant data...")
    results = query_index(question, top_k=top_k)

    # Step 2: AUGMENT — build prompt with context
    messages = build_prompt(question, results["documents"])

    # Step 3: GENERATE — get LLM answer
    print("Generating answer...")
    answer = call_llm(messages)

    print(f"Answer: {answer[:200]}...")  # preview first 200 chars

    return {
        "answer": answer,
        "sources": results["documents"],
        "metadatas": results["metadatas"],
        "distances": results["distances"],
    }


# ============================================================
# MAIN
# ============================================================

def run_chain():
    """
    Test the RAG chain with sample questions.
    """
    print("=" * 60)
    print("  RAG CHAIN — Ad Campaign Intelligence")
    print("=" * 60)

    test_questions = [
        "Which campaign has the worst ROAS and why?",
        "Are there any anomalies in social media campaigns?",
        "Compare the performance of facebook and google campaigns.",
    ]

    for question in test_questions:
        print("\n" + "-" * 60)
        result = ask(question)
        print(f"\nFull answer:\n{result['answer']}")
        print(f"\nSources used: {len(result['sources'])} documents")
        print(f"Source campaigns: {[m['campaign_name'] for m in result['metadatas']]}")

    print("\n" + "*" * 60)
    print("  RAG Chain complete!")
    print("*" * 60)


if __name__ == "__main__":
    run_chain()