import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from typing import List, Dict, Tuple
from .openai_client import get_openai_client

load_dotenv()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

client = get_openai_client()


def generate_cluster_headings(
    clusters: List[List[str]],
) -> Tuple[List[Dict[str, str]], int]:
    """
    Generate headings for multiple clusters in a single prompt to ensure global context and cohesion.

    Args:
        clusters: List of clusters, where each cluster is a list of text chunks

    Returns:
        tuple[list[str], int]: List of generated headings and total token count
    """
    if not clusters:
        return [{"concept": "Unknown", "heading": "Untitled Topic", "summary": ""}], 0

    # For each cluster, get the most representative chunks
    selected_chunks_by_cluster = []
    for cluster in clusters:
        # Embed all chunks in this cluster
        embeddings = embedding_model.encode(cluster)

        # Calculate centroid of the cluster
        centroid = np.mean(embeddings, axis=0, keepdims=True)

        # Rank chunks by similarity to centroid
        sims = cosine_similarity(embeddings, centroid).flatten()
        top_indices = sims.argsort()[::-1][
            : min(3, len(cluster))
        ]  # Reduced to 3 chunks per cluster

        # Get top representative chunks
        selected_chunks = [cluster[i] for i in top_indices]
        selected_chunks_by_cluster.append(selected_chunks)

    # Build the prompt
    prompt = (
        "You are generating section headings for a lecture-based study outline.\n"
        "Below are multiple clusters of transcript content. Each represents a different topic from the lecture.\n\n"
        "For each cluster:\n"
        "1. First, identify 1-3 distinct key concepts that best represent the cluster's content\n"
        "2. Then, create a specific and concise heading (under 12 words) based on those concepts\n\n"
        "Guidelines:\n"
        "- Each cluster must focus on a different concept than the others\n"
        "- Use clear and informative academic language\n"
        "- Avoid vague terms like 'Understanding' or 'Overview'\n"
        "- Focus on specific concepts, processes, methods, or theories\n"
        "- Ensure headings are distinct and non-overlapping\n\n"
    )  # Add the content for each cluster
    for i, chunks in enumerate(selected_chunks_by_cluster, 1):
        prompt += f"Section {i}:\n"
        prompt += "\n".join(chunks)
        prompt += "\n\n"

    prompt += (
        "For each section above:\n"
        "1. First identify 1-3 key concepts that best represent the content\n"
        "2. Then generate a heading based on those concepts (12 words max)\n"
        "3. Then write a 1–3 sentence summary of the topic, suitable for a student reading study notes\n"
        "Format your response as:\n"
        "Concept: [concept1, concept2, concept3] (separate multiple concepts with commas)\n"
        "Heading: [heading]\n"
        "Summary: [1–3 sentence summary]\n"
        "---\n"
        "Separate each section's response with '|||'. Respond only with the formatted sections above, nothing else.\n"
    )

    # Count tokens
    token_count = len(encoding.encode(prompt))

    # GPT call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000,  # Increased to accommodate longer responses for multiple headings
    )

    # Ensure response and content exist
    if not response.choices or not response.choices[0].message.content:
        print("Warning: GPT response is empty or malformed.")
        return ([{"concept": "", "heading": "", "summary": ""}], token_count)

    raw_content = response.choices[0].message.content.strip()
    sections = raw_content.split("|||")

    parsed = []
    for section in sections:
        lines = section.strip().splitlines()
        concept = ""
        heading = ""
        summary = ""

        for line in lines:
            line = line.strip()
            if line.lower().startswith("concept:"):
                concept = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line.lower().startswith("heading:"):
                heading = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip() if ":" in line else ""

        # Validate and append only non-empty entries
        if concept or heading or summary:
            parsed.append({"concept": concept, "heading": heading, "summary": summary})

    # Ensure it matches cluster count
    while len(parsed) < len(clusters):
        parsed.append({"concept": "", "heading": "Untitled Topic", "summary": ""})

    return parsed, token_count
