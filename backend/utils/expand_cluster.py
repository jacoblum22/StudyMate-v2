import json
import os
from dotenv import load_dotenv
from .openai_client import get_openai_client

load_dotenv()

client = get_openai_client()


def expand_cluster(filename: str, cluster_id: str) -> dict:
    """
    Expand a cluster by generating a bullet point list of important points.

    Args:
        filename (str): The name of the file in the processed folder.
        cluster_id (str): The ID of the cluster to expand.

    Returns:
        dict: A dictionary with the following structure:
            {
                "message": str,  # Status message
                "cluster": dict  # Updated cluster metadata, including "bullet_points"
            }
    """
    processed_file = os.path.join("processed", filename)

    print(
        f"Received request to expand cluster: filename={filename}, cluster_id={cluster_id}"
    )

    if not os.path.exists(processed_file):
        print(f"Error: File {filename} not found in processed folder.")
        return {"error": f"File {filename} not found in processed folder."}

    with open(processed_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    clusters = data.get("clusters", [])
    cluster = None
    for c in clusters:
        if str(c.get("cluster_id")) == str(cluster_id):
            cluster = c
            break

    if not cluster:
        print(f"Error: Cluster ID {cluster_id} not found in file {filename}.")
        return {"error": f"Cluster ID {cluster_id} not found in file {filename}."}

    print(f"Cluster data before expansion: {cluster}")

    # Check and log the presence of the segment_positions field
    if "segment_positions" not in cluster:
        print(f"Warning: 'segment_positions' field is missing in cluster {cluster_id}.")
    else:
        print(
            f"'segment_positions' field found in cluster {cluster_id}: {cluster['segment_positions']}"
        )

    # Retrieve segment texts based on segment_positions
    segment_positions = cluster.get("segment_positions", [])
    segments = data.get("segments", [])

    print(f"Segment positions for cluster {cluster_id}: {segment_positions}")
    print(f"Total number of segments available: {len(segments)}")

    # Extract the text of the segments at the specified positions
    chunks = []
    for pos in segment_positions:
        matching_segment = next(
            (segment for segment in segments if segment.get("position") == pos), None
        )
        if matching_segment:
            chunks.append(matching_segment["text"])
            print(
                f"Successfully retrieved segment at position {pos}: {matching_segment['text']}"
            )
        else:
            print(f"No matching segment found for position: {pos}")

    if not chunks:
        print(f"Error: No valid segments found for Cluster ID {cluster_id}.")
        return {"error": f"No valid segments found for Cluster ID {cluster_id}."}

    print(f"Extracted text chunks for GPT-4: {chunks}")

    # Build the prompt for GPT-4
    prompt = (
        "You are a helpful assistant generating concise bullet points for study notes.\n"
        "Below is a cluster of text chunks. Each chunk represents a key idea or example.\n"
        "Generate a list of 5-15 concise bullet points summarizing the most important ideas from the cluster.\n"
        "- Each bullet should start with a bolded 'mini-header' that is ALWAYS phrased as a question, followed by a question mark.\n"
        "- Examples: '**What is photosynthesis?**', '**How does neural transmission work?**', '**Why is this concept important?**'\n"
        "- The question should act as a semantic anchor that students can use for self-testing.\n"
        "- After the question, provide a clear and concise answer.\n"
        "- Keep each bullet to a single idea.\n"
        "- Use clear and concise language.\n"
        "- Focus on key takeaways and avoid redundancy.\n"
        "- Ensure the bullet points are suitable for study notes and self-testing.\n\n"
    )
    prompt += "\n".join(chunks)

    print(f"Generated GPT-4 prompt: {prompt}")

    # GPT-4o call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,  # Adjust as needed for bullet points
    )

    # Ensure response and content exist
    if not response.choices or not response.choices[0].message.content:
        print("Error: GPT-4 response is empty or malformed.")
        return {"error": "GPT-4 response is empty or malformed."}

    raw_bullet_points = response.choices[0].message.content.strip()
    print(f"Raw GPT-4 response: {raw_bullet_points}")

    # Parse the bullet points from the response
    bullet_points = [
        point.strip() for point in raw_bullet_points.split("\n") if point.strip()
    ]

    print(f"Parsed bullet points: {bullet_points}")

    # Update cluster metadata
    cluster["bullet_points"] = bullet_points

    print(f"Updated cluster data: {cluster}")
    print(f"Saving updated data to file: {processed_file}")

    # Save updated data back to the file
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(
        "Backend Response:",
        {"message": "Cluster expanded successfully.", "cluster": cluster},
    )

    return {"message": "Cluster expanded successfully.", "cluster": cluster}
