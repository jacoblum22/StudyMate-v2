import json
import sys

# Load the processed data
try:
    with open("processed/COGS 200 L1_processed.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: The file 'processed/COGS 200 L1_processed.json' was not found.")
    print("Please ensure the file exists and try again.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON in 'processed/COGS 200 L1_processed.json': {e}")
    print("Please check the file format and try again.")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error loading the file: {e}")
    sys.exit(1)
print("Looking for segment_positions in clusters:")
if "clusters" in data:
    for cluster in data["clusters"]:
        cluster_id = cluster.get("cluster_id")
        segment_positions = cluster.get("segment_positions", [])
        examples = cluster.get("examples", [])

        print(f'Cluster {cluster_id}: {cluster.get("heading", "Unknown")}')
        print(f"  - Examples: {len(examples)} chunks")
        print(f"  - Segment positions: {len(segment_positions)} chunks")
        print(
            f"  - Improvement: {len(segment_positions) - len(examples)} additional chunks"
        )
        print()
else:
    print("No clusters found in data")

print("\nData keys:", list(data.keys()))
print("Topics keys:", list(data["topics"].keys()) if "topics" in data else "No topics")
if "topics" in data:
    sample_topic = next(iter(data["topics"].values()))
    print("Sample topic keys:", list(sample_topic.keys()))
