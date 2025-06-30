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

print("Data structure analysis:")
print(f'Total segments: {len(data["segments"])}')
print(f'Total topics: {len(data["topics"])}')
print()

for topic_id, topic in data["topics"].items():
    segment_positions = topic.get("segment_positions", [])
    examples = topic.get("examples", [])

    print(f'Topic {topic_id}: {topic["heading"]}')
    print(f"  - Examples (current): {len(examples)} chunks")
    print(f"  - Segment positions: {len(segment_positions)} chunks")
    print(
        f"  - Improvement: {len(segment_positions) - len(examples)} additional chunks"
    )
    print()
