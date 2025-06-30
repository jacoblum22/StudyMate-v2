import json
from utils.bertopic_processor import process_with_bertopic

# Load existing chunks
with open("processed/COGS 200 L1_processed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = data["segments"]
print(f"Processing {len(chunks)} chunks...")

# Process without saving to file to test the basic structure
try:
    result = process_with_bertopic(chunks)
except Exception as e:
    print(f"Error during BERTopic processing: {e}")
    exit(1)
print(f"Result keys: {list(result.keys())}")
print(f'Number of segments in result: {len(result.get("segments", []))}')

# Check if topics have segment_positions
sample_topic_id = next(iter(result["topics"].keys()))
sample_topic = result["topics"][sample_topic_id]
print(f"Sample topic keys: {list(sample_topic.keys())}")
if "segment_positions" in sample_topic:
    print(
        f'Sample topic has {len(sample_topic["segment_positions"])} segment positions'
    )
    print(f'Sample topic has {len(sample_topic["examples"])} examples')
else:
    print("No segment_positions found in sample topic")
