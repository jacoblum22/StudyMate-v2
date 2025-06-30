import json

# Load current processed data to see structure
with open("processed/COGS 200 L1_processed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Current data keys:", list(data.keys()))
print("Topics keys:", list(data.get("topics", {}).keys()))

# Check a sample topic
if "topics" in data:
    sample_topic_id = next(iter(data["topics"].keys()))
    sample_topic = data["topics"][sample_topic_id]
    print(f"Sample topic ({sample_topic_id}) keys:", list(sample_topic.keys()))

    if "segment_positions" in sample_topic:
        print(
            f'Sample topic has {len(sample_topic["segment_positions"])} segment positions'
        )
    else:
        print("Sample topic missing segment_positions")

    if "examples" in sample_topic:
        print(f'Sample topic has {len(sample_topic["examples"])} examples')
    else:
        print("Sample topic missing examples")
