"""
Utility function to extract all text chunks for a topic by cross-referencing
segment positions with the segments array.
"""


def get_all_topic_chunks(processed_data: dict, topic_id: str) -> list:
    """
    Extract all text chunks for a specific topic by cross-referencing
    segment_positions with the segments array.

    Args:
        processed_data (dict): The processed JSON data containing segments and topics
        topic_id (str): The ID of the topic to extract chunks for

    Returns:
        list: List of text chunks for the topic
    """
    try:
        print(f"\n🔍 get_all_topic_chunks called for topic {topic_id}")

        # Get the topic data
        if "topics" not in processed_data or topic_id not in processed_data["topics"]:
            print(f"❌ Topic {topic_id} not found in processed data")
            print(f"Available topics: {list(processed_data.get('topics', {}).keys())}")
            return []

        topic = processed_data["topics"][topic_id]
        print(f"✅ Found topic: {topic.get('heading', 'Unknown heading')}")

        # Get segment positions for this topic
        if "segment_positions" not in topic:
            print(f"❌ No segment_positions found for topic {topic_id}")
            print(f"Available topic keys: {list(topic.keys())}")
            print(
                f"📝 Falling back to examples: {len(topic.get('examples', []))} available"
            )
            return topic.get("examples", [])

        segment_positions = topic["segment_positions"]
        examples_count = len(topic.get("examples", []))
        print(
            f"📊 Topic has {len(segment_positions)} segment positions vs {examples_count} examples"
        )

        # Get all segments
        if "segments" not in processed_data:
            print("❌ No segments found in processed data")
            print(f"Available data keys: {list(processed_data.keys())}")
            return topic.get("examples", [])

        segments = processed_data["segments"]
        print(f"📦 Total segments available: {len(segments)}")

        # Create a lookup dictionary for faster access
        segments_by_position = {seg["position"]: seg["text"] for seg in segments}
        print(f"🗂️ Created lookup map with {len(segments_by_position)} positions")

        # Extract chunks for this topic
        topic_chunks = []
        missing_positions = []

        for i, position in enumerate(segment_positions):
            if position in segments_by_position:
                chunk_text = segments_by_position[position]
                topic_chunks.append(chunk_text)
                if i < 3:  # Log first 3 for debugging
                    print(f"  ✅ Position {position}: {chunk_text[:60]}...")
            else:
                missing_positions.append(position)

        if missing_positions:
            print(
                f"⚠️ Warning: {len(missing_positions)} positions not found: {missing_positions[:5]}..."
            )

        improvement = len(topic_chunks) - examples_count
        print(
            f"🎯 Successfully extracted {len(topic_chunks)} chunks for topic {topic_id}"
        )
        print(
            f"📈 Improvement: +{improvement} chunks over examples ({examples_count} -> {len(topic_chunks)})"
        )

        return topic_chunks

    except Exception as e:
        print(f"💥 Error extracting chunks for topic {topic_id}: {e}")
        import traceback

        traceback.print_exc()
        return []


def get_all_topic_chunks_from_file(filename: str, topic_id: str) -> list:
    """
    Load processed data from file and extract all chunks for a topic.

    Args:
        filename (str): The filename of the processed data (without .json extension)
        topic_id (str): The ID of the topic to extract chunks for

    Returns:
        list: List of text chunks for the topic
    """
    import json
    import os

    try:
        # Construct the file path
        processed_file = os.path.join("processed", f"{filename}_processed.json")

        if not os.path.exists(processed_file):
            print(f"Processed file not found: {processed_file}")
            return []

        # Load the processed data
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_data = json.load(f)

        return get_all_topic_chunks(processed_data, topic_id)

    except Exception as e:
        print(f"Error loading processed data from {filename}: {e}")
        return []
