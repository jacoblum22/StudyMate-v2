import os
import json
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import statistics
from typing import List, Dict, Any, cast, Optional, Tuple
from .generate_cluster_heading import generate_cluster_headings

# Handle NLTK stopwords with graceful fallback
try:
    from nltk.corpus import stopwords
    import nltk

    # Try to access stopwords, download if needed
    try:
        stopwords.words("english")
    except LookupError:
        print("Downloading NLTK stopwords corpus...")
        nltk.download("stopwords", quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: NLTK not available, using sklearn's built-in stopwords")
    NLTK_AVAILABLE = False

PROCESSED_DIR = "processed"  # Folder to save processed JSON files


def get_stopwords():
    """
    Get English stopwords with fallback options.

    Returns:
        List of stopwords or 'english' string for sklearn built-in stopwords
    """
    if NLTK_AVAILABLE:
        try:
            return stopwords.words("english")
        except LookupError:
            print(
                "Warning: NLTK stopwords not available, using sklearn's built-in stopwords"
            )
            return "english"
    else:
        return "english"


def pre_cluster_with_kmeans(
    chunks: List[Dict[str, str]],
    min_cluster_size: int = 10,
    min_words_per_cluster: int = 500,
) -> List[List[Dict[str, str]]]:
    """
    Use k-means clustering to split chunks into 2 clusters before running BERTopic.
    Only performs clustering if each resulting cluster would have enough content.

    Args:
        chunks: List of chunk dictionaries with 'position' and 'text' keys
        min_cluster_size: Minimum number of chunks required per cluster
        min_words_per_cluster: Minimum number of words required per cluster

    Returns:
        List of chunk clusters. If clustering criteria aren't met, returns [chunks] (single cluster)
    """
    # Check if we have enough chunks to split
    if len(chunks) < min_cluster_size * 2:
        print(
            f"Not enough chunks ({len(chunks)}) for k-means pre-clustering. Need at least {min_cluster_size * 2}."
        )
        return [chunks]

    # Check if we have enough total words
    total_words = sum(len(chunk["text"].split()) for chunk in chunks)
    if total_words < min_words_per_cluster * 2:
        print(
            f"Not enough words ({total_words}) for k-means pre-clustering. Need at least {min_words_per_cluster * 2}."
        )
        return [chunks]

    print(
        f"\nPerforming k-means pre-clustering on {len(chunks)} chunks..."
    )  # Extract texts for vectorization
    texts = [chunk["text"] for chunk in chunks]

    # Use TF-IDF vectorization for k-means
    stopword_list = get_stopwords()
    tfidf_vectorizer = TfidfVectorizer(
        stop_words=stopword_list,
        max_features=1000,  # Limit features for efficiency
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 2),
    )

    try:
        # Vectorize texts
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        # Perform k-means clustering with k=2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        # Group chunks by cluster
        cluster_0 = []
        cluster_1 = []

        for chunk, label in zip(chunks, cluster_labels):
            if label == 0:
                cluster_0.append(chunk)
            else:
                cluster_1.append(chunk)

        # Calculate cluster statistics
        cluster_0_words = sum(len(chunk["text"].split()) for chunk in cluster_0)
        cluster_1_words = sum(len(chunk["text"].split()) for chunk in cluster_1)

        print(f"K-means clustering results:")
        print(f"  Cluster 0: {len(cluster_0)} chunks, {cluster_0_words} words")
        print(f"  Cluster 1: {len(cluster_1)} chunks, {cluster_1_words} words")

        # Check if both clusters meet minimum requirements
        if (
            len(cluster_0) >= min_cluster_size
            and cluster_0_words >= min_words_per_cluster
            and len(cluster_1) >= min_cluster_size
            and cluster_1_words >= min_words_per_cluster
        ):

            print(
                "Both clusters meet minimum requirements. Proceeding with split clustering."
            )
            return [cluster_0, cluster_1]
        else:
            print(
                "One or both clusters don't meet minimum requirements. Using single cluster."
            )
            return [chunks]

    except Exception as e:
        print(f"K-means clustering failed: {e}. Using single cluster.")
        return [chunks]


def process_cluster_with_bertopic(
    cluster_chunks: List[Dict[str, str]], cluster_id: int
) -> Tuple[Dict[str, List[Dict[str, str]]], List[Dict[str, str]], Any]:
    """
    Process a single cluster of chunks with BERTopic.

    Args:
        cluster_chunks: List of chunks in this cluster
        cluster_id: ID for this cluster (for logging)

    Returns:
        Tuple of (topic_map, noise_chunks, all_chunks)
    """
    print(f"\n--- Processing Cluster {cluster_id} ---")
    print(f"Chunks in cluster: {len(cluster_chunks)}")

    # Configure vectorizer
    stopword_list = get_stopwords()
    texts = [chunk["text"] for chunk in cluster_chunks]

    # Calculate dynamic min_topic_size based on cluster size
    # Aim for ~1/5 of chunks, with min=2 and max=10
    dynamic_min_topic_size = max(2, min(10, len(cluster_chunks) // 5))
    print(
        f"Using min_topic_size: {dynamic_min_topic_size} (based on {len(cluster_chunks)} chunks)"
    )

    # Try with original parameters first
    try:
        vectorizer_model = CountVectorizer(
            stop_words=stopword_list,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),  # Include both unigrams and bigrams
        )

        # Configure BERTopic
        topic_model = BERTopic(
            vectorizer_model=vectorizer_model,
            min_topic_size=dynamic_min_topic_size,  # Dynamic minimum based on cluster size
            nr_topics="auto",  # Automatically determine number of topics
            calculate_probabilities=True,
            verbose=True,
        )

        # Fit the model
        topics, probs = topic_model.fit_transform(texts)

    except ValueError as e:
        if "max_df corresponds to < documents than min_df" in str(e):
            print(f"Falling back to lenient parameters for cluster {cluster_id}...")
            # Fall back to lenient parameters
            vectorizer_model = CountVectorizer(
                stop_words=stopword_list,
                min_df=1,  # More lenient minimum document frequency
                max_df=0.95,
                ngram_range=(1, 2),
            )

            topic_model = BERTopic(
                vectorizer_model=vectorizer_model,
                min_topic_size=2,
                nr_topics="auto",
                calculate_probabilities=True,
                verbose=True,
            )

            topics, probs = topic_model.fit_transform(texts)
        else:
            # Re-raise if it's a different error
            raise

    # Extract topic-to-chunks mapping
    topic_map = {}
    noise_chunks = []  # Track chunks that weren't assigned to any topic
    for chunk, topic_id in zip(cluster_chunks, topics):
        if topic_id == -1:
            noise_chunks.append(chunk)  # Store noise chunks separately
        else:
            # Offset topic IDs by cluster to avoid conflicts
            adjusted_topic_id = topic_id + (
                cluster_id * 1000
            )  # Large offset to avoid conflicts
            topic_map.setdefault(adjusted_topic_id, []).append(
                chunk
            )  # Print topic assignment stats for this cluster
    print(f"Cluster {cluster_id} topic assignment stats:")
    print(f"  Number of topics found: {len(topic_map)}")
    print(f"  Number of noise chunks: {len(noise_chunks)}")
    print(
        f"  Number of chunks assigned to topics: {sum(len(chunks) for chunks in topic_map.values())}"
    )

    # Redistribute overly large topics
    if len(topic_map) > 1:  # Only redistribute if we have multiple topics
        topic_map = redistribute_large_topics(topic_map, topic_model, cluster_id)

    return topic_map, noise_chunks, topic_model


def _print_initial_statistics(chunks: List[Dict[str, str]]) -> int:
    """
    Print initial statistics about the chunks to be processed.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Total word count across all chunks
    """
    total_words = sum(len(chunk["text"].split()) for chunk in chunks)
    print(f"\nInitial stats:")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Total words: {total_words}")
    print(f"Average words per chunk: {total_words/len(chunks):.1f}")
    return total_words


def _process_clusters_with_bertopic(
    clusters: List[List[Dict[str, str]]],
) -> Tuple[
    Dict[str, List[Dict[str, str]]], List[Dict[str, str]], List[Tuple[Any, int]]
]:
    """
    Process each cluster with BERTopic and collect results.

    Args:
        clusters: List of chunk clusters

    Returns:
        Tuple of (all_topic_maps, all_noise_chunks, all_topic_models)
    """
    all_topic_maps = {}
    all_noise_chunks = []
    all_topic_models = []
    cluster_topic_counts = []  # Track topics per cluster

    for cluster_idx, cluster_chunks in enumerate(clusters):
        topic_map, noise_chunks, topic_model = process_cluster_with_bertopic(
            cluster_chunks, cluster_idx
        )

        # Count topics in this cluster
        cluster_topics = len(
            [
                tid
                for tid in topic_map.keys()
                if int(tid) >= cluster_idx * 1000
                and int(tid) < (cluster_idx + 1) * 1000
            ]
        )
        cluster_topic_counts.append(cluster_topics)
        print(f"📊 Group {cluster_idx} → Generated {cluster_topics} topics")

        all_topic_maps.update(topic_map)
        all_noise_chunks.extend(noise_chunks)
        all_topic_models.append((topic_model, cluster_idx))

    # Print detailed clustering summary
    print(f"\n🎯 Clustering Summary:")
    print(f"K-means groups: {len(clusters)}")
    for i, count in enumerate(cluster_topic_counts):
        print(f"  Group {i}: {count} topics")
    print(f"Total topics across all groups: {sum(cluster_topic_counts)}")

    return all_topic_maps, all_noise_chunks, all_topic_models


def _print_overall_results(
    clusters: List[List[Dict[str, str]]],
    all_topic_maps: Dict[str, List[Dict[str, str]]],
    all_noise_chunks: List[Dict[str, str]],
) -> Tuple[int, int]:
    """
    Print overall processing results and word count statistics.

    Args:
        clusters: List of chunk clusters
        all_topic_maps: Dictionary mapping topic IDs to lists of chunks
        all_noise_chunks: List of chunks not assigned to any topic

    Returns:
        Tuple of (noise_words, topic_words)
    """
    total_topics = len(all_topic_maps)
    total_noise = len(all_noise_chunks)
    total_assigned = sum(len(chunks) for chunks in all_topic_maps.values())

    print(f"\n=== Overall Results ===")
    print(f"Total clusters processed: {len(clusters)}")
    print(f"Total topics found: {total_topics}")
    print(f"Total noise chunks: {total_noise}")
    print(f"Total chunks assigned to topics: {total_assigned}")

    # Print word counts for noise vs topic chunks
    noise_words = sum(len(chunk["text"].split()) for chunk in all_noise_chunks)
    topic_words = sum(
        len(chunk["text"].split())
        for chunks in all_topic_maps.values()
        for chunk in chunks
    )
    print(f"Words in noise chunks: {noise_words}")
    print(f"Words in topic chunks: {topic_words}")
    print(f"Total words processed: {noise_words + topic_words}")

    return noise_words, topic_words


def _generate_topic_headings(
    all_topic_maps: Dict[str, List[Dict[str, str]]],
) -> Tuple[List[str], List[List[Dict[str, str]]], List[Dict[str, str]], int]:
    """
    Generate topic headings and organize topic data.

    Args:
        all_topic_maps: Dictionary mapping topic IDs to lists of chunks

    Returns:
        Tuple of (ordered_topic_ids, topic_chunks_list, cluster_info, total_tokens)
    """
    # Get all topics in chronological order
    ordered_topic_ids = order_topics_chronologically(all_topic_maps)
    topic_chunks_list = [all_topic_maps[tid] for tid in ordered_topic_ids]

    headings_data: List[Dict[str, str]]
    headings_data, total_tokens = generate_cluster_headings(
        [[chunk["text"] for chunk in cluster] for cluster in topic_chunks_list]
    )

    # Parse concepts and headings from the response
    cluster_info = []
    for i in range(len(topic_chunks_list)):
        # Handle concepts that might have brackets or other formatting
        concepts_raw = headings_data[i]["concept"]
        # Remove brackets and split by comma
        concepts_clean = concepts_raw.strip("[]").replace("[", "").replace("]", "")
        concepts_list = [c.strip() for c in concepts_clean.split(",") if c.strip()]

        cluster_info.append(
            {
                "concepts": concepts_list,
                "heading": headings_data[i]["heading"],
                "summary": headings_data[i]["summary"],
            }
        )

    return ordered_topic_ids, topic_chunks_list, cluster_info, total_tokens


def _calculate_topic_statistics(
    ordered_topic_ids: List[str], all_topic_maps: Dict[str, List[Dict[str, str]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate and print statistics for each topic.

    Args:
        ordered_topic_ids: List of topic IDs in chronological order
        all_topic_maps: Dictionary mapping topic IDs to lists of chunks

    Returns:
        Dictionary of topic statistics
    """
    topic_stats = {}
    for i, tid in enumerate(ordered_topic_ids):
        topic_chunks = all_topic_maps[tid]
        chunk_sizes = [len(chunk["text"].split()) for chunk in topic_chunks]
        topic_stats[tid] = {
            "num_chunks": len(topic_chunks),
            "min_size": min(chunk_sizes),
            "mean_size": round(statistics.mean(chunk_sizes), 1),
            "max_size": max(chunk_sizes),
        }

    # Print topic statistics
    print("\nTopic statistics:")
    for tid, stats in topic_stats.items():
        print(f"Topic {tid}:")
        print(f"  Number of chunks: {stats['num_chunks']}")
        print(f"  Word count range: {stats['min_size']} - {stats['max_size']}")
        print(f"  Average words per chunk: {stats['mean_size']}")

    return topic_stats


def _build_result_topics(
    ordered_topic_ids: List[str],
    all_topic_maps: Dict[str, List[Dict[str, str]]],
    all_topic_models: List[Tuple[Any, int]],
    cluster_info: List[Dict[str, str]],
    topic_stats: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Build the result topics dictionary with all topic information.

    Args:
        ordered_topic_ids: List of topic IDs in chronological order
        all_topic_maps: Dictionary mapping topic IDs to lists of chunks
        all_topic_models: List of topic models and their cluster IDs
        cluster_info: List of cluster information (concepts, headings, summaries)
        topic_stats: Dictionary of topic statistics

    Returns:
        Dictionary of result topics
    """
    result_topics = {}

    # Build result topics from all clusters
    for i, tid in enumerate(ordered_topic_ids):
        # Find which topic model this topic belongs to
        topic_model = None
        original_tid = tid

        # Check which cluster this topic came from based on the offset
        for tm, cluster_idx in all_topic_models:
            cluster_offset = cluster_idx * 1000
            if int(tid) >= cluster_offset and int(tid) < (cluster_offset + 1000):
                topic_model = tm
                original_tid = int(tid) - cluster_offset
                break

        # Get topic words if we found the right model
        topic_words_list = []
        if topic_model:
            try:
                raw_topic_words = topic_model.get_topic(original_tid)
                topic_words_list = cast(
                    List[Tuple[str, float]],
                    raw_topic_words if isinstance(raw_topic_words, list) else [],
                )
            except:
                # If we can't get topic words, use empty list
                topic_words_list = []

        result_topics[str(tid)] = {
            "concepts": cluster_info[i]["concepts"],
            "heading": cluster_info[i]["heading"],
            "summary": cluster_info[i]["summary"],
            "keywords": [word for word, _ in topic_words_list[:5]],
            "examples": [chunk["text"] for chunk in all_topic_maps[tid][:3]],
            "segment_positions": [chunk["position"] for chunk in all_topic_maps[tid]],
            "stats": topic_stats[tid],
        }

    return result_topics


def _save_processed_data(
    filename: str,
    chunks: List[Dict[str, str]],
    all_topic_maps: Dict[str, List[Dict[str, str]]],
    all_noise_chunks: List[Dict[str, str]],
    total_tokens: int,
    total_words: int,
    noise_words: int,
    topic_words: int,
    ordered_topic_ids: List[str],
    all_topic_models: List[Tuple[Any, int]],
    cluster_info: List[Dict[str, str]],
    topic_stats: Dict[str, Dict[str, Any]],
    result_topics: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Save processed data to a JSON file and clean up temporary files.

    Args:
        filename: Name of the file to save
        chunks: Original list of chunks
        all_topic_maps: Dictionary mapping topic IDs to lists of chunks
        all_noise_chunks: List of chunks not assigned to any topic
        total_tokens: Total tokens used for processing
        total_words: Total word count
        noise_words: Word count in noise chunks
        topic_words: Word count in topic chunks
        ordered_topic_ids: List of topic IDs in chronological order
        all_topic_models: List of topic models and their cluster IDs
        cluster_info: List of cluster information
        topic_stats: Dictionary of topic statistics
        result_topics: Dictionary of result topics

    Returns:
        Dictionary containing the saved data structure
    """
    print(f"\nSaving processed data:")
    print(f"Creating processed directory at: {PROCESSED_DIR}")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    base_name = os.path.splitext(filename)[0]
    save_path = os.path.join(PROCESSED_DIR, f"{base_name}_processed.json")
    print(f"Saving processed data to: {save_path}")

    # Prepare data for saving - collect all chunks from topics and noise
    all_segments = []
    # Add chunks from topics
    for topic_chunks in all_topic_maps.values():
        all_segments.extend(topic_chunks)
    # Add noise chunks
    all_segments.extend(all_noise_chunks)
    # Sort by position
    all_segments.sort(key=lambda x: x["position"])

    segments_word_count = sum(len(seg["text"].split()) for seg in all_segments)
    print(f"\nJSON saving stats:")
    print(f"Number of segments being saved: {len(all_segments)}")
    print(f"Total words in segments: {segments_word_count}")
    print(f"Expected total words: {topic_words + noise_words}")

    clusters = []
    for i, tid in enumerate(ordered_topic_ids):
        # Find which topic model this topic belongs to
        topic_model = None
        original_tid = tid

        # Check which cluster this topic came from based on the offset
        for tm, cluster_idx in all_topic_models:
            cluster_offset = cluster_idx * 1000
            if int(tid) >= cluster_offset and int(tid) < (cluster_offset + 1000):
                topic_model = tm
                original_tid = int(tid) - cluster_offset
                break

        # Get topic words if we found the right model
        topic_words_list = []
        if topic_model:
            try:
                raw_topic_words = topic_model.get_topic(original_tid)
                topic_words_list = cast(
                    List[Tuple[str, float]],
                    raw_topic_words if isinstance(raw_topic_words, list) else [],
                )
            except:
                # If we can't get topic words, use empty list
                topic_words_list = []

        clusters.append(
            {
                "cluster_id": tid,
                "heading": cluster_info[i]["heading"],
                "concepts": cluster_info[i]["concepts"],
                "segment_positions": [
                    chunk["position"] for chunk in all_topic_maps[tid]
                ],
                "keywords": [word for word, _ in topic_words_list[:5]],
                "examples": [chunk["text"] for chunk in all_topic_maps[tid][:3]],
                "stats": topic_stats[tid],
                "summary": cluster_info[i]["summary"],
            }
        )

    # --- FIX: Always include segments and meta in the saved file ---
    save_data = {
        "segments": all_segments,  # All original chunks, sorted by position
        "clusters": clusters,
        "meta": {
            "num_chunks": len(chunks),
            "num_topics": len(all_topic_maps),
            "num_noise_chunks": len(all_noise_chunks),
            "total_tokens_used": total_tokens,
            "total_words": total_words,
            "words_in_topics": topic_words,
            "words_in_noise": noise_words,
            "words_in_segments": segments_word_count,
            "kmeans_clusters_used": len(clusters),
        },
        # Also include the top-level topic modeling summary for compatibility
        "num_chunks": len(chunks),
        "num_topics": len(all_topic_maps),
        "total_tokens_used": total_tokens,
        "topics": result_topics,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved processed data to: {save_path}")
    print(f"JSON file size: {os.path.getsize(save_path) / 1024:.1f} KB")

    # Attempt to remove original transcript if it exists
    transcript_path = os.path.join("output", f"{base_name}_transcription.txt")
    try:
        if os.path.exists(transcript_path):
            os.remove(transcript_path)
            print(f"Deleted original transcript: {transcript_path}")
    except Exception as e:
        print(f"Warning: Failed to delete original transcript: {e}")

    # Attempt to remove the corresponding _chunks.json file after saving _processed.json
    chunks_path = os.path.join(PROCESSED_DIR, f"{base_name}_chunks.json")
    try:
        if os.path.exists(chunks_path):
            os.remove(chunks_path)
            print(f"Deleted chunk file: {chunks_path}")
    except Exception as e:
        print(f"Warning: Failed to delete chunk file: {e}")

    return save_data


def process_with_bertopic(
    chunks: List[Dict[str, str]], filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process chunks using BERTopic to generate topics and analyze them.
    First uses k-means to pre-cluster chunks, then runs BERTopic on each cluster.
    Falls back to single-cluster processing if pre-clustering criteria aren't met.

    Args:
        chunks: List of chunk dictionaries with 'position' and 'text' keys
        filename: Optional filename to save processed data (if None, no file is saved)

    Returns:
        Dictionary containing topic analysis results
    """
    # Check if chunks is empty
    if not chunks:
        print("Error: No chunks provided to process_with_bertopic")
        return {"num_chunks": 0, "num_topics": 0, "total_tokens_used": 0, "topics": {}}

    # Step 1: Print initial statistics
    total_words = _print_initial_statistics(chunks)

    # Step 2: Pre-cluster with k-means
    clusters = pre_cluster_with_kmeans(chunks)

    print(f"\n🔬 K-Means Clustering Results:")
    print(f"Number of k-means groups created: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        cluster_words = sum(len(chunk["text"].split()) for chunk in cluster)
        print(f"  Group {i}: {len(cluster)} chunks, {cluster_words} words")

    # Step 3: Process each cluster with BERTopic
    all_topic_maps, all_noise_chunks, all_topic_models = (
        _process_clusters_with_bertopic(clusters)
    )

    # Step 4: Print overall results
    noise_words, topic_words = _print_overall_results(
        clusters, all_topic_maps, all_noise_chunks
    )

    # Step 5: Generate topic headings and organize data
    ordered_topic_ids, topic_chunks_list, cluster_info, total_tokens = (
        _generate_topic_headings(all_topic_maps)
    )

    # Step 6: Calculate topic statistics
    topic_stats = _calculate_topic_statistics(ordered_topic_ids, all_topic_maps)

    # Step 7: Build result topics dictionary
    result_topics = _build_result_topics(
        ordered_topic_ids, all_topic_maps, all_topic_models, cluster_info, topic_stats
    )

    # Create basic result structure with segments
    # Prepare segments data similar to _save_processed_data
    all_segments = []
    # Add chunks from topics
    for topic_chunks in all_topic_maps.values():
        all_segments.extend(topic_chunks)
    # Add noise chunks
    all_segments.extend(all_noise_chunks)
    # Sort by position
    all_segments.sort(key=lambda x: x["position"])

    result = {
        "num_chunks": len(chunks),
        "num_topics": len(all_topic_maps),
        "total_tokens_used": total_tokens,
        "segments": all_segments,  # Include segments for frontend usage
        "topics": result_topics,
    }

    # Step 8: Save to file if filename is provided
    if filename:
        return _save_processed_data(
            filename,
            chunks,
            all_topic_maps,
            all_noise_chunks,
            total_tokens,
            total_words,
            noise_words,
            topic_words,
            ordered_topic_ids,
            all_topic_models,
            cluster_info,
            topic_stats,
            result_topics,
        )

    return result  # Return basic structure if no file was saved


def redistribute_large_topics(
    topic_map: Dict[str, List[Dict[str, str]]],
    topic_model: Any,
    cluster_id: int,
    max_topic_percentage: float = 0.6,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Redistribute chunks from overly large topics to smaller topics based on semantic similarity.

    Redistribution Logic:
    - Identify topics exceeding the maximum allowable percentage of total chunks.
    - Calculate the centroid (average vector) for each topic using TF-IDF vectorization.
    - For overly large topics, determine the least similar chunks to the topic's centroid.
    - Redistribute these chunks to the most semantically similar smaller topics based on cosine similarity.
    - Update the topic map with redistributed chunks and print a summary of changes.

    Args:
        topic_map: Dictionary mapping topic IDs to lists of chunks.
        topic_model: The BERTopic model used for this cluster.
        cluster_id: ID of the cluster being processed.
        max_topic_percentage: Maximum percentage of total chunks a single topic should contain.

    Returns:
        Updated topic_map with redistributed chunks.
    """
    if len(topic_map) < 2:
        print(
            f"Cluster {cluster_id}: Only {len(topic_map)} topics, skipping redistribution"
        )
        return topic_map

    # Calculate the total number of chunks across all topics
    total_chunks = sum(len(chunks) for chunks in topic_map.values())

    # Determine the maximum allowable chunks per topic based on the percentage threshold
    max_chunks_per_topic = int(total_chunks * max_topic_percentage)

    # Find overly large topics
    large_topics = {
        tid: chunks
        for tid, chunks in topic_map.items()
        if len(chunks) > max_chunks_per_topic
    }

    if not large_topics:
        print(
            f"Cluster {cluster_id}: No topics exceed {max_topic_percentage*100}% threshold, skipping redistribution"
        )
        return topic_map

    print(
        f"Cluster {cluster_id}: Found {len(large_topics)} large topics requiring redistribution"
    )  # Import necessary libraries for similarity calculation
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import scipy.sparse
    from typing import Union

    # Get all texts for vectorization
    all_texts = []
    text_to_topic = {}
    topic_to_texts = {}

    for tid, chunks in topic_map.items():
        topic_texts = [chunk["text"] for chunk in chunks]
        topic_to_texts[tid] = topic_texts
        for i, text in enumerate(topic_texts):
            all_texts.append(text)
            text_to_topic[len(all_texts) - 1] = (tid, i)
    # Vectorize all texts
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=500,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        text_vectors = vectorizer.fit_transform(all_texts)

        # Convert sparse matrix to dense array for easier manipulation
        import numpy as np

        text_vectors_dense = np.array(text_vectors.todense())

        # Calculate topic centroids (average vector for each topic)
        topic_centroids = {}
        for tid, chunks in topic_map.items():
            topic_indices = [i for i, (t_id, _) in text_to_topic.items() if t_id == tid]
            if topic_indices:
                # Select rows using dense array
                topic_vectors = text_vectors_dense[topic_indices]
                centroid = np.mean(topic_vectors, axis=0)
                topic_centroids[tid] = centroid

        redistributed_map = topic_map.copy()

        for large_tid, large_chunks in large_topics.items():
            # Calculate how many chunks to redistribute
            excess_chunks = len(large_chunks) - max_chunks_per_topic
            print(
                f"  Topic {large_tid}: {len(large_chunks)} chunks, redistributing {excess_chunks}"
            )
            # Get vectors for chunks in this large topic
            large_topic_indices = [
                i for i, (t_id, _) in text_to_topic.items() if t_id == large_tid
            ]
            large_topic_vectors = text_vectors_dense[large_topic_indices]

            # Calculate distance from each chunk to the topic centroid
            large_centroid = topic_centroids[large_tid].reshape(1, -1)
            distances = cosine_similarity(large_topic_vectors, large_centroid).flatten()

            # Sort chunks by distance (farthest first - least similar to topic center)
            chunk_distances = list(zip(large_chunks, distances))
            chunk_distances.sort(
                key=lambda x: x[1]
            )  # Sort by similarity (ascending = least similar first)

            # Select chunks to redistribute (least similar ones)
            chunks_to_redistribute = [
                chunk for chunk, _ in chunk_distances[:excess_chunks]
            ]
            remaining_chunks = [chunk for chunk, _ in chunk_distances[excess_chunks:]]

            # Update the large topic with remaining chunks
            redistributed_map[large_tid] = remaining_chunks

            # Find best topics for each chunk to redistribute
            other_topics = {
                tid: centroid
                for tid, centroid in topic_centroids.items()
                if tid != large_tid
            }

            for chunk in chunks_to_redistribute:
                chunk_text = chunk["text"]
                chunk_vector = vectorizer.transform([chunk_text])

                # Calculate similarity to all other topic centroids
                best_topic = None
                best_similarity = -1

                for other_tid, other_centroid in other_topics.items():
                    similarity = cosine_similarity(
                        chunk_vector, other_centroid.reshape(1, -1)
                    )[0][0]
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_topic = other_tid

                # Add chunk to the most similar topic
                if best_topic:
                    redistributed_map[best_topic].append(chunk)
                    print(
                        f"    Moved chunk to topic {best_topic} (similarity: {best_similarity:.3f})"
                    )

        # Print redistribution summary
        print(f"Cluster {cluster_id} redistribution summary:")
        for tid, chunks in redistributed_map.items():
            original_count = len(topic_map[tid])
            new_count = len(chunks)
            change = new_count - original_count
            change_str = f"+{change}" if change > 0 else str(change)
            print(
                f"  Topic {tid}: {original_count} → {new_count} chunks ({change_str})"
            )

        return redistributed_map

    except Exception as e:
        print(f"Cluster {cluster_id}: Error during redistribution: {e}")
        print("Returning original topic map")
        return topic_map


def order_topics_chronologically(
    all_topic_maps: Dict[str, List[Dict[str, str]]],
) -> List[str]:
    """
    Order topics chronologically based on the timestamp/position metadata of their most important chunks.

    This function improves the user experience by ordering topics according to their natural progression
    in the original content, rather than by arbitrary topic IDs. This ensures that:
    - Topics follow the logical flow of the original lecture/document
    - Students can read topics in the order they were presented
    - The content maintains its intended pedagogical structure
    - Related concepts appear in their proper sequence

    The function determines the "importance" of chunks within each topic by their position within the topic
    (assuming BERTopic places more representative chunks first) and orders topics by the earliest
    position of the most important chunks.

    Args:
        all_topic_maps: Dictionary mapping topic IDs to lists of chunks

    Returns:
        List of topic IDs ordered chronologically

    Example:
        >>> topic_maps = {
        ...     "1000": [{"position": "45", "text": "later content"}],
        ...     "1001": [{"position": "10", "text": "early content"}]
        ... }
        >>> order_topics_chronologically(topic_maps)
        ['1001', '1000']  # 1001 comes first because position 10 < 45
    """
    topic_min_positions = {}

    for tid, chunks in all_topic_maps.items():
        if not chunks:
            topic_min_positions[tid] = float("inf")
            continue

        # Get positions as integers for proper sorting
        # Take the first few chunks as they are most representative of the topic
        num_important_chunks = min(
            3, len(chunks)
        )  # Use top 3 most representative chunks
        important_chunks = chunks[:num_important_chunks]

        # Convert positions to integers and find the minimum
        positions = []
        for chunk in important_chunks:
            try:
                position = int(chunk["position"])
                positions.append(position)
            except (ValueError, KeyError):
                # If position is not a valid integer, skip this chunk
                continue

        if positions:
            # Use the earliest position among the important chunks
            topic_min_positions[tid] = min(positions)
        else:
            # If no valid positions found, put at the end
            topic_min_positions[tid] = float("inf")

    # Sort topics by their minimum position (chronological order)
    ordered_topic_ids = sorted(
        all_topic_maps.keys(), key=lambda tid: topic_min_positions[tid]
    )

    print(f"\n📍 Topic chronological ordering:")
    for i, tid in enumerate(ordered_topic_ids):
        min_pos = topic_min_positions[tid]
        chunk_count = len(all_topic_maps[tid])
        if min_pos == float("inf"):
            print(f"  {i+1}. Topic {tid}: No valid positions, {chunk_count} chunks")
        else:
            print(
                f"  {i+1}. Topic {tid}: Starting at position {min_pos}, {chunk_count} chunks"
            )

    return ordered_topic_ids
