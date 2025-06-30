from fastapi import APIRouter, UploadFile, File
from starlette.responses import StreamingResponse
import os
from pydub import AudioSegment
import json
from uuid import uuid4
from fastapi import BackgroundTasks
import asyncio
from datetime import datetime
from utils.bullet_point_debugger import debug_bullet_point

router = APIRouter()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"  # Folder to save the extracted/transcribed text files

JOB_STATUS: dict[str, dict] = {}  # {job_id: {stage:…, current:…, total:…}}


def set_status(job_id: str, **kwargs):
    old_status = JOB_STATUS.get(job_id, {})
    new_status = {**old_status, **kwargs}
    JOB_STATUS[job_id] = new_status

    # 💡 Print to terminal
    stage = new_status.get("stage", "unknown")
    msg = f"[{job_id[:8]}] → stage: {stage}"
    if "current" in new_status and "total" in new_status:
        msg += f" ({new_status['current']}/{new_status['total']})"
    if "error" in new_status:
        msg += f" ⚠️ error: {new_status['error']}"
    print(msg)


# Create the directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_m4a_to_wav(input_path: str) -> str:
    """
    Convert m4a file to wav format.

    Args:
        input_path: Path to the m4a file

    Returns:
        Path to the converted wav file
    """
    output_path = input_path.rsplit(".", 1)[0] + ".wav"
    audio = AudioSegment.from_file(input_path, format="m4a")
    audio.export(output_path, format="wav")
    return output_path


@router.get("/progress/{job_id}")
async def progress_stream(job_id: str):
    async def event_generator():
        while True:
            status = JOB_STATUS.get(job_id)
            if status is None:
                yield f"event: error\ndata: {json.dumps({'error': 'Invalid job ID'})}\n\n"
                break
            yield f"data: {json.dumps(status)}\n\n"

            if status.get("stage") in ["done", "error"]:
                break
            await asyncio.sleep(0.5)

    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_generator(), headers=headers)


@router.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid4())
    set_status(job_id, stage="uploading")

    def process_file(file_bytes: bytes, filename: str):
        try:
            ext = filename.split(".")[-1].lower()

            if ext not in ["pdf", "mp3", "wav", "txt", "m4a"]:
                return {"error": "Unsupported file type."}

            file_location = os.path.join(UPLOAD_DIR, filename)

            # Save the original file to the 'uploads' folder
            with open(file_location, "wb") as f:
                f.write(file_bytes)

            set_status(job_id, stage="preprocessing")

            # Define paths
            base_name = (file.filename or "uploaded_file").rsplit(".", 1)[0]
            output_filename = f"{base_name}_transcription.txt"
            output_file_location = os.path.join(OUTPUT_DIR, output_filename)
            processed_path = os.path.join("processed", f"{base_name}_processed.json")

            # If fully processed JSON exists, reconstruct transcript and load headings
            if os.path.exists(processed_path):
                with open(processed_path, "r", encoding="utf-8") as f:
                    processed_data = json.load(f)

                segments = processed_data.get("segments", [])
                clusters = processed_data.get("clusters", [])
                meta = processed_data.get("meta", {})

                print(f"[{job_id[:8]}] Loaded cached JSON: {processed_path}")
                print(
                    f"[{job_id[:8]}] Segments: {len(segments)}, Clusters: {len(clusters)}"
                )
                print(f"\nReconstructing transcript from JSON:")
                print(f"Number of segments in JSON: {len(segments)}")
                print(
                    f"Total words in segments: {meta.get('words_in_segments', 'N/A')}"
                )
                print(f"Words in topics: {meta.get('words_in_topics', 'N/A')}")
                print(f"Words in noise: {meta.get('words_in_noise', 'N/A')}")

                # Sort segments by position to reconstruct full transcript
                full_text = "\n\n".join(
                    seg["text"] for seg in sorted(segments, key=lambda x: x["position"])
                )

                # Print reconstruction stats
                reconstructed_words = len(full_text.split())
                print(f"\nReconstruction stats:")
                print(f"Words in reconstructed text: {reconstructed_words}")
                print(
                    f"Expected words from segments: {meta.get('words_in_segments', 'N/A')}"
                )
                if meta.get("words_in_segments"):
                    print(
                        f"Word count difference: {reconstructed_words - meta.get('words_in_segments')}"
                    )  # Convert cluster structure to match TopicResponse in frontend
                topics = {
                    str(cluster["cluster_id"]): {
                        "concepts": cluster.get("concepts", []),
                        "heading": cluster.get("heading", ""),
                        "summary": cluster.get("summary", ""),  # Include summary field
                        "keywords": cluster.get("keywords", []),
                        "examples": cluster.get("examples", []),
                        "segment_positions": cluster.get(
                            "segment_positions", []
                        ),  # Include segment_positions for full chunk access
                        "stats": cluster.get("stats", {}),
                        "bullet_points": cluster.get(
                            "bullet_points", None
                        ),  # Include bullet_points if they exist
                        "bullet_expansions": cluster.get(
                            "bullet_expansions", {}
                        ),  # Include bullet_expansions if they exist
                    }
                    for cluster in clusters
                }

                # Clean up the original file since we're using cached data
                try:
                    os.remove(file_location)
                except Exception as e:
                    print(
                        f"Warning: Failed to remove original file {file_location}: {e}"
                    )

                set_status(
                    job_id,
                    stage="done",
                    result={
                        "filename": file.filename,
                        "filetype": ext,
                        "text": full_text.strip(),
                        "message": "Using previously generated topics.",
                        "segments": segments,  # Include segments for frontend chunk access
                        "topics": topics,
                    },
                )
                return

            # If just the transcription exists (but not processed data)
            if os.path.exists(output_file_location):
                with open(output_file_location, "r", encoding="utf-8") as f:
                    existing_text = f.read().strip()

                # Clean up the original file since we're using cached transcription
                try:
                    os.remove(file_location)
                except Exception as e:
                    print(
                        f"Warning: Failed to remove original file {file_location}: {e}"
                    )

                set_status(
                    job_id,
                    stage="done",
                    result={
                        "filename": file.filename,
                        "filetype": ext,
                        "text": existing_text,
                        "message": "Transcription file already exists. Skipping processing.",
                        "transcription_file": output_file_location,
                    },
                )
                return

            set_status(job_id, stage="transcribing")
            text = ""
            rnnoise_file = None  # Track RNNoise file path if created

            # Extract text for PDF files
            if ext == "pdf":
                import fitz  # PyMuPDF

                with fitz.open(file_location) as doc:
                    for page in doc:
                        text += page.get_text()  # type: ignore

            # Extract text for TXT files
            elif ext == "txt":
                with open(file_location, "r", encoding="utf-8") as f:
                    text = f.read()

            # Extract text for audio files
            elif ext in ["mp3", "wav", "m4a"]:
                try:
                    from utils.transcribe_audio import transcribe_audio_in_chunks

                    # Convert m4a to wav if necessary
                    if ext == "m4a":
                        print("Converting m4a to wav...")
                        file_location = convert_m4a_to_wav(file_location)
                    text, rnnoise_file = transcribe_audio_in_chunks(
                        file_location,
                        progress_callback=lambda current, total: set_status(
                            job_id, stage="transcribing", current=current, total=total
                        ),
                    )

                except FileNotFoundError as e:
                    if "ffmpeg" in str(e).lower():
                        return {
                            "error": "FFmpeg is not installed or not found in PATH. Please install FFmpeg and ensure it's in your system PATH.",
                            "details": str(e),
                        }
                    return {"error": f"File not found: {str(e)}"}
                except Exception as e:
                    return {"error": f"Error processing audio: {str(e)}"}
                finally:
                    # Clean up converted wav file if it was created
                    if ext == "m4a" and file_location.endswith(".wav"):
                        try:
                            os.remove(file_location)
                        except Exception as e:
                            print(
                                f"Warning: Failed to remove converted file {file_location}: {e}"
                            )

            # Save the transcribed/extracted text to a file in the 'output' folder
            set_status(job_id, stage="saving_output")
            with open(output_file_location, "w", encoding="utf-8") as f:
                f.write(text.strip())

            # Clean up files
            try:
                # Remove original file
                os.remove(file_location)
                # Remove RNNoise file if it was created
                if rnnoise_file and os.path.exists(rnnoise_file):
                    os.remove(rnnoise_file)
                    print(f"Deleted RNNoise file: {rnnoise_file}")
            except Exception as e:
                print(f"Warning: Failed to remove file {file_location}: {e}")

            set_status(
                job_id,
                stage="done",
                result={
                    "filename": file.filename,
                    "filetype": ext,
                    "text": text.strip(),
                    "message": f"{ext.upper()} file processed successfully.",
                    "transcription_file": output_file_location,
                },
            )
        except Exception as e:
            print(f"[{job_id[:8]}] ❌ ERROR: {e}")
            set_status(job_id, stage="error", error=str(e))

    # Read file content NOW, while request is active
    file_bytes = await file.read()
    filename = file.filename or "uploaded_file"

    # Validate file extension before processing
    ext = filename.split(".")[-1].lower()
    if ext not in ["pdf", "mp3", "wav", "txt", "m4a"]:
        return {"error": "Unsupported file type."}

    # Start background task and pass the raw data
    background_tasks.add_task(process_file, file_bytes, filename)

    return {"job_id": job_id, "message": "Upload accepted. Processing started."}


from utils.semantic_segmentation import semantic_segment
from utils.filter_chunks import filter_chunks
from utils.chunk_size_optimizer import optimize_chunk_sizes
from utils.bertopic_processor import process_with_bertopic


@router.post("/test-bertopic")
def test_bertopic(data: dict):
    text = data.get("text", "")
    full_filename = data.get("filename") or "default"  # Get filename from request
    filename = os.path.splitext(full_filename)[
        0
    ]  # Step 1: Segment → Filter → Optimize (using our improved pipeline)
    raw_chunks = semantic_segment(text, similarity_threshold=0.5)
    filtered_chunks = filter_chunks(raw_chunks, min_words=4, max_stopword_ratio=0.75)
    chunks = optimize_chunk_sizes(
        filtered_chunks, min_words=75, max_words=150, target_size=125
    )

    # Step 2: Process with BERTopic
    result = process_with_bertopic(chunks, filename)

    # Convert result to match frontend expectations
    return {
        "num_chunks": result["num_chunks"],
        "num_topics": result["num_topics"],
        "total_tokens_used": result["total_tokens_used"],
        "topics": result["topics"],
    }


@router.post("/process-chunks")
def process_chunks(data: dict):
    text = data.get("text", "")
    full_filename = data.get("filename", "default")
    filename = os.path.splitext(full_filename)[0]  # Step 1: Chunking pipeline
    raw_chunks = semantic_segment(text, similarity_threshold=0.5)
    filtered_chunks = filter_chunks(raw_chunks, min_words=4, max_stopword_ratio=0.75)
    chunks = optimize_chunk_sizes(
        filtered_chunks, min_words=50, max_words=100, target_size=75
    )

    # Step 2: Save the chunks to disk
    os.makedirs("processed", exist_ok=True)
    processed_path = os.path.join("processed", f"{filename}_chunks.json")
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "chunks": chunks,
                "meta": {
                    "total_words": sum(len(c["text"].split()) for c in chunks),
                    "num_chunks": len(chunks),
                },
            },
            f,
            indent=2,
        )

    word_counts = [len(c["text"].split()) for c in chunks]
    min_words = min(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0
    avg_words = round(sum(word_counts) / len(word_counts), 2) if word_counts else 0

    # Calculate second-minimum to show actual minimum compliance (excluding the one allowed outlier)
    second_min_words = min_words
    if len(word_counts) >= 2:
        sorted_counts = sorted(word_counts)
        second_min_words = sorted_counts[1]  # Second smallest value

    print(f"\n🔍 Chunking Summary for '{filename}':")
    print(f"Total Chunks: {len(chunks)}")
    print(
        f"Words per Chunk → second-min: {second_min_words}, max: {max_words}, avg: {avg_words}"
    )

    return {
        "message": "Chunks saved successfully.",
        "filename": filename,
        "num_chunks": len(chunks),
        "total_words": sum(word_counts),
        "chunk_stats": {
            "second_min": second_min_words,
            "max": max_words,
            "avg": avg_words,
        },
    }


@router.post("/generate-headings")
def generate_headings(data: dict):
    full_filename = data.get("filename")
    if not full_filename:
        return {"error": "Filename is required."}
    filename = os.path.splitext(full_filename)[0]

    chunk_file = os.path.join("processed", f"{filename}_chunks.json")
    processed_file = os.path.join("processed", f"{filename}_processed.json")
    chunks = []
    # Try to load chunks from _chunks.json, else fallback to _processed.json
    if os.path.exists(chunk_file):
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
            chunks = chunk_data.get("chunks", [])
    elif os.path.exists(processed_file):
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
            chunks = processed_data.get("segments", [])
    else:
        return {"error": f"Chunks not found for filename: {filename}"}

    # Run BERTopic
    result = process_with_bertopic(chunks, filename)

    # Save full result
    processed_path = os.path.join("processed", f"{filename}_processed.json")
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return {
        "message": "Topics generated successfully.",
        "num_topics": result["num_topics"],
        "num_chunks": result.get("num_chunks", 0),
        "total_tokens_used": result.get("total_tokens_used", 0),
        "segments": result.get("segments", chunks),  # Include segments for frontend
        "topics": result["topics"],
    }


@router.post("/expand-cluster")
def expand_cluster(data: dict):
    """
    Expand a specific cluster in the processed file for the given filename and cluster ID.

    Args:
        data (dict): A dictionary containing 'filename' (str) and 'cluster_id' (str or int).

    Returns:
        dict:
            On success, returns a dictionary with expanded cluster information, e.g.:
    full_filename = data.get("filename")
    cluster_id = data.get("cluster_id")

    if full_filename is None or cluster_id is None:
        return {"error": "Filename and cluster ID are required."}

    # Validate and convert cluster_id to int if possible
    try:
        cluster_id = int(cluster_id)
    except (ValueError, TypeError):
        return {"error": "cluster_id must be an integer."}

    # Prepare the filename relative to the 'processed' folder as expected by expand_cluster
    processed_filename = os.path.splitext(full_filename)[0] + "_processed.json"

    from utils.expand_cluster import expand_cluster as expand_cluster_util

    # Pass only the filename (not the full path) to expand_cluster
    result = expand_cluster_util(processed_filename, cluster_id)
    return result
        - Processed file not found or invalid.
        - Cluster ID not found in the processed data.
    """
    full_filename = data.get("filename")
    cluster_id = data.get("cluster_id")

    if full_filename is None or cluster_id is None:
        return {"error": "Filename and cluster ID are required."}

    # Prepare the filename relative to the 'processed' folder as expected by expand_cluster
    processed_filename = os.path.splitext(full_filename)[0] + "_processed.json"

    from utils.expand_cluster import expand_cluster as expand_cluster_util

    # Pass only the filename (not the full path) to expand_cluster
    result = expand_cluster_util(processed_filename, cluster_id)
    return result


@router.post("/debug-bullet-point")
def debug_bullet_point_endpoint(data: dict):
    """
    Debug a bullet point by finding its most similar chunk and comparing it to other topics.

    Args:
        data (dict): A dictionary containing 'bullet_point' (str), 'chunks' (list of str), and 'topics' (dict).

    Returns:
        dict: Debugging result including the most similar chunk, similarity to the current topic, and topic similarities.
    """
    bullet_point = data.get("bullet_point")
    chunks = data.get("chunks", [])
    topics = data.get("topics", {})

    if not bullet_point or not chunks or not topics:
        return {
            "error": "Missing required fields: 'bullet_point', 'chunks', or 'topics'."
        }

    try:
        result = debug_bullet_point(bullet_point, chunks, topics)

        # Convert numpy types to native Python types
        result["similarity_to_current_topic"] = float(
            result["similarity_to_current_topic"]
        )

        # Convert top_similar_chunks similarities to float
        if "top_similar_chunks" in result:
            result["top_similar_chunks"] = [
                {"chunk": chunk["chunk"], "similarity": float(chunk["similarity"])}
                for chunk in result["top_similar_chunks"]
            ]

        topic_similarities = result.get("topic_similarities")
        if isinstance(topic_similarities, dict):
            result["topic_similarities"] = {
                key: float(value) for key, value in topic_similarities.items()
            }
        else:
            result["topic_similarities"] = {}

        return result
    except Exception as e:
        return {"error": f"Failed to debug bullet point: {str(e)}"}


@router.post("/expand-bullet-point")
def expand_bullet_point_endpoint(data: dict):
    """
    Expand a bullet point with additional detail and context.

    Args:
        data (dict): A dictionary containing 'bullet_point' (str), 'chunks' (list of str), 'topic_heading' (str),
                     'filename' (str), 'topic_id' (str), and optional 'layer' (int).

    Returns:
        dict: Expansion result including the original bullet point and expanded content.
    """
    bullet_point = data.get("bullet_point")
    chunks = data.get("chunks", [])
    topic_heading = data.get("topic_heading", "Unknown Topic")
    filename = data.get("filename")
    topic_id = data.get("topic_id")
    layer = data.get("layer", 1)  # Default to layer 1 if not specified

    print(f"\n🎯 Expand bullet point endpoint called")
    print(f"📝 Bullet point: {bullet_point[:50] if bullet_point else 'None'}...")
    print(f"📦 Received {len(chunks)} chunks for expansion")
    print(f"🏷️ Topic heading: {topic_heading}")
    print(f"📁 Filename: {filename}")
    print(f"🆔 Topic ID: {topic_id}")
    print(f"🔢 Expansion layer: {layer}")

    if not bullet_point or not chunks:
        error_msg = "Missing required fields: 'bullet_point' or 'chunks'."
        print(f"❌ Error: {error_msg}")
        return {"error": error_msg}

    try:
        from utils.expand_bullet_point import expand_bullet_point

        result = expand_bullet_point(bullet_point, chunks, topic_heading, layer)
        print(f"✅ Expansion completed successfully")

        # Save the expansion to the processed JSON file following the same pattern as expand_cluster.py
        if filename and topic_id and not result.get("error"):
            try:
                base_name = os.path.splitext(filename)[0]
                processed_file = os.path.join(
                    "processed", f"{base_name}_processed.json"
                )

                if os.path.exists(processed_file):
                    with open(processed_file, "r", encoding="utf-8") as f:
                        processed_data = json.load(f)

                    # Find the cluster in the processed data (same as expand_cluster.py)
                    clusters = processed_data.get("clusters", [])
                    cluster = None
                    for c in clusters:
                        if str(c.get("cluster_id")) == str(topic_id):
                            cluster = c
                            break

                    if cluster:
                        # Initialize bullet_expansions if it doesn't exist
                        if "bullet_expansions" not in cluster:
                            cluster["bullet_expansions"] = {}

                        # Create a unique key for this bullet point
                        import re

                        clean_bullet = re.sub(r"^[-*+]\s*", "", bullet_point).strip()
                        bullet_key = clean_bullet[:80]

                        print(f"🔑 Generated bullet key: '{bullet_key}'")

                        # Initialize the expansion for this bullet if it doesn't exist
                        if bullet_key not in cluster["bullet_expansions"]:
                            cluster["bullet_expansions"][bullet_key] = {}

                        # Store the expansion result
                        expansion_data = {
                            "original_bullet": bullet_point,
                            "expanded_bullets": result.get("expanded_bullets", []),
                            "layer": layer,
                            "topic_heading": topic_heading,
                            "chunks_used": result.get("chunks_used", 0),
                            "timestamp": str(datetime.now()),
                        }

                        if layer == 1:
                            cluster["bullet_expansions"][bullet_key] = expansion_data
                        elif layer == 2:
                            # For layer 2, we need to save it nested under the parent bullet
                            parent_bullet = data.get("parent_bullet", "")
                            if parent_bullet:
                                # Extract the actual bullet text from the parent key (remove topic ID prefix)
                                if parent_bullet.startswith(f"{topic_id}_"):
                                    parent_bullet_text = parent_bullet[len(f"{topic_id}_"):]
                                else:
                                    parent_bullet_text = parent_bullet
                                
                                # Find the parent bullet's expansion
                                parent_found = False
                                for existing_key, existing_data in cluster["bullet_expansions"].items():
                                    if existing_key == parent_bullet_text or (existing_data.get("original_bullet") and existing_data["original_bullet"] == parent_bullet_text):
                                        # Initialize sub_expansions if it doesn't exist
                                        if "sub_expansions" not in existing_data:
                                            existing_data["sub_expansions"] = {}
                                        
                                        # Save the layer 2 expansion under the parent
                                        existing_data["sub_expansions"][bullet_key] = expansion_data
                                        parent_found = True
                                        print(f"✔️ Saved layer 2 expansion '{bullet_key}' under parent '{existing_key}'")
                                        break
                                
                                if not parent_found:
                                    print(f"⚠️ Parent bullet not found for layer 2 expansion. Parent: '{parent_bullet_text}'")
                                    # Fallback: save as separate entry
                                    cluster["bullet_expansions"][bullet_key] = expansion_data
                            else:
                                print(f"⚠️ No parent_bullet provided for layer 2 expansion")
                                # Fallback: save as separate entry
                                cluster["bullet_expansions"][bullet_key] = expansion_data
                        else:
                            print(f"⚠️ Unsupported expansion layer: {layer}")

                        # Save back to file (same as expand_cluster.py)
                        with open(processed_file, "w", encoding="utf-8") as f:
                            json.dump(processed_data, f, indent=2)

                        print(
                            f"💾 Saved expansion to cluster {topic_id} in {processed_file}"
                        )
                    else:
                        print(f"⚠️ Cluster {topic_id} not found in processed data")
                else:
                    print(f"⚠️ Processed file not found: {processed_file}")
            except Exception as save_error:
                print(f"⚠️ Failed to save expansion: {save_error}")
                # Don't fail the request if saving fails

        return result
    except Exception as e:
        error_msg = f"Failed to expand bullet point: {str(e)}"
        print(f"💥 Error: {error_msg}")
        return {"error": error_msg}
