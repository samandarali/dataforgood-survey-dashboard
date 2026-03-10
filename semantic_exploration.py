"""
Semantic exploration functions for analyzing open-ended text responses using BERTopic.
"""

# Standard library imports
import pandas as pd
from functools import lru_cache

# Third-party imports
try:
    import umap
except ImportError:
    umap = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

try:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    BERTopic = None
    KeyBERTInspired = None
    MaximalMarginalRelevance = None
    SentenceTransformer = None
    UMAP = None
    HDBSCAN = None
    CountVectorizer = None

# Load embedding model once globally to avoid reloading for every question
EMBEDDING_MODEL = None
if SentenceTransformer is not None:
    try:
        EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        EMBEDDING_MODEL = None

# Cache embeddings for repeated analyses (huge speed improvement)
@lru_cache(maxsize=128)
def embed_texts(text_tuple):
    """Cache embeddings for repeated text analyses."""
    if EMBEDDING_MODEL is None:
        return None
    return EMBEDDING_MODEL.encode(list(text_tuple), show_progress_bar=False)

import re

def clean_responses(df_sub):
    df_sub = df_sub.copy()

    df_sub["response_clean"] = (
        df_sub["response"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"http\S+", "", regex=True)
        .str.replace(r"[^a-zA-Z\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df_open = df_sub[
        df_sub["response_clean"].str.len() > 3
    ].copy()

    return df_open

def compute_umap(embeddings, n_components=2, random_state=42):
    """
    Compute UMAP embeddings for visualization.
    """
    if umap is not None:
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        umap_embeddings = reducer.fit_transform(embeddings)
        return umap_embeddings
    elif PCA is not None:
        # Fallback to PCA if UMAP not available
        reducer = PCA(n_components=n_components, random_state=random_state)
        pca_embeddings = reducer.fit_transform(embeddings)
        return pca_embeddings
    else:
        raise ImportError("Neither umap-learn nor scikit-learn is installed.")

# -----------------------------
# BERTopic-based functions
# -----------------------------

def cluster_responses_bertopic(
    df_open,
    min_topic_size: int = 10,
    n_neighbors: int = 15,
    n_components: int = 2,
    min_dist: float = 0.1, # min distance between points in the low-dimensional space
):
    """
    Cluster responses using BERTopic.

    Notes
    -----
    - Uses a globally loaded embedding model (loaded once, reused for all questions)
      to avoid expensive model reloading and recomputation.
    - Uses KeyBERTInspired representation model for better topic labels.
    - Automatically adapts clustering parameters based on dataset size:
      * < 20 responses: relaxed clustering (min_topic_size=2)
      * 20-100 responses: moderate clustering (min_topic_size = n_docs // 10)
      * > 100 responses: normal/default clustering
    - Generates improved topic labels using BERTopic's label generation.
    """
    if BERTopic is None or UMAP is None or HDBSCAN is None or CountVectorizer is None:
        raise ImportError("BERTopic dependencies are not installed. Install with: pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn")
    
    if EMBEDDING_MODEL is None:
        raise ImportError("SentenceTransformer model could not be loaded. Please check installation.")

    try:
        # Prepare documents
        documents = df_open["response_clean"].tolist()
        n_docs = len(documents)

        # Use cached embedding function for speed improvement on repeated analyses
        if EMBEDDING_MODEL is None:
            raise Exception("Embedding model not available. Please install sentence-transformers.")
        embeddings = embed_texts(tuple(documents))

        # Adapt parameters based on dataset size (3-tier approach)
        if n_docs == 5:
            # Smallest dataset: relaxed but valid clustering (min_cluster_size must be >= 2)
            effective_min_topic_size = 2
            effective_n_neighbors = 2
            effective_n_components = 2
        elif n_docs < 52:
            # Very small datasets: relaxed clustering
            effective_min_topic_size = 2
            effective_n_neighbors = max(2, n_docs // 2)
            effective_n_components = max(2, min(n_components, n_docs - 1))
        elif n_docs < 100:
            # Medium datasets: moderate clustering
            effective_min_topic_size = max(3, n_docs // 10)
            effective_n_neighbors = max(5, min(n_neighbors, n_docs - 1))
            effective_n_components = max(3, min(n_components, n_docs - 1))
        else:
            # Large datasets: normal/default clustering
            effective_min_topic_size = min_topic_size
            effective_n_neighbors = n_neighbors
            effective_n_components = n_components

        # Initialize UMAP
        umap_model = UMAP(
            n_neighbors=effective_n_neighbors,
            n_components=effective_n_components,
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
        )

        # Ensure HDBSCAN min_cluster_size respects library constraint (>= 2)
        if effective_min_topic_size < 2:
            effective_min_topic_size = 2

        # Initialize HDBSCAN
        hdbscan_model = HDBSCAN(
            min_cluster_size=effective_min_topic_size,
            min_samples = max(2, int(effective_min_topic_size * 0.75)),
            metric="euclidean",
            cluster_selection_method="eom",
        )
        # Initialize CountVectorizer for c-TF-IDF with custom stopwords and token pattern
        custom_stopwords = [
            "would", "could", "use", "using", 
            "thing", "things", "someone", "person", "session",
            
            "activity",
            "activities",
            "session",
            "group",
            "today",
            "really",
            "like"

        ]
        
        # Combine sklearn's English stopwords with custom ones
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            all_stopwords = list(set(ENGLISH_STOP_WORDS).union(custom_stopwords))
        except ImportError:
            # Fallback to just custom stopwords if sklearn stopwords not available
            all_stopwords = custom_stopwords
        
        # Adapt vectorizer parameters based on dataset size
        # For very small datasets, reduce min_df to avoid the "max_df < min_df" error
        if n_docs < 20:
            effective_min_df = 1  # Allow words that appear in at least 1 document
            effective_max_df = 0.95  # More lenient max_df for small datasets
        elif n_docs < 52:
            effective_min_df = 1
            effective_max_df = 0.95
        else:
            effective_min_df = 1
            effective_max_df = 0.9
        
        vectorizer_model = CountVectorizer(
            stop_words=all_stopwords,
            min_df=effective_min_df,
            max_df=effective_max_df,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # Keep only alphabetic words of length >= 2
        )

        # Initialize representation model with MMR for keyword diversity
        representation_model = None
        if KeyBERTInspired is not None and MaximalMarginalRelevance is not None:
            try:
                representation_model = [
                    KeyBERTInspired(),
                    MaximalMarginalRelevance(diversity=0.5)
                ]
            except Exception:
                # Fallback to single KeyBERTInspired if MMR fails
                try:
                    representation_model = KeyBERTInspired()
                except Exception:
                    representation_model = None
        elif KeyBERTInspired is not None:
            try:
                representation_model = KeyBERTInspired()
            except Exception:
                representation_model = None

        # Initialize BERTopic with custom vectorizer and representation model
        topic_model = BERTopic(
            embedding_model=EMBEDDING_MODEL,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            verbose=False,
        )

        # Fit the model
        topics, probs = topic_model.fit_transform(documents)
        topic_model.reduce_topics(documents, nr_topics="auto")

        # Generate better topic labels
        try:
            topic_model.set_topic_labels(topic_model.generate_topic_labels())
        except Exception:
            # If label generation fails, continue with default labels
            pass

        # Add cluster assignments to dataframe
        df_open = df_open.copy()
        df_open["cluster"] = topics

        # Get topic info (now with improved labels)
        topic_info = topic_model.get_topic_info()

        return df_open, topic_model, topic_info, embeddings

    except Exception as e:
        raise Exception(f"Error in BERTopic clustering: {str(e)}")

def extract_topics_bertopic(topic_model, df_open):
    """
    Extract topics and keywords from BERTopic model.
    
    Args:
        topic_model: Fitted BERTopic model
        df_open: DataFrame with cluster assignments
    
    Returns:
        Dictionary mapping cluster_id to keywords and topic representation
    """
    topics_dict = {}
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    
    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        
        # Skip outlier topic (-1)
        if topic_id == -1:
            continue
        
        # Get top words for this topic
        topic_words = topic_model.get_topic(topic_id)
        # Filter out empty strings, None values, and ensure words are strings
        keywords = [
            str(word).strip() 
            for word, _ in topic_words[:10] 
            if word and str(word).strip()
        ]
        
        # Get topic representation
        topic_representation = topic_model.get_topic(topic_id)
        
        topics_dict[topic_id] = {
            "keywords": keywords,
            "representation": topic_representation,
            "name": row.get("Name", f"Topic {topic_id}")
        }
    
    return topics_dict

def summarize_clusters_bertopic(df_open, topics_dict, topic_info, topic_model, total_responses=None):
    """
    Summarize clusters using BERTopic results.
    
    Args:
        df_open: DataFrame with cluster assignments
        topics_dict: Dictionary from extract_topics_bertopic
        topic_info: Topic info DataFrame from BERTopic
        topic_model: Fitted BERTopic model
        total_responses: Total number of responses
    
    Returns:
        DataFrame with cluster summaries
    """
    if total_responses is None:
        total_responses = len(df_open)
    
    summaries = []
    
    # Create mapping from topic_id to BERTopic-generated topic name
    topic_name_map = dict(zip(topic_info["Topic"], topic_info.get("Name", topic_info["Topic"])))
    
    for cluster_id in sorted(df_open["cluster"].unique()):
        if cluster_id == -1:
            continue
        
        cluster_df = df_open[df_open["cluster"] == cluster_id]
        cluster_size = len(cluster_df)
        if cluster_size < 5:
            continue
        percentage = (cluster_size / total_responses * 100) if total_responses > 0 else 0
        
        topic_data = topics_dict.get(cluster_id, {})
        keywords = topic_data.get("keywords", [])
        topic_name = topic_name_map.get(cluster_id, f"Topic {cluster_id}")
        
        # Filter out empty strings and None values, limit to top 6 keywords
        keywords_clean = [
            kw for kw in keywords[:6] 
            if kw and str(kw).strip()
        ]
        keywords_str = ", ".join(keywords_clean) if keywords_clean else "No keywords"
        
        # Get representative documents from BERTopic
        try:
            examples = topic_model.get_representative_docs(cluster_id)
            if examples is None:
                examples = []
            else:
                examples = examples[:3] if len(examples) > 3 else examples
        except Exception:
            # Fallback to sampling from cluster_df if get_representative_docs fails
            examples = cluster_df["response_clean"].sample(
                min(3, len(cluster_df))
            ).tolist()
        
        # Filter out empty example responses and ensure all are strings
        examples_clean = [
            str(ex).strip()
            for ex in examples 
            if ex is not None and str(ex).strip()
        ]
        examples_str = " | ".join(examples_clean) if examples_clean else "No examples"
        
        summaries.append({
            "cluster": cluster_id,
            "topic_name": topic_name,
            "size": cluster_size,
            "percentage": percentage,
            "keywords": keywords_str,
            "example_responses": examples_str
        })
    
    if not summaries:
        return pd.DataFrame(columns=["cluster", "topic_name", "size", "percentage", "keywords", "example_responses"])
    
    return pd.DataFrame(summaries).sort_values("size", ascending=False)

def semantic_analysis_per_question_bertopic(df, min_topic_size=10):
    """
    Run semantic analysis per question using BERTopic.
    
    Args:
        df: DataFrame with concept_key and response columns
        min_topic_size: Minimum topic size for BERTopic
    
    Returns:
        DataFrame with summaries per concept_key
    """
    results = []
    questions = df["concept_key"].dropna().unique()
    
    for q in questions:
        df_q = df[df["concept_key"] == q].copy()
        df_q = clean_responses(df_q)
        
        if len(df_q) < 3:
            continue
        
        try:
            # Use BERTopic for clustering
            df_q, topic_model, topic_info, embeddings = cluster_responses_bertopic(
                df_q, 
                min_topic_size=min_topic_size
            )
            
            # Extract topics
            topics_dict = extract_topics_bertopic(topic_model, df_q)
            
            # Summarize clusters
            summary = summarize_clusters_bertopic(
                df_q, 
                topics_dict, 
                topic_info,
                topic_model,
                total_responses=len(df_q)
            )
            
            # Skip if no clusters found
            if len(summary) == 0:
                continue
            
            summary["concept_key"] = q
            
            # Store topic model in results (optional, for visualization)
            results.append({
                "summary": summary,
                "topic_model": topic_model,
                "topic_info": topic_info,
                "embeddings": embeddings,
                "concept_key": q
            })
            
        except Exception as e:
            print(f"Error processing concept {q}: {str(e)}")
            continue
    
    if not results:
        return pd.DataFrame(columns=["cluster", "topic_name", "size", "percentage", "keywords", "example_responses", "concept_key"]), {}
    
    # Combine all summaries
    all_summaries = pd.concat([r["summary"] for r in results], ignore_index=True)
    
    # Store topic models by concept_key
    topic_models = {r["concept_key"]: r["topic_model"] for r in results}
    
    return all_summaries, topic_models

def run_semantic_pipeline_bertopic(df, min_topic_size=10):
    """
    Run semantic pipeline using BERTopic.
    
    Args:
        df: DataFrame with concept_key and response columns
        min_topic_size: Minimum topic size for BERTopic
    
    Returns:
        df_open, summary DataFrame, topic_models dictionary
    """
    df_open = clean_responses(df)
    summary, topic_models = semantic_analysis_per_question_bertopic(
        df_open, min_topic_size=min_topic_size
    )
    return df_open, summary, topic_models


def summarize_small_dataset(df_q):
    """
    Fallback summarization for very small datasets (e.g., exactly 5 responses)
    where BERTopic is unstable. Each response becomes its own 'topic'.
    """
    import numpy as np

    df_q = df_q.copy()

    # Ensure we have a clean text column to work with
    if "response_clean" in df_q.columns:
        texts = df_q["response_clean"].fillna("").astype(str)
    else:
        texts = df_q["response"].fillna("").astype(str)

    n = len(df_q)
    if n == 0:
        return pd.DataFrame(
            columns=["cluster", "topic_name", "size", "percentage", "keywords", "example_responses"]
        )

    rows = []
    for idx, text in enumerate(texts):
        tokens = [t for t in text.split() if len(t) > 3]
        keywords = ", ".join(tokens[:5]) if tokens else "No keywords"

        rows.append(
            {
                "cluster": idx,
                "topic_name": f"Response {idx + 1}",
                "size": 1,
                "percentage": float(np.round(100.0 / n, 1)),
                "keywords": keywords,
                "example_responses": text or "No text",
            }
        )

    return pd.DataFrame(rows)

