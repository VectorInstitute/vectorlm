import os
import shutil
from tqdm import tqdm
from functools import partial
from datasets import load_dataset, disable_caching
from rag import EmbeddingModel, RerankerModel, VectorDB

BASE_DIR = "/fs01/home/ziwenhan/.cache/huggingface/datasets/lukaemon___mmlu" # TODO download to centralized place
EMBEDDING_MODEL = "/checkpoint/opt_test/original/clinical_llm/models--BAAI--bge-large-zh-v1.5/snapshots/b5c9d86d763d9945f7c0a73e549a4a39c423d520/"

RAG_BASE_PATH = "/scratch/ssd002/projects/opt_test/clinical_llm/datasets/rag"
KEY_PATH = os.path.join(RAG_BASE_PATH, "test_embeddings_512.pt")
VALUE_PATH = os.path.join(RAG_BASE_PATH, "test_text_512.pt")

def find_file_dir(path, filename):
    """Find all directories containing a specific filename.
    """
    for root, _, files in os.walk(path):
        for name in files:
            if name == filename:
                yield root


def replace_with_copy(old, new):
    """Replace the old file with a new file in the same directory.
    Rename the old file in the same directory as a copy, if no copy exists.
    """
    target_dir = os.path.dirname(old)
    for f in os.listdir(target_dir):
        if "OLD" in f:
            print(f"OLD-{os.path.basename(old)} exists, skipping caching...")
            shutil.copyfile(new, old)
            return
    os.rename(old, os.path.join(target_dir, "OLD-" + os.path.basename(old)))
    shutil.copyfile(new, old)


def rag_prefix(example, db, embedding_model, reranker_model, k = 3, 
               metaprompt = ""):
    
    """Add a retrieval augmented prefix (RAG) with reranking
    to each example based on a corpus"""
    question = example["input"]
    queries = [question] # TODO add batching & parallelism
    embedded_queries = embedding_model(queries)
    retrieved = db.topk_cosine(embedded_queries, k = k * 10)
    reranked = reranker_model(queries, retrieved, k = k)[0]
    prefix = metaprompt + "\n\n".join([f"[{j + 1}] " + reranked[j] for j in range(len(reranked))]) + "\n\n" + question + "\n"
    print(prefix)
    example["input"] = prefix + question
    return example


def main():
    print("*** Loading vector database ***")
    db = VectorDB(
        key_path = KEY_PATH,
        value_path = VALUE_PATH
    )

    print("*** Loading embedding model ***")
    embedding_model = EmbeddingModel(
        model_name_or_path = EMBEDDING_MODEL,
        retrieval_instruction = ""
    )

    print("*** Loading reranker model ***")
    reranker_model = RerankerModel(
        model_name_or_path = "BAAI/bge-reranker-large"
    )

    prefix_func = partial(rag_prefix, db = db, 
                          embedding_model = embedding_model, 
                          reranker_model = reranker_model)

    dataset_paths = find_file_dir(BASE_DIR, "dataset_info.json")
    for path in tqdm(dataset_paths):
        if any(["OLD" in f for f in os.listdir(path)]):
            print(f"Skipping {path} as it is already processed...")
            continue 
        print("*** Processing ***")
        print(path)
        dataset = load_dataset(path)
        test_split = dataset["test"].map(prefix_func)
        test_split.save_to_disk("rag_cache", num_shards=1)

        test_path = os.path.join(path, "mmlu-test.arrow")
        replace_with_copy(test_path, 
                          os.path.join("rag_cache", "data-00000-of-00001.arrow"))
        shutil.rmtree("rag_cache")


if __name__ == "__main__":
    disable_caching()
    main()