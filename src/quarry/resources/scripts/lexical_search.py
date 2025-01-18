import bm25s

def lexical_search(query: str, corpus: list[str], k: int) -> list[str]:
    """Lexical search using BM25 algorithm where query is compared against a corpus to find the top n most relevant data."""
    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(bm25s.tokenize(corpus))
    # Query the corpus and get top-k results
    results, _ = retriever.retrieve(bm25s.tokenize(query), k=k)
    return results[0]
