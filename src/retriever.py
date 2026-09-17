"""Create LangChain retrievers backed by the project vector store."""

from typing import Any, Dict, Optional

from langchain_community.vectorstores import Chroma


def create_retriever(
	vector_store: Chroma,
	*,
	k: int = 4,
	fetch_k: int = 16,
	search_type: str = "mmr",
	search_kwargs: Optional[Dict[str, Any]] = None,
):
	"""Return a retriever built from a Chroma vector store.

	The default `mmr` search gives more diverse chunks than plain similarity,
	which helps summarization cover problem setup, method, experiments, and
	conclusions instead of retrieving several near-duplicate passages.
	"""
	if vector_store is None:
		raise ValueError("Vector store is required to create a retriever.")

	if k <= 0:
		raise ValueError("Retriever k must be greater than zero.")

	resolved_search_kwargs: Dict[str, Any] = {"k": k}
	if search_type == "mmr":
		resolved_search_kwargs["fetch_k"] = max(fetch_k, k)

	if search_kwargs:
		resolved_search_kwargs.update(search_kwargs)

	return vector_store.as_retriever(
		search_type=search_type,
		search_kwargs=resolved_search_kwargs,
	)
