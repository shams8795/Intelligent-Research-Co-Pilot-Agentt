"""Ollama handler for local LLM fallback when Groq API rate limit is hit."""

from typing import Optional
import requests


def is_ollama_running() -> bool:
    """Check if Ollama is running locally."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def get_available_models() -> list:
    """Get list of available models in Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
    except Exception:
        pass
    return []


def summarize_with_ollama(
    context: str,
    prompt: str,
    model: str = "mistral"
) -> Optional[str]:
    """Generate summary using local Ollama model.
    
    Args:
        context: The retrieved context/chunks about the paper
        model: Model name (default: mistral)
        prompt: The system prompt for summarization
        
    Returns:
        Generated summary or None if failed
    """
    if not is_ollama_running():
        return None
    
    try:
        # Check if model is available, if not use default
        available = get_available_models()
        if not available:
            return None
        
        # Use provided model if available, otherwise use first available
        model_to_use = model if model in available else available[0]
        
        full_prompt = f"{prompt}\n\nContext:\n{context}"
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_to_use,
                "prompt": full_prompt,
                "stream": False,
                "temperature": 0.7,
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        
    except Exception as e:
        print(f"Ollama summarization failed: {e}")
    
    return None


def get_ollama_status() -> dict:
    """Get Ollama status information."""
    status = {
        "running": is_ollama_running(),
        "models": []
    }
    
    if status["running"]:
        status["models"] = get_available_models()
    
    return status
