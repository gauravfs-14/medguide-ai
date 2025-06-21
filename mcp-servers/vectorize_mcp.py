from mcp.server.fastmcp import FastMCP

app = FastMCP("vectorize_mcp")

@app.tool()
def vectorize_mcp(
    text: str,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> list:
    """
    Vectorize text using a specified model.

    Args:
        text (str): The input text to vectorize.
        model (str): The model to use for vectorization.
        batch_size (int): The batch size for processing.

    Returns:
        list: A list of vectors representing the input text.
    """
    return _vectorize(text, model=model, batch_size=batch_size)


def _vectorize(text: str, model: str, batch_size: int) -> list:
    """Internal function to vectorize text."""
    return []

if __name__ == "__main__":
    app.run()
