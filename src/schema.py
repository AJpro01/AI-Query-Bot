"""
schema.py
---------
Defines the data contract for every piece of text that comes out of the
parsing pipeline. Every downstream step (embedding, storage, retrieval,
citation) trusts that a chunk looks EXACTLY like this. If a chunk doesn't
match this shape, Pydantic raises an error immediately instead of letting
bad data quietly flow into your vector database.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChunkMetadata(BaseModel):
    """Structural 'address' of a chunk inside the book."""
    book_title: str
    chapter: Optional[str] = "Unknown"
    section: Optional[str] = "Unknown"
    page_number: int
    chunk_index_on_page: int = Field(
        description="Order of this chunk within its page, useful for debugging/ordering"
    )


class BookChunk(BaseModel):
    """A single retrievable unit of text plus its full lineage."""
    chunk_id: str = Field(description="Deterministic unique ID, e.g. 'goodfellow_p114_c2'")
    text: str = Field(description="The actual chunk text that gets embedded")
    metadata: ChunkMetadata


if __name__ == "__main__":
    # Quick sanity check: this should succeed
    example = BookChunk(
        chunk_id="goodfellow_p114_c0",
        text="The margin of a linear classifier is defined as...",
        metadata=ChunkMetadata(
            book_title="Deep Learning",
            chapter="3",
            section="3.2",
            page_number=114,
            chunk_index_on_page=0,
        ),
    )
    print(example.model_dump_json(indent=2))

    # This should FAIL loudly (page_number missing) -- proving validation works
    try:
        BookChunk(chunk_id="bad", text="oops", metadata={"book_title": "X"})
    except Exception as e:
        print("\nValidation correctly caught a bad chunk:")
        print(e)
