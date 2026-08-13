from app.models.document import Document, DocumentSearchStatus


def _document() -> Document:
    return Document(
        indexed=False,
        index_error=None,
        index_generation=0,
        active_embedding_profile=None,
    )


def test_document_search_status_tracks_initial_indexing() -> None:
    document = _document()
    assert document.search_status is DocumentSearchStatus.WAITING_FOR_INDEX

    document.index_error = "bounded failure"
    assert document.search_status is DocumentSearchStatus.INDEXING_FAILED


def test_previous_active_generation_remains_searchable_after_failed_reindex() -> None:
    document = _document()
    document.index_error = "new generation failed"
    document.index_generation = 2
    document.active_embedding_profile = "bge-small:test-profile"

    assert document.search_status is DocumentSearchStatus.READY
