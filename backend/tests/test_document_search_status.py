from app.models.document import Document, DocumentSearchStatus


def _document() -> Document:
    return Document(
        indexed=False,
        index_error=None,
    )


def test_document_search_status_tracks_initial_indexing() -> None:
    document = _document()
    assert document.search_status is DocumentSearchStatus.WAITING_FOR_INDEX

    document.index_error = "bounded failure"
    assert document.search_status is DocumentSearchStatus.INDEXING_FAILED
