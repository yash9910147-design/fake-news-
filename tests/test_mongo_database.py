from unittest.mock import Mock

from bson import ObjectId

from backend.app.database import MongoDatabase


def test_mongo_save_analysis_returns_string_id():
    repository = MongoDatabase.__new__(MongoDatabase)
    object_id = ObjectId()
    repository.analyses = Mock()
    repository.analyses.insert_one.return_value.inserted_id = object_id
    document = {"original_input": "A test headline"}

    saved = repository.save_analysis(document)

    assert saved["id"] == str(object_id)
    repository.analyses.insert_one.assert_called_once_with(document)


def test_mongo_history_serializes_ids_and_timestamps():
    from datetime import datetime

    object_id = ObjectId()
    row = {
        "_id": object_id,
        "input_type": "text",
        "headline": None,
        "ml": {"label": "Real"},
        "final_assessment": "UNVERIFIED",
        "created_at": datetime(2026, 1, 1),
    }
    repository = MongoDatabase.__new__(MongoDatabase)
    cursor = Mock()
    cursor.sort.return_value.limit.return_value = [row]
    repository.analyses = Mock()
    repository.analyses.find.return_value = cursor

    history = repository.list_history(10)

    assert history[0]["id"] == str(object_id)
    assert history[0]["ml_prediction"] == "Real"
    assert history[0]["created_at"] == "2026-01-01T00:00:00"


def test_mongo_analysis_lookup_rejects_invalid_object_id():
    repository = MongoDatabase.__new__(MongoDatabase)
    repository.analyses = Mock()

    assert repository.get_analysis("not-an-object-id") is None
    repository.analyses.find_one.assert_not_called()
