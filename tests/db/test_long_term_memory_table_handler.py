from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)


def test_save_load_long_term_memory_item(
    long_term_memory_table_handler: LongTermMemoryTableHandler,
) -> None:
    first_item = {"a1": "b"}
    long_term_memory_table_handler.save_history([first_item])
    results = long_term_memory_table_handler.search()
    assert len(results) == 1

    # Now test filtering based on datetime
    timestamp = utcnow()
    second_item = {"a2": "c"}
    long_term_memory_table_handler.save_history([second_item])

    results = long_term_memory_table_handler.search(to_=timestamp)
    assert len(results) == 1
    assert results[0].metadata_dict == first_item

    results = long_term_memory_table_handler.search(from_=timestamp)
    assert len(results) == 1
    assert results[0].metadata_dict == second_item

    # Retrieve all
    assert len(long_term_memory_table_handler.search()) == 2
