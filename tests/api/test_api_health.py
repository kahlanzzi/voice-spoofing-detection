import logging


def test_health_is_exactly_ok_true_with_no_session_origin_or_log(client, caplog):
    with caplog.at_level(logging.DEBUG):
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.content == b'{"ok":true}'
    assert [r for r in caplog.records if r.name == "api"] == []  # httpx logs its own client side


def test_health_reads_no_database(client):
    client.app.state.database.close()  # no connection can be taken from here on
    assert client.get("/api/health").json() == {"ok": True}
    assert not client.app.state.database.is_open
