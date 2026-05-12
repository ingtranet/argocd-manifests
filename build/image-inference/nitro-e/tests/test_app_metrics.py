def test_metrics_endpoint_exposes_counters(client):
    # Trigger one generation so counters are non-zero.
    r = client.post("/v1/images/generations", json={"prompt": "a cat"})
    assert r.status_code == 200

    m = client.get("/metrics")
    assert m.status_code == 200
    body = m.text
    assert "nitro_e_generations_total" in body
    assert "nitro_e_generation_seconds" in body


def test_metrics_records_errors(client, fake_pipeline):
    async def boom(*, prompt, n, steps, guidance, seed):
        raise ValueError("nope")

    fake_pipeline.generate = boom
    client.post("/v1/images/generations", json={"prompt": "x"})

    m = client.get("/metrics")
    assert m.status_code == 200
    assert 'nitro_e_generation_errors_total{reason="ValueError"}' in m.text
