import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_smoke_training import smoke_config

from gae_credit.logging import RunLogger, validate_run
from gae_credit.logging.schemas import TABLE_SCHEMAS
from gae_credit.train import run_training


def test_failed_logger_has_typed_empty_tables(tmp_path):
    cfg = smoke_config(tmp_path)
    logger = RunLogger(cfg)
    with pytest.raises(ValueError, match="fields"):
        logger.log_update({"actor_loss": float("nan")})
    logger.fail("intentional test failure")
    assert json.loads((logger.path / "manifest.json").read_text())["status"] == "failed"
    for name, schema in TABLE_SCHEMAS.items():
        table = pq.read_table(logger.path / f"{name}.parquet")
        assert len(table) == 0
        assert table.schema.equals(schema)
    with pytest.raises(ValueError):
        validate_run(logger.path)


@pytest.mark.parametrize("damage", ["hash", "horizon", "nan", "duplicates", "schema", "steps"])
def test_validator_detects_corruption(tmp_path, damage):
    path = run_training(smoke_config(tmp_path), verbose=False)
    if damage in ("hash", "horizon"):
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["config_hash" if damage == "hash" else "critic_horizon"] = "wrong"
        (path / "manifest.json").write_text(json.dumps(manifest))
    else:
        table = pq.read_table(path / "updates.parquet")
        rows = table.to_pylist()
        if damage == "nan":
            rows[0]["actor_loss"] = float("nan")
        elif damage == "duplicates":
            rows[1]["update_index"] = rows[0]["update_index"]
        elif damage == "steps":
            rows[1]["env_steps"] -= 1
        table = pa.Table.from_pylist(rows, schema=table.schema)
        if damage == "schema":
            table = table.drop(["entropy"])
        pq.write_table(table, path / "updates.parquet")
    with pytest.raises(ValueError):
        validate_run(path)


def test_overwrite_cannot_remove_unrelated_directory(tmp_path):
    cfg = smoke_config(tmp_path)
    path = tmp_path / cfg.run_id
    path.mkdir()
    sentinel = path / "unrelated.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="valid manifest"):
        RunLogger(cfg, overwrite=True)
    assert sentinel.read_text() == "keep"
