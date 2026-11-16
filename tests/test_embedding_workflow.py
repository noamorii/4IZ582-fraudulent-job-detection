"""Offline regressions: no real endpoint or course key is used."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from embedding_export import save_cache
from embeddings import get_embeddings_and_stats
from model_comparison import grouped_cv, load_cache, run_search


class FakeClient:
    def __init__(self):
        self.calls = []
        self.embeddings = self
        self.with_raw_response = self
        self.fail_batch = None

    def create(self, *, model, input, **kwargs):
        self.calls.append((model, input))
        if input == self.fail_batch:
            raise ValueError("simulated permanent failure")
        # Reverse response order to exercise alignment by index.
        response = SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=[float(len(text)), float(ord(text[0]))])
            for i, text in reversed(list(enumerate(input)))
        ])
        return SimpleNamespace(parse=lambda: response, retries_taken=0)


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.settings = patch("embeddings.settings", return_value=("https://school.invalid/v1/", "", "test-model"))
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.client = FakeClient()
        self.client_patch = patch("embeddings.OpenAI", return_value=self.client)
        self.client_factory = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)

    def test_cache_covers_text_order_model_and_endpoint(self):
        client = self.client
        arr, info = get_embeddings_and_stats(["a", "bbb"], self.path)
        np.testing.assert_array_equal(arr[:, 0], [1, 3])
        self.assertEqual(arr.dtype, np.float32)
        self.assertEqual(info["endpoint_requests"], 1)
        cached, info = get_embeddings_and_stats(["a", "bbb"], self.path, offline=True)
        np.testing.assert_array_equal(arr, cached)
        self.assertEqual(info["new_endpoint_requests"], 0)
        for texts, configuration in [(["bbb", "a"], {}), (["x", "yyy"], {}),
                              (["a", "bbb"], {"model": "changed"}),
                              (["a", "bbb"], {"base_url": "https://other.invalid"})]:
            configuration = {
                "base_url": "https://school.invalid/v1/", "model": "test-model", **configuration,
            }
            settings = (configuration["base_url"], "", configuration["model"])
            with self.subTest(texts=texts, configuration=configuration):
                with patch("embeddings.settings", return_value=settings), self.assertRaises(FileNotFoundError):
                    get_embeddings_and_stats(texts, self.path, offline=True)
        self.assertEqual(len(client.calls), 1)
        repeated_client = FakeClient()
        self.client_factory.return_value = repeated_client
        _, repeated_info = get_embeddings_and_stats(
            ["a", "a"], self.path / "repeated", batch_size=1,
        )
        self.assertEqual(len(repeated_client.calls), 1)
        self.assertEqual(repeated_info["endpoint_requests"], 1)

    def test_resume_uses_completed_batches(self):
        client = self.client
        client.fail_batch = ["ccc"]
        with self.assertRaises(ValueError):
            get_embeddings_and_stats(["a", "bb", "ccc"], self.path, batch_size=2)
        client.fail_batch = None
        _, info = get_embeddings_and_stats(["a", "bb", "ccc"], self.path, batch_size=2)
        self.assertEqual(len(client.calls), 3)  # First successful batch was not repeated.
        self.assertEqual(info["new_endpoint_requests"], 1)
        self.assertEqual(info["endpoint_requests"], 2)  # Provenance of successful cached batches.

    def test_fixed_batches_preserve_order_and_offline_cache(self):
        texts = ["a", "b", "cccccc", "d", "e", "f"]
        client = self.client
        vectors, info = get_embeddings_and_stats(
            texts, self.path, batch_size=2,
        )
        self.assertEqual([batch for _, batch in client.calls],
                         [["a", "b"], ["cccccc", "d"], ["e", "f"]])
        np.testing.assert_array_equal(vectors[:, 0], [1, 1, 6, 1, 1, 1])
        cached, _ = get_embeddings_and_stats(texts, self.path, batch_size=2, offline=True)
        np.testing.assert_array_equal(cached, vectors)

    def test_fresh_batches_are_all_counted(self):
        client = self.client
        from embeddings import request_batch
        def timed_request(client, texts, model):
            vectors, _, requests = request_batch(client, texts, model)
            return vectors, 1.5, requests
        with patch("embeddings.request_batch", side_effect=timed_request):
            _, info = get_embeddings_and_stats(
                ["a", "b"], self.path, batch_size=1,
            )
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(info["new_endpoint_requests"], 2)
        self.assertEqual(info["endpoint_requests"], 2)
        self.assertEqual(info["embedding_seconds"], 3.0)

    def test_main_build_cache_and_offline_read(self):
        import main
        metadata, _ = sample_metadata()
        client = self.client
        with patch("main.prepare_data", return_value=(metadata, [])), patch("embeddings.OpenAI", return_value=client):
            vectors, _, _, _ = main.build_cache("unused.csv", self.path, batch_size=7, offline=False)
            np.testing.assert_array_equal(load_cache(self.path)[0], vectors)
            with patch("sys.argv", ["main.py", "--output", str(self.path)]):
                main.main()

    def test_invalid_responses_are_not_cached(self):
        for data in [[SimpleNamespace(index=0, embedding=[1.])],
                     [SimpleNamespace(index=0, embedding=[1.]), SimpleNamespace(index=0, embedding=[2.])],
                     [SimpleNamespace(index=0, embedding=[float("nan")]), SimpleNamespace(index=1, embedding=[2.])]]:
            client = self.client
            response = SimpleNamespace(data=data)
            raw = SimpleNamespace(parse=lambda: response, retries_taken=0)
            with patch.object(client, "create", return_value=raw):
                with self.assertRaises(ValueError):
                    get_embeddings_and_stats(["a", "b"], self.path)
            self.assertFalse(list(self.path.glob("*.json")))

    def test_sdk_compatible_client_with_mock_transport(self):
        import httpx
        from openai import OpenAI
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["input"], ["a", "bbb"])
            self.assertEqual(payload["model"], "test-model")
            return httpx.Response(200, json={
                "object": "list", "model": "test-model",
                "data": [{"object": "embedding", "index": 1, "embedding": [3., 4.]},
                         {"object": "embedding", "index": 0, "embedding": [1., 2.]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            })
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            with OpenAI(api_key="offline-placeholder", base_url="https://school.invalid/v1/",
                        http_client=http, max_retries=0) as client:
                with patch("embeddings.OpenAI", return_value=client):
                    arr, _ = get_embeddings_and_stats(["a", "bbb"], self.path)
        np.testing.assert_array_equal(arr, [[1., 2.], [3., 4.]])

    def test_retry_exhaustion_and_authentication_failure(self):
        import httpx
        from openai import APIStatusError, OpenAI
        for status, expected_attempts in [(429, 3), (500, 3), (503, 3), (401, 1)]:
            with self.subTest(status=status):
                calls = []
                def handler(request):
                    calls.append(request)
                    return httpx.Response(status, headers={"retry-after-ms": "1"},
                                          json={"error": {"message": "server error"}})
                with httpx.Client(transport=httpx.MockTransport(handler)) as http:
                    # Exercise the production constructor, substituting only the transport/key.
                    def factory(**kwargs):
                        self.assertEqual(kwargs["max_retries"], 2)
                        kwargs["api_key"] = "offline-placeholder"
                        return OpenAI(**kwargs, http_client=http)
                    with patch("embeddings.OpenAI", side_effect=factory):
                        with self.assertRaises(APIStatusError) as caught:
                            get_embeddings_and_stats(["a"], self.path)
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(len(calls), expected_attempts)
                self.assertFalse(list(self.path.glob("*.json")))

    def test_successful_retry_is_counted_and_cached(self):
        import httpx
        from openai import OpenAI
        calls = []
        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(503, headers={"retry-after-ms": "1"},
                                      json={"error": {"message": "temporarily unavailable"}})
            return httpx.Response(200, json={
                "object": "list", "model": "test-model",
                "data": [{"object": "embedding", "index": 0, "embedding": [1., 2.]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            })
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            def factory(**kwargs):
                self.assertEqual(kwargs["max_retries"], 2)
                kwargs["api_key"] = "offline-placeholder"
                return OpenAI(**kwargs, http_client=http)
            with patch("embeddings.OpenAI", side_effect=factory):
                vectors, info = get_embeddings_and_stats(["a"], self.path)
                cached, cache_info = get_embeddings_and_stats(["a"], self.path, offline=True)
        np.testing.assert_array_equal(cached, vectors)
        self.assertEqual(len(calls), 2)
        self.assertEqual(info["endpoint_requests"], 2)
        self.assertEqual(info["new_endpoint_requests"], 2)
        self.assertEqual(cache_info["endpoint_requests"], 2)
        self.assertEqual(cache_info["new_endpoint_requests"], 0)

def sample_metadata():
    groups = np.repeat(np.arange(60), 2)
    labels = (groups % 2).astype(int)
    text = [f"text {g}" for g in groups]
    meta = pd.DataFrame({"job_id": np.arange(120), "fraudulent": labels,
                         "text": text, "group_id": groups})
    meta["text_hash"] = meta.text.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    train = groups < 48
    meta["split"] = np.where(train, "train", "test")
    meta["cv_fold"] = -1
    for fold, (_, valid) in enumerate(grouped_cv(labels[train], groups[train], n_splits=3)):
        meta.loc[valid, "cv_fold"] = fold
    vectors = np.random.default_rng(42).normal(size=(120, 4)).astype(np.float32)
    return meta, vectors


class GroupedModelTests(unittest.TestCase):
    def test_csv_roundtrip_allows_repeats_but_rejects_group_leakage(self):
        meta, vectors = sample_metadata()
        with tempfile.TemporaryDirectory() as tmp:
            save_cache(tmp, vectors, meta, source="MOCK_test", is_mock=True,
                       embedding_seconds=0, endpoint_requests=0)
            loaded, metadata, manifest, train, cv = load_cache(tmp)
            np.testing.assert_array_equal(loaded, vectors)
            self.assertTrue(metadata.text.duplicated().any())
            # CSV row order may change; job_id must still align vectors correctly.
            path = Path(tmp) / "embeddings.csv"
            pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
            np.testing.assert_array_equal(load_cache(tmp)[0], vectors)
            # Alter group membership across train/test, with a freshly saved export.
            broken = meta.copy()
            broken.loc[broken.group_id.eq(49), "group_id"] = 0
            save_cache(tmp, vectors, broken, source="MOCK_test", is_mock=True,
                       embedding_seconds=0, endpoint_requests=0)
            with self.assertRaisesRegex(ValueError, "overlap train/test"):
                load_cache(tmp)

    def test_nested_search_uses_grouped_inner_folds(self):
        meta, vectors = sample_metadata()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = save_cache(tmp, vectors, meta, source="MOCK_test", is_mock=True,
                                  embedding_seconds=0, endpoint_requests=0)
            _, _, _, train, cv = load_cache(tmp)
            groups = meta.loc[train, "group_id"].to_numpy()
            with patch("model_comparison.grouped_cv", wraps=grouped_cv) as inner:
                search, table = run_search(
                    "smoke_rf", Pipeline([("clf", RandomForestClassifier(random_state=42))]),
                    {"clf__n_estimators": [5], "clf__max_depth": [2]},
                    vectors[train], meta.loc[train, "fraudulent"].to_numpy(), cv,
                    manifest, tmp, groups=groups,
                )
            self.assertEqual(inner.call_count, 1 + len(cv))
            self.assertEqual(len(table), 1)
            self.assertEqual(table.iloc[0]["evaluation_protocol"], "nested_cv")
            self.assertTrue((Path(tmp) / "comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
