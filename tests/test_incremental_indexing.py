import importlib
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeDoc:
    def __init__(self, page_content: str, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class _FakeTextLoader:
    def __init__(self, path: str, encoding: str | None = None):
        self.path = path
        self.encoding = encoding

    def load(self):
        text = Path(self.path).read_text(encoding=self.encoding or "utf-8")
        return [_FakeDoc(page_content=text, metadata={})]


class _FakeMarkdownLoader(_FakeTextLoader):
    pass


class _FakePdfLoader(_FakeTextLoader):
    pass


class _FakeSplitter:
    def __init__(self, chunk_size=800, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, docs):
        chunks = []
        for doc in docs:
            text = doc.page_content
            if not text:
                continue
            chunks.append(_FakeDoc(page_content=text, metadata=dict(doc.metadata or {})))
        return chunks


class _FakeEmbeddings:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def _to_vec(self, text: str):
        seed = abs(hash(text)) % 10_000
        a = (seed % 97) / 97.0
        b = (seed % 89) / 89.0
        c = (seed % 83) / 83.0
        d = (seed % 79) / 79.0
        return [a, b, c, d]

    def embed_documents(self, texts):
        return [self._to_vec(t) for t in texts]

    def embed_query(self, text):
        return self._to_vec(text)


class _FakeIndexIDMap2:
    def __init__(self, base=None):
        self._vectors = {}

    @property
    def ntotal(self):
        return len(self._vectors)

    def add_with_ids(self, vectors, ids):
        for vec, idx in zip(vectors, ids):
            self._vectors[int(idx)] = np.array(vec, dtype="float32")

    def remove_ids(self, ids_array):
        removed = 0
        for idx in ids_array:
            idx = int(idx)
            if idx in self._vectors:
                removed += 1
                del self._vectors[idx]
        return removed

    def search(self, query_vector, k):
        q = np.array(query_vector[0], dtype="float32")
        ranked = []
        for idx, vec in self._vectors.items():
            ranked.append((float(np.dot(q, vec)), int(idx)))
        ranked.sort(key=lambda item: item[0], reverse=True)

        top = ranked[:k]
        scores = np.array([[item[0] for item in top]], dtype="float32")
        ids = np.array([[item[1] for item in top]], dtype="int64")

        if len(top) < k:
            pad = k - len(top)
            scores = np.concatenate([scores, np.full((1, pad), -1.0, dtype="float32")], axis=1)
            ids = np.concatenate([ids, np.full((1, pad), -1, dtype="int64")], axis=1)

        return scores, ids


class _FakeFaiss(types.SimpleNamespace):
    class IndexFlatIP:
        def __init__(self, dim):
            self.dim = dim

    @staticmethod
    def IndexIDMap2(base):
        return _FakeIndexIDMap2(base)

    @staticmethod
    def write_index(index, path):
        payload = {int(k): v.tolist() for k, v in index._vectors.items()}
        Path(path).write_text(repr(payload), encoding="utf-8")

    @staticmethod
    def read_index(path):
        idx = _FakeIndexIDMap2()
        p = Path(path)
        if p.exists():
            data = eval(p.read_text(encoding="utf-8"), {}, {})
            for k, v in data.items():
                idx._vectors[int(k)] = np.array(v, dtype="float32")
        return idx


class IncrementalIndexingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="rag-incremental-tests-")
        self._cwd = os.getcwd()
        os.chdir(self._tmp)

        self._module_patcher = patch.dict(
            sys.modules,
            {
                "faiss": _FakeFaiss(),
                "langchain_text_splitters": types.SimpleNamespace(
                    RecursiveCharacterTextSplitter=_FakeSplitter
                ),
                "langchain_community.document_loaders": types.SimpleNamespace(
                    TextLoader=_FakeTextLoader,
                    UnstructuredMarkdownLoader=_FakeMarkdownLoader,
                    PyPDFLoader=_FakePdfLoader,
                ),
                "rag.embeddings": types.SimpleNamespace(
                    embeddings=_FakeEmbeddings(dim=4),
                    cleanup_after_embedding=lambda: None,
                ),
            },
        )
        self._module_patcher.start()

        for mod_name in ["rag.index", "rag.docHandlers"]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        os.environ["DEFAULT_FAISS_DIM"] = "4"
        self.index = importlib.import_module("rag.index")

    def tearDown(self):
        self._module_patcher.stop()
        os.chdir(self._cwd)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _write_file(self, rel_path: str, content: str) -> str:
        path = Path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path.resolve())

    def test_add_update_delete_flow_is_incremental(self):
        source_path = self._write_file("external/policy.txt", "first content")

        add_result = self.index.upsert_file_to_index(source_path)
        self.assertEqual(add_result.get("status"), "success")
        self.assertEqual(add_result.get("operation"), "add")
        self.assertGreater(add_result.get("added_chunks", 0), 0)
        self.assertEqual(self.index.get_chunk_count(), add_result.get("indexed_chunks"))

        noop_result = self.index.upsert_file_to_index(source_path)
        self.assertEqual(noop_result.get("status"), "success")
        self.assertTrue(noop_result.get("skipped"))
        self.assertEqual(noop_result.get("operation"), "noop")

        Path(source_path).write_text("updated content", encoding="utf-8")
        update_result = self.index.upsert_file_to_index(source_path)
        self.assertEqual(update_result.get("status"), "success")
        self.assertEqual(update_result.get("operation"), "update")
        self.assertGreaterEqual(update_result.get("removed_chunks", 0), 1)
        self.assertGreaterEqual(update_result.get("added_chunks", 0), 1)

        delete_result = self.index.delete_document_by_source_path(source_path)
        self.assertEqual(delete_result.get("status"), "success")
        self.assertEqual(delete_result.get("operation"), "delete")
        self.assertGreaterEqual(delete_result.get("chunks_removed", 0), 1)

    def test_filename_delete_reports_ambiguity(self):
        p1 = self._write_file("a/same.txt", "doc one")
        p2 = self._write_file("b/same.txt", "doc two")

        self.index.upsert_file_to_index(p1)
        self.index.upsert_file_to_index(p2)

        result = self.index.delete_document_by_filename("same.txt")
        self.assertEqual(result.get("status"), "ambiguous")
        self.assertEqual(len(result.get("matches", [])), 2)

    def test_catalog_crud_roundtrip(self):
        from rag import catalog

        catalog_path = Path("vectorstore/test_catalog.db")
        catalog.init_catalog(catalog_path)

        source_path = self._write_file("source/doc.txt", "body")
        catalog.upsert_document(
            catalog_path,
            source_path=source_path,
            source_hash="hash1",
            mtime=1.0,
            size=4,
            chunk_count=2,
        )
        doc = catalog.get_document(catalog_path, source_path)
        self.assertIsNotNone(doc)
        self.assertEqual(doc["source_hash"], "hash1")

        ids = catalog.next_chunk_ids(catalog_path, 2)
        self.assertEqual(len(ids), 2)
        catalog.insert_chunk_rows(
            catalog_path,
            [
                {
                    "chunk_id": ids[0],
                    "source_path": source_path,
                    "chunk_order": 0,
                    "chunk_hash": "c1",
                    "text": "chunk one",
                    "metadata": {"source": source_path, "chunk_id": ids[0]},
                },
                {
                    "chunk_id": ids[1],
                    "source_path": source_path,
                    "chunk_order": 1,
                    "chunk_hash": "c2",
                    "text": "chunk two",
                    "metadata": {"source": source_path, "chunk_id": ids[1]},
                },
            ],
        )

        chunk_ids = catalog.get_chunk_ids_for_source_path(catalog_path, source_path)
        self.assertEqual(chunk_ids, ids)

        docs = catalog.list_documents(catalog_path)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["filename"], "doc.txt")

        catalog.delete_document_and_chunks(catalog_path, source_path)
        self.assertIsNone(catalog.get_document(catalog_path, source_path))
        self.assertEqual(catalog.get_chunk_ids_for_source_path(catalog_path, source_path), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
