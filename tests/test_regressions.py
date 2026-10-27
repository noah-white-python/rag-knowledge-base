"""Regression checks that do not load models or call the API."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chunker
import config


class ChunkingTests(unittest.TestCase):
    def test_full_size_pieces_do_not_exceed_body_limit(self):
        text = "a" * 400 + "b" * 400
        with patch.object(config, "CHUNK_SIZE", 400), patch.object(config, "CHUNK_OVERLAP", 80):
            chunks = chunker.chunk_document({"source": "sample.txt", "text": text})
        bodies = [chunk["text"].split("\n", 1)[1] for chunk in chunks]
        self.assertTrue(all(len(body) <= 400 for body in bodies))
        self.assertEqual("".join(bodies), text)

    def test_zero_overlap_does_not_copy_previous_chunk(self):
        pieces = ["a" * 250, "b" * 200]
        with patch.object(config, "CHUNK_SIZE", 400), patch.object(config, "CHUNK_OVERLAP", 0):
            self.assertEqual(chunker._merge_pieces(pieces), pieces)

    def test_overlap_uses_remaining_capacity(self):
        with patch.object(config, "CHUNK_SIZE", 400), patch.object(config, "CHUNK_OVERLAP", 80):
            chunks = chunker._merge_pieces(["a" * 300, "b" * 350])
        self.assertEqual(chunks, ["a" * 300, "a" * 50 + "b" * 350])


class CLITests(unittest.TestCase):
    def test_single_question_with_optional_checks(self):
        pipeline = ModuleType("pipeline")
        pipeline.RAGPipeline = Mock()
        pipeline.low_confidence = Mock(return_value=False)
        llm = ModuleType("llm")
        llm.verify_answer = Mock(return_value={"supported": True})
        spec = importlib.util.spec_from_file_location(
            "query_under_test", Path(__file__).resolve().parents[1] / "src" / "query.py"
        )
        query = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"pipeline": pipeline, "llm": llm}):
            spec.loader.exec_module(query)
        pipe = pipeline.RAGPipeline.return_value
        for flags in ([], ["--crag", "--check"]):
            with self.subTest(flags=flags):
                pipe.ask.reset_mock()
                llm.verify_answer.reset_mock()
                pipe.ask.return_value = ("question", [{"source": "sample.txt"}], iter(["answer"]))
                with patch.object(sys, "argv", ["query.py", "question", *flags]), \
                     patch.object(config, "DEEPSEEK_API_KEY", "test-key"), \
                     patch.object(config, "CHUNKS_FILE") as chunks_file, \
                     patch.object(query, "print_sources"), patch("builtins.print"):
                    chunks_file.exists.return_value = True
                    query.main()
                pipe.ask.assert_called_once_with("question", [], use_crag="--crag" in flags)
                self.assertEqual(llm.verify_answer.call_count, int("--check" in flags))


if __name__ == "__main__":
    unittest.main()
