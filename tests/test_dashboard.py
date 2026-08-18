import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import APIRouter
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DashboardRouteTests(unittest.TestCase):
    def setUp(self):
        self._module_patcher = patch.dict(
            sys.modules,
            {
                "mcp": type("_McpStub", (), {"mcp_router": APIRouter()}),
                "api": type("_ApiStub", (), {"api_router": APIRouter()}),
            },
        )
        self._module_patcher.start()

        if "app" in sys.modules:
            del sys.modules["app"]

        self.app_module = importlib.import_module("app")
        self.client = TestClient(self.app_module.app)

    def tearDown(self):
        self._module_patcher.stop()

    def test_dashboard_route_serves_html(self):
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("<h1>Local RAG Server</h1>", response.text)

    def test_dashboard_html_uses_source_path_delete_endpoint(self):
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/api/document?source_path=", response.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
