import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SearXNGDeploymentTest(unittest.TestCase):
    def test_compose_only_publishes_search_on_loopback(self) -> None:
        compose_path = PROJECT_ROOT / "deploy" / "searxng" / "compose.yaml"
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        service = compose["services"]["searxng"]

        self.assertEqual(service["ports"], ["127.0.0.1:8080:8080"])
        self.assertIn("SEARXNG_SECRET", service["environment"])
        self.assertNotIn("privileged", service)

    def test_settings_enable_bounded_json_api_for_local_client(self) -> None:
        settings_path = PROJECT_ROOT / "deploy" / "searxng" / "settings.yml"
        settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))

        self.assertTrue(settings["use_default_settings"])
        self.assertIn("json", settings["search"]["formats"])
        self.assertEqual(settings["search"]["safe_search"], 1)
        self.assertFalse(settings["server"]["limiter"])
        self.assertFalse(settings["server"]["public_instance"])


if __name__ == "__main__":
    unittest.main()
