"""Offline tests: section routing and passage ranking. Run: python -m unittest discover -s tests"""
import unittest

from secfeed_langchain import core


class Routing(unittest.TestCase):
    def test_sections(self):
        self.assertEqual(core._section_for("What are the biggest risks?"), "risk")
        self.assertEqual(core._section_for("Who are its main competitors?"), "business")
        self.assertEqual(core._section_for("How exposed is it to interest rate changes?"), "market_risk")
        self.assertEqual(core._section_for("How did revenue grow?"), "mdna")


class Passages(unittest.TestCase):
    def test_keeps_matching_paragraphs_in_order(self):
        filler = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore. " * 3
        text = "\n\n".join([filler, "Export controls on China reduced our data center revenue in the quarter. " * 2,
                            filler, "Tariffs and export controls may change again and hurt China sales. " * 2])
        out = core.best_passages(text, "export controls China", max_chars=400)
        self.assertIn("reduced our data center", out)
        self.assertIn("Tariffs", out)
        self.assertLess(out.index("reduced"), out.index("Tariffs"))
        self.assertNotIn("Lorem", out)

    def test_money(self):
        self.assertEqual(core._money("215938000000"), "215.94B")
        self.assertEqual(core._money(5000), "5.00K")


if __name__ == "__main__":
    unittest.main()
