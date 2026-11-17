import tempfile
import unittest
from pathlib import Path
from audit_micromix_port import compare, function_body_source, tokens


class BodyAudit(unittest.TestCase):
    def test_function_body_extraction_ignores_layout(self):
        source = "static inline\nint legal(\n int x\n) /* layout */ { return x >= 1 && x <= 32; }"
        body = function_body_source(source, "legal")
        self.assertEqual(tokens(body), tokens("{ return x >= 1 && x <= 32; }"))
        self.assertIn("&&", body)
        with self.assertRaises(ValueError):
            function_body_source(source, "missing")

    def test_instruction_and_loop_mutations(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "a.c"
            b = Path(d) / "b.c"
            original = 'static int f(void){asm volatile("fence":::"memory");for(int i=0;i<8;i++){}return 1;}'
            a.write_text(original)
            for mutated in (
                original.replace('"fence"', '"nop"'),
                original.replace("i<8", "i<4"),
                "static int g(void){return 1;}",
            ):
                b.write_text(mutated)
                with self.assertRaises(ValueError):
                    compare(a, b)
            b.write_text("/* portable comment */\n" + original.replace("return 1;", "return   1 ;"))
            self.assertEqual(compare(a, b)["count"], 1)


if __name__ == "__main__":
    unittest.main()
