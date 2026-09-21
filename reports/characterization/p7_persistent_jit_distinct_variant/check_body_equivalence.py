"""Reject arithmetic drift between isolated serial and parallel variants."""
import ast
from pathlib import Path

source = Path(__file__).parent / "src/solweig_light/radiation/patch_radiation.py"
tree = ast.parse(source.read_text())
for name in ("_shortwave", "_longwave"):
    functions = []
    for target in (name, name + "_serial"):
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == target)
        function.name = "normalized"
        function.decorator_list = []
        for node in ast.walk(function):
            if isinstance(node, ast.Name) and node.id == "prange":
                node.id = "range"
        functions.append(ast.dump(function, include_attributes=False))
    assert functions[0] == functions[1], f"arithmetic drift in {name}"
print("Both serial/parallel bodies match after scheduling-only normalization.")
