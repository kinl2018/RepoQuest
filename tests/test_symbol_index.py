import ast
from ast_parser import extract_symbols

def test_class_methods():
    code = """
class Student:
    def run(self):
        pass
"""
    tree = ast.parse(code)
    result = extract_symbols(tree)

    assert result == [
        {"name": "Student", "kind": "class"},
        {"name": "Student.run", "kind": "method"}
    ]