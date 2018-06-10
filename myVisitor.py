from __future__ import print_function
import argparse
import sys

from pycparser import c_parser, c_ast, parse_file


class MyVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.values = []
    def visit_Decl(self, node):
        self.values.append(node.name)
        self.generic_visit(node)
    def visit_For(self, node):
        self.values.append("for")
        self.generic_visit(node)
    def visit_If(self, node):
        self.values.append("if")
        self.generic_visit(node)
    def visit_While(self, node):
        self.values.append("while")
        self.generic_visit(node)
    def visit_Switch(self, node):
        self.values.append('switch')
        self.generic_visit(node)
    def visit_Case(self, node):
        self.values.append('case')
        self.generic_visit(node)
    def visit_Break(self, node):
        self.values.append("break")
    
    def visit_FuncCall(self, node):
        #self.values.append(node.name.name)
        self.generic_visit(node)
    def visit_UnaryOp(self, node):
        self.values.append(node.op)
        self.generic_visit(node)
    def visit_Assignment(self, node):
        self.values.append("=")
        self.generic_visit(node)
    def visit_Return(self, node):
        self.values.append("return")
        self.generic_visit(node)
    def visit_BinaryOp(self, node):
        self.values.append(node.op)
        self.generic_visit(node)

    def visit_IdentifierType(self, node):
        self.values = self.values + node.names
    def visit_ID(self, node):
        self.values.append(node.name)
    def visit_Constant(self, node):
        #self.values.append(node.type)
        self.values.append(node.value)
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser('Dump AST')
    argparser.add_argument('filename', help='name of file to parse')
    args = argparser.parse_args()

    ast = parse_file(args.filename, use_cpp=False)
    #ast.show()
    cv = MyVisitor()
    cv.visit(ast)
    print(cv.values)
