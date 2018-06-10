#-----------------------------------------------------------------
# pycparser: dump_ast.py
#
# Basic example of parsing a file and dumping its parsed AST.
#
# Eli Bendersky [http://eli.thegreenplace.net]
# License: BSD
#-----------------------------------------------------------------
from __future__ import print_function
import argparse
import sys

from pycparser import c_parser, c_ast, parse_file

if __name__ == "__main__":
    argparser = argparse.ArgumentParser('Dump AST')
    argparser.add_argument('filename', help='name of file to parse')
    args = argparser.parse_args()

    ast = parse_file(args.filename, use_cpp=False)
    ast.show()
    # cv = MyVisitor()
    # cv.visit(ast)
    # print(cv.values)
