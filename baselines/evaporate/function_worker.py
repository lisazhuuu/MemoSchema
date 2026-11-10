"""Runs ONLY inside a networkless, credential-free container. Input/output: JSON."""
import ast
import contextlib
import io
import json
import math
import os
import pickle
from pathlib import Path
import re
import random
import signal
import sys
import time
from collections import Counter, defaultdict


def main():
    payload = json.load(sys.stdin)
    ns = dict(os=os, re=re, signal=signal, contextmanager=contextlib.contextmanager,
              json=json, math=math, pickle=pickle, random=random, sys=sys,
              time=time, Counter=Counter, defaultdict=defaultdict)
    # Generated functions in the author implementation see these common globals.
    import numpy as np
    import pandas as pd
    from bs4 import BeautifulSoup
    ns.update(np=np, pd=pd, BeautifulSoup=BeautifulSoup)
    tree = ast.parse((Path(__file__).parent / 'full_upstream/profiler.py').read_text())
    names = {'TimeoutException', 'time_limit', 'get_function_field_from_attribute',
             'deduplicate_extractions', 'apply_final_profiling_functions'}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'author_worker', 'exec'), ns)
    with contextlib.redirect_stdout(io.StringIO()):
        values, errors = ns['apply_final_profiling_functions'](
            payload['contents'], payload['papers'], payload['function'], payload['attribute'], function_cache=False)
    print(json.dumps({'values': values, 'errors': errors}, default=str))


if __name__ == '__main__':
    main()
