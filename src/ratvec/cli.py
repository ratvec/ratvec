# -*- coding: utf-8 -*-

"""The command line interface for ratvec.

Why does this file exist, and why not put this in ``__main__``?
You might be tempted to import things from ``__main__`` later, but that will cause
problems--the code will get executed twice:

- When you run `python3 -m ratvec` python will execute
  ``__main__.py`` as a script. That means there won't be any
  ``ratvec.__main__`` in ``sys.modules``.
- When you import ``__main__`` it will get executed again (as a module) because
  there's no ``ratvec.__main__`` in ``sys.modules``.

Also see https://click.pocoo.org/latest/setuptools/
"""

import click

from ratvec.evaluation import main as evaluate
from ratvec.protein_seq_reader import main as generate
from ratvec.train import main as train, infer

main = click.Group(commands={
    'generate': generate,
    'train': train,
    'infer': infer,
    'evaluate': evaluate,
})

if __name__ == '__main__':
    main()
