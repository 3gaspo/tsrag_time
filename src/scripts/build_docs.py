"""Validate public views and optionally compile the project PDFs."""

import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('method_overview', 'experiment_guideline', 'executive_summary')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', choices=('method', 'protocol', 'all'))
    args = parser.parse_args()
    required = ['README.md', 'docs/architecture.md', 'docs/experiment_catalog.md', 'docs/results_recap.md']
    required += [f'latex/{name}.{extension}' for name in SOURCES for extension in ('tex',)]
    missing = [name for name in required if not (ROOT/name).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.render:
        pdflatex = shutil.which('pdflatex')
        if pdflatex is None:
            pdflatex = r'C:\Users\Gaspard\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe'
        names = (SOURCES if args.render == 'all' else ('method_overview', 'experiment_guideline')
                 if args.render == 'protocol' else ('method_overview',))
        for name in names:
            build = ROOT/'outputs/documentation_build'/name
            build.mkdir(parents=True, exist_ok=True)
            for repeat in range(2):
                result = subprocess.run([pdflatex, '-interaction=nonstopmode', '-halt-on-error',
                                         f'-output-directory={build}', str(ROOT/f'latex/{name}.tex')],
                                        cwd=ROOT/'latex', capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(result.stdout[-4000:])
            shutil.copy2(build/f'{name}.pdf', ROOT/f'latex/{name}.pdf')
            print(f'Rendered latex/{name}.pdf')
    for name in SOURCES:
        if not (ROOT/f'latex/{name}.pdf').is_file():
            raise FileNotFoundError(f'latex/{name}.pdf')
    forbidden = list((ROOT/'latex').glob('*.aux')) + list((ROOT/'latex').glob('*.log')) + list((ROOT/'latex').glob('*.out'))
    if forbidden:
        raise ValueError(f'Stale LaTeX build files: {forbidden}')
    print('Public documentation views present.')


if __name__ == '__main__':
    main()
