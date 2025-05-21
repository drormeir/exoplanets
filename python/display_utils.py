import IPython.display as display
from IPython import get_ipython
import pandas as pd

use_markdown = False
if 'get_ipython' in globals():
    ipython = get_ipython()
    if ipython is not None and 'IPKernelApp' in ipython.config:
        use_markdown = True


def display_with_title(df : pd.DataFrame, title : str):
    print()
    if use_markdown:
        display.display(display.Markdown(f"### {title}"))
    else:
        print(f"{title}\n" + "-"*len(title))
    display.display(df)
