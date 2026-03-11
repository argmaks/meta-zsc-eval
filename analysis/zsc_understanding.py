# %%
import pandas as pd
from utils import read_experiment_to_dataframe
from IPython.display import display, Markdown, Latex
# %%
pd.set_option('display.max_columns', None)
# %%
df = read_experiment_to_dataframe("explain_zsc_methods")
# %%
df.head()
# %%
completed_df = df.loc[df.results_missing!="True"]
# %%
for idx, row in completed_df.iterrows():
    print(f"Sample {idx+1}:")
    print(f"Model: {row.get('config_runner_model', 'N/A')}")
    print("Result Response:")
    if pd.isna(row.get('result_response')):
        response = row.get('results_runner_response')
    else:
        response = row.get('result_response')
    display(Markdown(response if response else 'No response found'))
    print("-" * 40)

# %%
completed_df.groupby(["config_prompt_contents_zsc_method", "config_runner_model", ]).size()
# %%
