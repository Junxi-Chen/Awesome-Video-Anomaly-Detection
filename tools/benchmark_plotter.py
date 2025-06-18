import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.font_manager import FontProperties

font_prop = FontProperties(family='Times New Roman')

with open('benchmarks/benchmark.yaml', 'r') as f:
    data = yaml.safe_load(f)

all_keys = set()
for entry in data:
    all_keys.update(entry.keys())

dataset_fields = sorted([
    key for key in all_keys
    if '(AUC)' in key or '(AP)' in key and any(char.isdigit() for char in str(entry.get(key, '')))
])

records = []
for entry in data:
    method = entry.get('Method', 'Unknown')
    supervision = entry.get('Supervision', 'Unknown')
    publication = entry.get('Publication', '')
    year = publication.split(
        "'")[-1].split(" ")[0] if "'" in publication else ''
    method_label = f"{method} ('{year})" if year else method
    for field in dataset_fields:
        value = entry.get(field, '-')
        if value != '-' and isinstance(value, (int, float, str)) and str(value).replace('.', '', 1).isdigit():
            dataset_name = field.split()[0]
            records.append({
                'Dataset': dataset_name,
                'Score': float(value),
                'Method': method,
                'Method_Label': method_label,
                'Supervision': supervision
            })

df = pd.DataFrame(records)

sns.set(style="whitegrid")
plt.figure(figsize=(14, 6))

dataset_order = df['Dataset'].unique().tolist()
df['Dataset'] = pd.Categorical(
    df['Dataset'], categories=dataset_order, ordered=True)
x_map = {name: i for i, name in enumerate(dataset_order)}

methods = sorted(df['Method'].unique().tolist())
palette = sns.color_palette("tab20", len(methods))
method_to_color = dict(zip(methods, palette))

markers = ['o', 's', 'D', 'd', '^', 'p', 'P', 'X', '*', 'H', 'h']
while len(markers) < len(methods):
    markers += markers
method_to_marker = dict(zip(methods, markers))

hue_order = methods
style_order = methods

idx_max = df.groupby(['Supervision', 'Dataset'],
                     observed=True)['Score'].idxmax()
df_max = df.loc[idx_max]
df_others = df.drop(idx_max)

ax = sns.scatterplot(
    data=df_others, x='Dataset', y='Score',
    hue='Method', style='Method', s=100, ax=None,
    hue_order=hue_order, style_order=style_order,
    palette=method_to_color, markers=method_to_marker
)

sns.scatterplot(
    data=df_max, x='Dataset', y='Score',
    hue='Method', style='Method', s=200, ax=ax,
    hue_order=hue_order, style_order=style_order,
    palette=method_to_color, markers=method_to_marker,
    edgecolor='black', linewidth=1.5, legend=False,
    zorder=10
)

texts = []
for _, row in df_max.iterrows():
    x = x_map[row['Dataset']] + 0.15
    y = row['Score']
    label = row['Method_Label']
    texts.append(ax.text(x, y, label, ha='left', va='center', fontsize=10,
                         color='black', fontproperties=font_prop))

ax.set_xticks(range(len(dataset_order)))
ax.set_xticklabels(dataset_order, ha='right', fontproperties=font_prop)
ax.set_yticklabels(ax.get_yticks(), fontproperties=font_prop)
ax.set_xlabel("Dataset", fontproperties=font_prop, fontsize=18)
ax.set_ylabel("Metric", fontproperties=font_prop, fontsize=18)

ax.set_xlim(-0.5, len(dataset_order) - 0.3)
plt.title("VAD benchmark leaderboard", fontproperties=font_prop, fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(ha='center', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.5)

plt.legend(prop=font_prop)

plt.tight_layout()
plt.savefig('pics/benchmark.png', dpi=800, bbox_inches='tight')
