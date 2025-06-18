import yaml
from collections import defaultdict


def yaml_to_html_table(yaml_file):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    headers = list(data[0].keys())
    visual_metrics = [
        h for h in headers if h not in [
            "Supervision", "Method", "Publication", "Visual Features",
            "Audio Features", "Text Prompt", "LLM Involved", "Data Augumentation"
        ]
    ]

    supervision_groups = defaultdict(list)
    for row in data:
        supervision_groups[row["Supervision"]].append(row)

    html = "<table>\n  <tr>\n"
    for h in headers:
        html += f"    <th>{h}</th>\n"
    html += "  </tr>\n"

    for supervision, rows in supervision_groups.items():
        method_groups = defaultdict(list)
        for row in rows:
            key = (row["Method"], row["Publication"])
            method_groups[key].append(row)
        supervision_rowspan = sum(len(v) for v in method_groups.values())
        supervision_written = False

        for (method, publication), row_group in method_groups.items():
            method_rowspan = len(row_group)
            method_written = False

            for row in row_group:
                html += "  <tr>\n"

                if not supervision_written:
                    html += f'    <td rowspan="{supervision_rowspan}">{supervision}</td>\n'
                    supervision_written = True

                if not method_written:
                    html += f'    <td rowspan="{method_rowspan}">{method}</td>\n'
                    html += f'    <td rowspan="{method_rowspan}">{publication}</td>\n'
                    method_written = True

                html += f'    <td>{row["Visual Features"]}</td>\n'

                for h in visual_metrics:
                    html += f'    <td>{row[h]}</td>\n'

                for h in ["Audio Features", "Text Prompt", "LLM Involved", "Data Augumentation"]:
                    html += f'    <td>{row[h]}</td>\n'

                html += "  </tr>\n"

    html += "</table>\n"
    return html


html_output = yaml_to_html_table("benchmarks/benchmark.yaml")
with open("output.html", "w") as f:
    f.write(html_output)
