import argparse
import csv
from pathlib import Path


METRIC_FILES = {
    "IA": "image_rocauc_results.md",
    "PA": "pixel_rocauc_results.md",
    "PRO": "aupro_results.md",
}


def parse_markdown_table(lines):
    rows = [line.strip() for line in lines if line.strip().startswith("|")]
    if len(rows) < 3:
        return None
    header = [cell.strip() for cell in rows[0].strip("|").split("|")]
    values = [cell.strip() for cell in rows[2].strip("|").split("|")]
    if len(header) != len(values):
        return None
    return dict(zip(header, values))


def parse_result_file(path):
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    results = {}
    note = None
    table_lines = []

    def flush():
        if note and table_lines:
            row = parse_markdown_table(table_lines)
            if row:
                results[note] = row

    for line in text:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|"):
            table_lines.append(line)
            continue
        if table_lines:
            flush()
            table_lines = []
            note = None
        if not stripped.startswith("#"):
            note = stripped

    flush()
    return results


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def should_keep(note, filters):
    if not filters:
        return True
    return any(token in note for token in filters)


def build_summary(results_dir, filters):
    metric_tables = {
        metric: parse_result_file(results_dir / filename)
        for metric, filename in METRIC_FILES.items()
    }
    notes = sorted(set().union(*(table.keys() for table in metric_tables.values())))
    rows = []
    for note in notes:
        if not should_keep(note, filters):
            continue
        row = {"experiment_note": note}
        class_values = {}
        for metric, table in metric_tables.items():
            result = table.get(note, {})
            row[metric] = result.get("Mean", "")
            for key, value in result.items():
                if key in {"Method", "Mean"}:
                    continue
                class_values.setdefault(key, {})[metric] = value
        rows.append((row, class_values))
    return rows


def markdown_table(rows):
    headers = ["experiment_note", "IA", "PA", "PRO"]
    output = ["| " + " | ".join(headers) + " |"]
    output.append("| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |")
    for row, _ in rows:
        output.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(output)


def build_detail_rows(rows):
    detail_rows = []
    for row, class_values in rows:
        for class_name in sorted(class_values):
            detail_rows.append({
                "experiment_note": row["experiment_note"],
                "class": class_name,
                "IA": class_values[class_name].get("IA", ""),
                "PA": class_values[class_name].get("PA", ""),
                "PRO": class_values[class_name].get("PRO", ""),
            })
    return detail_rows


def detail_markdown_table(detail_rows):
    headers = ["experiment_note", "class", "IA", "PA", "PRO"]
    output = ["| " + " | ".join(headers) + " |"]
    output.append("| " + " | ".join(["---", "---"] + ["---:"] * 3) + " |")
    for row in detail_rows:
        output.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(output)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["experiment_note", "IA", "PA", "PRO"])
        writer.writeheader()
        for row, _ in rows:
            writer.writerow(row)


def write_detail_csv(path, detail_rows):
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["experiment_note", "class", "IA", "PA", "PRO"])
        writer.writeheader()
        writer.writerows(detail_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", type=Path)
    parser.add_argument(
        "--filter",
        nargs="*",
        default=["final_"],
        help="Keep experiment notes containing any of these tokens. Use --filter with no values to show all.",
    )
    parser.add_argument("--out_md", default="results/thesis_final_summary.md", type=Path)
    parser.add_argument("--out_csv", default="results/thesis_final_summary.csv", type=Path)
    parser.add_argument("--out_detail_md", default="results/thesis_final_detail.md", type=Path)
    parser.add_argument("--out_detail_csv", default="results/thesis_final_detail.csv", type=Path)
    parser.add_argument("--summary_only", default=False, action="store_true",
                        help="Print only the mean summary table.")
    parser.add_argument("--detail_only", default=False, action="store_true",
                        help="Print only the per-class detail table.")
    args = parser.parse_args()

    rows = build_summary(args.results_dir, args.filter)
    table = markdown_table(rows)
    detail_rows = build_detail_rows(rows)
    detail_table = detail_markdown_table(detail_rows)

    if not args.detail_only:
        print("## Mean Summary")
        print(table)
    if not args.summary_only:
        if not args.detail_only:
            print()
        print("## Per-Class Detail")
        print(detail_table)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(table + "\n", encoding="utf-8")
    write_csv(args.out_csv, rows)
    args.out_detail_md.write_text(detail_table + "\n", encoding="utf-8")
    write_detail_csv(args.out_detail_csv, detail_rows)
    print(f"\nSaved: {args.out_md}")
    print(f"Saved: {args.out_csv}")
    print(f"Saved: {args.out_detail_md}")
    print(f"Saved: {args.out_detail_csv}")


if __name__ == "__main__":
    main()
