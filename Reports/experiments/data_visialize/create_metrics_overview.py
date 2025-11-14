import os
import matplotlib.pyplot as plt
import seaborn as sns

def create_metrics_overview(output_dir: str, output_name: str) -> str:
    sns.set_palette("husl")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    rows = [
        ["context_precision", "Retrieval", "Proportion of relevant content in retrieved context"],
        ["context_recall", "Retrieval", "Degree to which necessary evidence is covered"],
        ["faithfulness", "Generation", "Consistency of the answer with supporting evidence"],
        ["answer_relevancy", "Generation", "Alignment between the answer and the question"],
        ["accuracy", "End-to-End", "Semantic correctness of final answers (LLM-as-Judge)"]
    ]
    cols = ["Metric", "Scope", "Definition"]
    table_data = [cols] + rows
    cell_text = table_data
    table = ax.table(cellText=cell_text, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path

def main():
    output_dir = "/home/pushihao/RAG/Reports/docs/pics"
    output_name = "metrics_overview.png"
    p = create_metrics_overview(output_dir, output_name)
    print(p)

if __name__ == "__main__":
    main()