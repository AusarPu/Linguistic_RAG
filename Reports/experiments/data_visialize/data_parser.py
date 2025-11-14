"""
Data Parser for RAG System Evaluation Results
CSV-based loader from runs directory
"""

import os
from typing import Dict, List, Any
import pandas as pd


class RAGDataParser:
    def __init__(self, base_dir: str = "/home/pushihao/RAG/Reports/experiments/datasets/runs"):
        self.base_dir = base_dir
        self.datasets = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']

    def get_dataset_labels(self) -> Dict[str, str]:
        return {
            'hotpotqa': 'HotpotQA',
            'ms_marco': 'MS MARCO',
            'natural_questions': 'Natural Questions',
            'triviaqa': 'TriviaQA'
        }

    def get_run_labels(self) -> Dict[str, str]:
        return {
            'result_1_chunk_only': 'Chunk Only',
            'result_2_chunk+question': 'Chunk+Question',
            'result_3_chunk+keyword': 'Chunk+Keyword',
            'result_4_chunk+question+keyword': 'Chunk+Question+Keyword',
            'result_5_full': 'Full System',
            'result_5_1': 'Full System v1',
            'result_6_bm25': 'BM25'
        }

    def load_run_summary(self, run_name: str) -> Dict[str, Any]:
        summary_path = os.path.join(self.base_dir, run_name, 'advanced_evaluation_results', 'ragas_summary.csv')
        df = pd.read_csv(summary_path)
        total_questions = int(df['total_questions'].sum())
        w = df['total_questions']
        overall_accuracy = float((df['accuracy'] * w).sum() / total_questions)
        overall_context_recall = float((df['context_recall'] * w).sum() / total_questions)
        overall_context_precision = float((df['context_precision'] * w).sum() / total_questions)
        overall_answer_relevancy = float((df['answer_relevancy'] * w).sum() / total_questions)
        overall_faithfulness = float((df['faithfulness'] * w).sum() / total_questions)
        datasets: Dict[str, Any] = {}
        for _, row in df.iterrows():
            d = str(row['dataset'])
            datasets[d] = {
                'accuracy': float(row['accuracy']),
                'context_recall': float(row['context_recall']),
                'context_precision': float(row['context_precision']),
                'answer_relevancy': float(row['answer_relevancy']),
                'faithfulness': float(row['faithfulness']),
                'total_questions': int(row['total_questions'])
            }
        return {
            'run_name': run_name,
            'overall': {
                'overall_accuracy': overall_accuracy,
                'overall_context_recall': overall_context_recall,
                'overall_context_precision': overall_context_precision,
                'overall_answer_relevancy': overall_answer_relevancy,
                'overall_faithfulness': overall_faithfulness,
                'total_questions': total_questions
            },
            'datasets': datasets
        }

    def load_all_runs(self) -> Dict[str, Any]:
        runs: Dict[str, Any] = {}
        items = sorted(os.listdir(self.base_dir))
        for item in items:
            if not item.startswith('result_'):
                continue
            run_dir = os.path.join(self.base_dir, item, 'advanced_evaluation_results')
            csv_path = os.path.join(run_dir, 'ragas_summary.csv')
            if not os.path.isfile(csv_path):
                continue
            runs[item] = self.load_run_summary(item)
        return runs

    def create_comparison_dataframe(self, runs_data: Dict[str, Any]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for run_name, data in runs_data.items():
            overall = data['overall']
            row: Dict[str, Any] = {
                'run': run_name,
                'total_questions': overall['total_questions'],
                'overall_accuracy': overall['overall_accuracy'],
                'overall_context_recall': overall['overall_context_recall'],
                'overall_context_precision': overall['overall_context_precision'],
                'overall_answer_relevancy': overall['overall_answer_relevancy'],
                'overall_faithfulness': overall['overall_faithfulness']
            }
            for d in self.datasets:
                if d in data['datasets']:
                    row[f'{d}_accuracy'] = data['datasets'][d]['accuracy']
                    row[f'{d}_context_recall'] = data['datasets'][d]['context_recall']
                else:
                    row[f'{d}_accuracy'] = 0.0
                    row[f'{d}_context_recall'] = 0.0
            rows.append(row)
        return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = RAGDataParser()
    all_runs = parser.load_all_runs()
    df = parser.create_comparison_dataframe(all_runs)
    print(df.to_string())