"""
Data Parser for RAG System Evaluation Results
Extracts metrics from evaluation_summary.txt and advanced_sample_results.json files
"""

import json
import re
import os
from typing import Dict, List, Any, Optional
import pandas as pd


class RAGDataParser:
    """Parser for RAG evaluation data"""
    
    def __init__(self, base_dir: str = "/home/pushihao/RAG/Reports/experiments/datasets"):
        self.base_dir = base_dir
        self.datasets = ['hotpotqa', 'ms_marco', 'natural_questions', 'triviaqa']
        
    def parse_summary_file(self, file_path: str) -> Dict[str, Any]:
        """Parse evaluation_summary.txt file"""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract overall metrics
            overall_metrics = {}
            
            # Total questions
            match = re.search(r'总问题数:\s*(\d+)', content)
            if match:
                overall_metrics['total_questions'] = int(match.group(1))
            
            # Overall accuracy
            match = re.search(r'总体准确率:\s*([\d.]+)%', content)
            if match:
                overall_metrics['overall_accuracy'] = float(match.group(1)) / 100
            
            # Overall retrieval success rate
            match = re.search(r'总体检索成功率:\s*([\d.]+)%', content)
            if match:
                overall_metrics['overall_retrieval_success_rate'] = float(match.group(1)) / 100
            
            # Overall F1 score
            match = re.search(r'总体F1分数:\s*([\d.]+)', content)
            if match:
                overall_metrics['overall_f1_score'] = float(match.group(1))
            
            # Average chunks per question
            match = re.search(r'平均检索块数:\s*([\d.]+)', content)
            if match:
                overall_metrics['avg_chunks_per_question'] = float(match.group(1))
            
            # Dataset-specific metrics
            dataset_metrics = {}
            
            for dataset in self.datasets:
                dataset_upper = dataset.upper().replace('_', '_')
                
                # Find dataset section
                pattern = rf'{dataset_upper}:\s*\n\s*-\s*准确率:\s*([\d.]+)%\s*\n\s*-\s*检索成功率:\s*([\d.]+)%\s*\n\s*-\s*F1分数:\s*([\d.]+)'
                match = re.search(pattern, content)
                
                if match:
                    dataset_metrics[dataset] = {
                        'accuracy': float(match.group(1)) / 100,
                        'retrieval_success_rate': float(match.group(2)) / 100,
                        'f1_score': float(match.group(3))
                    }
            
            return {
                'overall': overall_metrics,
                'datasets': dataset_metrics,
                'status': 'success'
            }
            
        except Exception as e:
            return {"error": f"Error parsing summary file: {str(e)}"}
    
    def parse_detailed_results(self, file_path: str) -> Dict[str, Any]:
        """Parse advanced_sample_results.json file"""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                return {"error": "Expected list format in JSON file"}
            
            # Calculate metrics
            total_questions = len(data)
            correct_answers = 0
            successful_retrievals = 0
            total_chunks = 0
            confidence_scores = []
            
            for item in data:
                # Answer correctness
                if 'answer_correctness' in item and item['answer_correctness'].get('is_correct', False):
                    correct_answers += 1
                
                # Retrieval success
                if 'retrieval_accuracy' in item and item['retrieval_accuracy'].get('is_retrieved', False):
                    successful_retrievals += 1
                
                # Chunk count
                if 'retrieved_chunk_ids' in item:
                    total_chunks += len(item['retrieved_chunk_ids'])
                
                # Confidence scores
                if 'answer_correctness' in item and 'confidence' in item['answer_correctness']:
                    confidence_scores.append(item['answer_correctness']['confidence'])
            
            accuracy = correct_answers / total_questions if total_questions > 0 else 0
            retrieval_rate = successful_retrievals / total_questions if total_questions > 0 else 0
            avg_chunks = total_chunks / total_questions if total_questions > 0 else 0
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            return {
                'total_questions': total_questions,
                'accuracy': accuracy,
                'retrieval_success_rate': retrieval_rate,
                'avg_chunks_per_question': avg_chunks,
                'avg_confidence': avg_confidence,
                'confidence_distribution': confidence_scores,
                'status': 'success'
            }
            
        except Exception as e:
            return {"error": f"Error parsing detailed results: {str(e)}"}
    
    def load_experiment_data(self, experiment_name: str) -> Dict[str, Any]:
        """Load data for a specific experiment"""
        experiment_dir = os.path.join(self.base_dir, experiment_name)
        
        if not os.path.exists(experiment_dir):
            return {"error": f"Experiment directory not found: {experiment_dir}"}
        
        # Parse summary file
        summary_path = os.path.join(experiment_dir, 'evaluation_summary.txt')
        summary_data = self.parse_summary_file(summary_path)
        
        # Parse detailed results for each dataset
        detailed_data = {}
        for dataset in self.datasets:
            dataset_path = os.path.join(experiment_dir, dataset, 'advanced_sample_results.json')
            detailed_data[dataset] = self.parse_detailed_results(dataset_path)
        
        return {
            'experiment_name': experiment_name,
            'summary': summary_data,
            'detailed': detailed_data
        }
    
    def load_all_experiments(self) -> Dict[str, Any]:
        """Load data for all available experiments"""
        experiments = {}
        
        # Look for final_result_* directories
        experiment_dirs = []
        
        # Check for final_result_* directories
        for item in os.listdir(self.base_dir):
            if item.startswith('final_result_') and os.path.isdir(os.path.join(self.base_dir, item)):
                experiment_dirs.append(item)
        
        # Sort to ensure consistent ordering
        experiment_dirs.sort()
        
        # Load each experiment
        for exp_dir in experiment_dirs:
            exp_data = self.load_experiment_data(exp_dir)
            experiments[exp_dir] = exp_data
        
        return experiments
    
    def create_comparison_dataframe(self, experiments_data: Dict[str, Any]) -> pd.DataFrame:
        """Create a DataFrame for easy comparison of experiments"""
        rows = []
        
        for exp_name, exp_data in experiments_data.items():
            if 'error' in exp_data:
                continue
                
            summary = exp_data.get('summary', {})
            if 'error' in summary:
                continue
            
            # Overall metrics
            overall = summary.get('overall', {})
            row = {
                'experiment': exp_name,
                'total_questions': overall.get('total_questions', 0),
                'overall_accuracy': overall.get('overall_accuracy', 0),
                'overall_retrieval_success_rate': overall.get('overall_retrieval_success_rate', 0),
                'overall_f1_score': overall.get('overall_f1_score', 0),
                'avg_chunks_per_question': overall.get('avg_chunks_per_question', 0)
            }
            
            # Dataset-specific metrics
            datasets = summary.get('datasets', {})
            for dataset in self.datasets:
                if dataset in datasets:
                    row[f'{dataset}_accuracy'] = datasets[dataset].get('accuracy', 0)
                    row[f'{dataset}_retrieval_rate'] = datasets[dataset].get('retrieval_success_rate', 0)
                    row[f'{dataset}_f1_score'] = datasets[dataset].get('f1_score', 0)
                else:
                    row[f'{dataset}_accuracy'] = 0
                    row[f'{dataset}_retrieval_rate'] = 0
                    row[f'{dataset}_f1_score'] = 0
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_experiment_labels(self) -> Dict[str, str]:
        """Get human-readable labels for experiments"""
        return {
            'advanced_evaluation_results': 'Current System',
            'final_result_2_preprocess_think': 'Complete System (Original)',
            'final_result_8_preprocess_think_usefulness_v2': 'Complete System (Optimized)',
            'final_result_3_preprocess_think_no_rewriter': 'No Query Rewriter',
            'final_result_4_preprocess_think_no_usefulness': 'No Usefulness Judge',
            'final_result_5_preprocess_think_no_dense_chunks': 'No Dense Chunk Retrieval',
            'final_result_6_preprocess_think_no_dense_keywords': 'No Dense Keyword Retrieval',
            'final_result_7_preprocess_think_no_dense_questions': 'No Dense Question Retrieval'
        }
    
    def get_dataset_labels(self) -> Dict[str, str]:
        """Get human-readable labels for datasets"""
        return {
            'hotpotqa': 'HotpotQA',
            'ms_marco': 'MS MARCO',
            'natural_questions': 'Natural Questions',
            'triviaqa': 'TriviaQA'
        }


if __name__ == "__main__":
    # Test the parser
    parser = RAGDataParser()
    
    # Load current available data
    current_data = parser.load_experiment_data('advanced_evaluation_results')
    print("Current experiment data:")
    print(json.dumps(current_data, indent=2, ensure_ascii=False))
    
    # Load all experiments
    all_data = parser.load_all_experiments()
    print(f"\nFound {len(all_data)} experiments")
    
    # Create comparison DataFrame
    df = parser.create_comparison_dataframe(all_data)
    print("\nComparison DataFrame:")
    print(df.to_string())