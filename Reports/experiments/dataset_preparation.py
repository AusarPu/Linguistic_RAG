#!/usr/bin/env python3
"""
数据集准备脚本
下载并预处理QuAC和Natural Questions数据集用于RAG系统评测
"""

import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_datasets_cache'

import json
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Any
import logging
import time
from collections import deque
import multiprocessing as mp
from datasets import load_dataset
import pandas as pd

# 尝试设置huggingface_hub的配置
try:
    from huggingface_hub import configure_http_backend
    import requests
    configure_http_backend(backend_factory=lambda: requests.Session())
except ImportError:
    pass
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreparator:
    def __init__(self, data_dir: str = "/home/pushihao/RAG/Reports/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
    # 计算缓存目录的累计大小（用于测速）
    def _get_combined_cache_size(self) -> int:
        total = 0
        for env_key in ("HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE"):
            d = os.environ.get(env_key)
            if d and os.path.isdir(d):
                for root, _, files in os.walk(d):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        try:
                            total += os.path.getsize(fp)
                        except OSError:
                            pass
        return total

    # 看门狗运行器：监控下载速度，低于阈值持续一段时间则中断并重试
    def _watchdog_run(self, target, args=(), min_speed_bytes: int = 1_048_576, window_seconds: int = 60, check_interval: float = 1.0, max_retries: int = 3):
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            logging.info(f"[Watchdog] 启动下载子进程 (第{attempt}次尝试)...")
            p = mp.Process(target=target, args=args)
            p.start()

            prev_size = self._get_combined_cache_size()
            speeds = deque(maxlen=window_seconds)
            aborted = False

            while p.is_alive():
                time.sleep(check_interval)
                cur_size = self._get_combined_cache_size()
                delta = max(0, cur_size - prev_size)
                prev_size = cur_size
                speeds.append(delta)

                if len(speeds) == window_seconds:
                    avg_speed = sum(speeds) / window_seconds
                    logging.info(f"[Watchdog] 最近{window_seconds}s平均速度: {avg_speed/1_048_576:.2f} MiB/s")
                    if avg_speed < min_speed_bytes:
                        logging.warning(f"[Watchdog] 速度低于阈值 ({avg_speed/1_048_576:.2f} < 1.00 MiB/s)，中断并重试...")
                        try:
                            p.terminate()
                            p.join(timeout=10)
                            if p.is_alive():
                                p.kill()
                                p.join(timeout=5)
                        finally:
                            aborted = True
                        break

            # 如果是因为低速而中断，直接重试
            if aborted:
                continue

            # 子进程自然结束
            p.join()
            if p.exitcode == 0:
                logging.info("[Watchdog] 子进程成功完成")
                return True
            else:
                logging.error(f"[Watchdog] 子进程异常退出，exitcode={p.exitcode}，准备重试...")
                continue

        raise Exception("下载在多次重试后仍未成功")



    
    def _download_nq_once(self):
        """处理NQ数据集：先下载后处理"""
        # 第一阶段：纯下载（使用测速监控）
        logger.info("第一阶段：下载Natural Questions数据集...")
        self._watchdog_run(self._download_nq_raw)
        
        # 第二阶段：数据处理和格式转换（不使用测速监控）
        logger.info("第二阶段：处理和转换Natural Questions数据集...")
        self._process_nq_data()

    
    def _process_nq_data(self):
        """处理和转换Natural Questions数据集"""
        # 确保不在离线模式
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('HF_DATASETS_OFFLINE', None)
        # 强制使用镜像与缓存位置
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ.setdefault('HUGGINGFACE_HUB_CACHE', '/tmp/huggingface_cache')
        os.environ.setdefault('HF_DATASETS_CACHE', '/tmp/hf_datasets_cache')

        # 从缓存加载数据集
        dataset = load_dataset(
            "google-research-datasets/natural_questions",
            trust_remote_code=True,
            token=os.environ.get('HF_TOKEN'),
            keep_in_memory=False,
            download_mode="reuse_cache_if_exists",
        )
        if dataset is None:
            raise Exception("数据集加载失败: 返回为空")

        # 保存到本地
        nq_dir = self.data_dir / "natural_questions"
        nq_dir.mkdir(exist_ok=True)

        # 处理训练集和验证集（限制数量以减少处理时间）
        train_data: List[Dict] = []
        val_data: List[Dict] = []
        train_limit = 9999999999999999999999999999
        val_limit = 9999999999999999999999999

        # 选择可用的分割名称并打印信息
        available_splits = list(dataset.keys())
        logging.info(f"NQ 可用分割: {available_splits}")
        train_split = 'train' if 'train' in dataset else (available_splits[0] if available_splits else None)
        val_split = 'validation' if 'validation' in dataset else ('dev' if 'dev' in dataset else ('test' if 'test' in dataset else None))
        logging.info(f"使用分割: train='{train_split}', val='{val_split}'")
        if train_split:
            logging.info(f"{train_split} 样本数: {len(dataset[train_split])}")
        if val_split:
            logging.info(f"{val_split} 样本数: {len(dataset[val_split])}")

        # 处理数据
        if train_split:
            for example in dataset[train_split]:
                if len(train_data) >= train_limit:
                    break
                processed_example = self._process_nq_example(example)
                if processed_example:
                    train_data.append(processed_example)
            logging.info(f"已收集训练样本: {len(train_data)}/{train_limit}")

        if val_split:
            for example in dataset[val_split]:
                if len(val_data) >= val_limit:
                    break
                processed_example = self._process_nq_example(example)
                if processed_example:
                    val_data.append(processed_example)
            logging.info(f"已收集验证样本: {len(val_data)}/{val_limit}")

        # 原子写入，避免部分写入
        tmp_train = nq_dir / "train.json.tmp"
        tmp_val = nq_dir / "validation.json.tmp"
        final_train = nq_dir / "train.json"
        final_val = nq_dir / "validation.json"

        with open(tmp_train, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        with open(tmp_val, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        os.replace(tmp_train, final_train)
        os.replace(tmp_val, final_val)

    def _clean_html_content(self, html_content: str) -> str:
        """
        深度清理HTML内容，移除标签、CSS样式等
        """
        import re
        
        if not html_content:
            return ""
        
        # 移除base64编码内容
        html_content = re.sub(r'data:[^;]+;base64,[A-Za-z0-9+/=]+', '', html_content)
        
        # 移除内联样式
        html_content = re.sub(r'style="[^"]*"', '', html_content)
        html_content = re.sub(r"style='[^']*'", '', html_content)
        
        try:
            from bs4 import BeautifulSoup
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 移除不需要的标签
            for tag in soup(['script', 'style', 'meta', 'link', 'noscript', 'head']):
                tag.decompose()
            
            # 获取纯文本
            text = soup.get_text(separator=' ', strip=True)
            
        except ImportError:
            # 如果没有BeautifulSoup，使用正则表达式
            text = re.sub(r'<[^>]+>', ' ', html_content)
        
        # 移除CSS相关内容
        text = re.sub(r'\.[a-zA-Z][a-zA-Z0-9_-]*\s*\{[^}]*\}', '', text)  # CSS规则
        text = re.sub(r'\.[a-zA-Z][a-zA-Z0-9_-]*', '', text)  # CSS类名
        text = re.sub(r'#[a-zA-Z][a-zA-Z0-9_-]*', '', text)  # CSS ID
        text = re.sub(r'@[a-zA-Z-]+[^;]*;', '', text)  # CSS @规则
        text = re.sub(r'url\([^)]*\)', '', text)  # URL引用
        text = re.sub(r'rgb\([^)]*\)', '', text)  # RGB颜色
        text = re.sub(r'rgba\([^)]*\)', '', text)  # RGBA颜色
        text = re.sub(r'#[0-9a-fA-F]{3,6}', '', text)  # 十六进制颜色
        text = re.sub(r'\b\d+px\b', '', text)  # 像素值
        text = re.sub(r'\b\d+%\b', '', text)  # 百分比值
        
        # 清理空白字符
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 如果文本包含大量CSS残留，尝试提取有意义的部分
        if len(text) > 1000 and ('mw-' in text or 'css' in text.lower()):
            # 尝试从Wikipedia标题或段落中提取内容
            sentences = text.split('.')
            meaningful_sentences = []
            for sentence in sentences[:10]:  # 只检查前10句
                sentence = sentence.strip()
                if (len(sentence) > 20 and 
                    not any(css_indicator in sentence.lower() for css_indicator in 
                           ['mw-', 'css', 'style', 'class', 'div', 'span'])):
                    meaningful_sentences.append(sentence)
            
            if meaningful_sentences:
                text = '. '.join(meaningful_sentences[:3]) + '.'
        
        # 限制长度
        if len(text) > 2000:
            text = text[:2000]
        
        return text
    
    def _process_nq_example(self, example: Dict) -> Dict:
        """
        处理Natural Questions数据集中的单个样本（兼容多种Schema）
        """
        try:
            # 1) 问题字段：可能是 'question' (str) 或 {'text': str} 或 'question_text'
            question = None
            q = example.get("question")
            if isinstance(q, str):
                question = q
            elif isinstance(q, dict):
                question = q.get("text")
            if not question:
                question = example.get("question_text") or ""
            if not question:
                return None

            # 2) 文档token与标题
            document = example.get("document", {}) if isinstance(example.get("document"), dict) else {}
            tokens = document.get("tokens") or example.get("document_tokens") or []
            title = example.get("document_title") or document.get("title", "")

            # 3) 注释与答案 - 修复annotations字段处理
            annotations = example.get("annotations")
            if not annotations:
                return None
            
            # 处理不同的annotations结构
            if isinstance(annotations, dict):
                # 如果annotations是字典，检查是否有train字段
                if "train" in annotations:
                    ann_list = annotations.get("train", [])
                    if isinstance(ann_list, list) and ann_list:
                        ann = ann_list[0]
                    else:
                        return None
                else:
                    ann = annotations
            elif isinstance(annotations, list) and annotations:
                ann = annotations[0]
            else:
                return None

            # 优先短答案
            answer_text = ""
            short_answers = ann.get("short_answers", []) or []
            if isinstance(short_answers, dict):
                short_answers = [short_answers]

            def _slice_tokens(ts, st, ed):
                if not isinstance(ts, list):
                    return ""
                try:
                    st = max(0, int(st))
                    ed = max(st, int(ed))
                    if st >= len(ts) or ed > len(ts):
                        return ""
                    token_slice = ts[st:ed]
                    result_tokens = []
                    for t in token_slice:
                        if isinstance(t, dict):
                            token_text = t.get("token", "")
                        else:
                            token_text = str(t) if t is not None else ""
                        if token_text:
                            result_tokens.append(token_text)
                    return " ".join(result_tokens)
                except Exception as e:
                    logger.warning(f"_slice_tokens错误: {e}")
                    return ""

            # 遍历所有短答案，找到有效的
            for sa in short_answers:
                if isinstance(sa, dict):
                    text_list = sa.get("text", [])
                    if isinstance(text_list, list) and text_list and text_list[0]:
                        answer_text = text_list[0]
                        break
                    elif isinstance(text_list, str) and text_list:
                        answer_text = text_list
                        break
                    else:
                        # 尝试从token位置提取
                        start_tokens = sa.get("start_token", [])
                        end_tokens = sa.get("end_token", [])
                        if (isinstance(start_tokens, list) and start_tokens and 
                            isinstance(end_tokens, list) and end_tokens and tokens):
                            st = start_tokens[0]
                            ed = end_tokens[0]
                            if st is not None and ed is not None:
                                answer_text = _slice_tokens(tokens, st, ed)
                                if answer_text:
                                    break
            
            # 其次长答案
            if not answer_text:
                long_answers = ann.get("long_answer", [])
                if isinstance(long_answers, dict):
                    long_answers = [long_answers]
                elif isinstance(long_answers, list):
                    for la in long_answers:
                        if isinstance(la, dict):
                            st = la.get("start_token")
                            ed = la.get("end_token")
                            if st is not None and ed is not None and st != -1 and ed != -1 and tokens:
                                answer_text = _slice_tokens(tokens, st, ed)
                                if answer_text:
                                    break

            if not answer_text:
                return None

            # 4) 构造上下文：直接使用原始内容
            context = ""
            if tokens and isinstance(tokens, list):
                try:
                    # 直接拼接所有token
                    result_tokens = []
                    for t in tokens:
                        if isinstance(t, dict):
                            token_text = t.get("token", "")
                        else:
                            token_text = str(t) if t is not None else ""
                        if token_text:
                            result_tokens.append(token_text)
                    context = " ".join(result_tokens)
                except Exception as e:
                    logger.warning(f"构造上下文失败: {e}")
                    context = ""
            
            if not context:
                # 使用原始HTML内容
                if isinstance(document, dict):
                    html_content = document.get("html", "")
                    if html_content:
                        context = html_content
                    elif title:  # 如果没有HTML内容，使用标题
                        context = title

            return {
                "id": example.get("example_id") or example.get("id", ""),
                "question": question,
                "answer": answer_text,
                "context": context,
                "document_title": title
            }
        except Exception as e:
            logger.warning(f"处理NQ样本失败: {e}")
            return None
    
    def create_evaluation_splits(self):
        """
        为实验创建评测数据分割
        """
        logger.info("创建评测数据分割...")
        
        eval_dir = self.data_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        # 为Natural Questions创建小规模测试集
        nq_val_file = self.data_dir / "natural_questions" / "validation.json"
        if nq_val_file.exists():
            with open(nq_val_file, "r", encoding="utf-8") as f:
                nq_data = json.load(f)
            
            # 确保数据是列表格式
            if isinstance(nq_data, list):
                # 取前100个问题用于快速测试
                nq_test = nq_data[:100] if len(nq_data) > 100 else nq_data
            else:
                # 如果不是列表，转换为列表
                nq_test = [nq_data] if nq_data else []
            
            with open(eval_dir / "nq_test.json", "w", encoding="utf-8") as f:
                json.dump(nq_test, f, ensure_ascii=False, indent=2)
        
        logger.info("评测数据分割创建完成")
    
    def _download_generic_once(self, dataset_id: str, out_subdir: str, process_fn_name: str, train_limit: int = 100, val_limit: int = 50, config_name: str = None):
        """通用子进程：下载并处理任意HF数据集（通过函数名定位处理器）"""
        # 确保不在离线模式
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('HF_DATASETS_OFFLINE', None)
        # 强制使用镜像与缓存位置
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ.setdefault('HUGGINGFACE_HUB_CACHE', '/tmp/huggingface_cache')
        os.environ.setdefault('HF_DATASETS_CACHE', '/tmp/hf_datasets_cache')

        # 构建load_dataset参数
        load_kwargs = {
            'trust_remote_code': True,
            'token': os.environ.get('HF_TOKEN'),
            'keep_in_memory': False,
            'download_mode': "reuse_cache_if_exists",
        }
        
        # 只有在config_name不为None时才添加该参数
        if config_name is not None:
            load_kwargs['name'] = config_name
            
        dataset = load_dataset(dataset_id, **load_kwargs)
        if dataset is None:
            raise Exception(f"数据集下载失败: {dataset_id} 返回为空")

        out_dir = self.data_dir / out_subdir
        out_dir.mkdir(exist_ok=True)

        # 选择可用的分割名称并打印信息
        available_splits = list(dataset.keys())
        logging.info(f"{dataset_id} 可用分割: {available_splits}")
        train_split = 'train' if 'train' in dataset else (available_splits[0] if available_splits else None)
        val_split = 'validation' if 'validation' in dataset else ('dev' if 'dev' in dataset else ('test' if 'test' in dataset else None))
        logging.info(f"使用分割: train='{train_split}', val='{val_split}'")
        if train_split:
            logging.info(f"{train_split} 样本数: {len(dataset[train_split])}")
        if val_split:
            logging.info(f"{val_split} 样本数: {len(dataset[val_split])}")

        process_fn = getattr(self, process_fn_name)

        train_data: List[Dict] = []
        val_data: List[Dict] = []

        if train_split:
            for example in dataset[train_split]:
                if len(train_data) >= train_limit:
                    break
                processed = process_fn(example)
                if processed:
                    train_data.append(processed)
            logging.info(f"[{dataset_id}] 已收集训练样本: {len(train_data)}/{train_limit}")

        if val_split:
            for example in dataset[val_split]:
                if len(val_data) >= val_limit:
                    break
                processed = process_fn(example)
                if processed:
                    val_data.append(processed)
            logging.info(f"[{dataset_id}] 已收集验证样本: {len(val_data)}/{val_limit}")

        # 原子写入
        tmp_train = out_dir / "train.json.tmp"
        tmp_val = out_dir / "validation.json.tmp"
        final_train = out_dir / "train.json"
        final_val = out_dir / "validation.json"
        with open(tmp_train, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        with open(tmp_val, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_train, final_train)
        os.replace(tmp_val, final_val)

    # ---- HotpotQA ----
    def _process_hotpotqa_example(self, ex: Dict) -> Dict:
        try:
            question = ex.get('question') or ex.get('query') or ''
            answer = ex.get('answer') or ''
            if not (question and isinstance(answer, str) and answer.strip()):
                return None
            context_text = ''
            ctx = ex.get('context')
            
            # HotpotQA context可能是字典或列表格式
            if isinstance(ctx, dict):
                # 字典格式: {'title': [...], 'sentences': [...]}
                titles = ctx.get('title', [])
                sentences = ctx.get('sentences', [])
                
                if isinstance(titles, list) and isinstance(sentences, list):
                    # 组合标题和句子
                    combined_text = []
                    for i, title in enumerate(titles[:10]):  # 限制标题数量
                        combined_text.append(f"Title: {title}")
                        if i < len(sentences) and isinstance(sentences[i], list):
                            # 每个标题对应的句子列表
                            combined_text.extend(sentences[i][:5])  # 每个标题最多5句
                    context_text = ' '.join(combined_text)
                    
            elif isinstance(ctx, list) and ctx:
                first = ctx[0]
                # 原始HotpotQA: [title, [sentences...]]
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    sents = first[1]
                    if isinstance(sents, list):
                        context_text = ' '.join(sents[:20])
                elif isinstance(first, dict):
                    sents = first.get('sentences') or first.get('text')
                    if isinstance(sents, list):
                        context_text = ' '.join(sents[:20])
                    elif isinstance(sents, str):
                        context_text = sents[:2000]
            
            # 限制最终上下文长度
            if len(context_text) > 4000:
                context_text = context_text[:4000]
                
            return {
                'id': ex.get('_id') or ex.get('id') or '',
                'question': question,
                'answer': answer,
                'context': context_text,
                'document_title': ''
            }
        except Exception as e:
            logger.warning(f"处理HotpotQA样本失败: {e}")
            return None

    def _download_hotpotqa_once(self):
        # 使用 fullwiki 配置
        return self._download_generic_once(
            dataset_id="hotpot_qa",
            out_subdir="hotpot_qa",
            process_fn_name="_process_hotpotqa_example",
            train_limit=100,
            val_limit=50,
            config_name="fullwiki"
        )

    # ---- TriviaQA ----
    def _process_triviaqa_example(self, ex: Dict) -> Dict:
        try:
            question = ex.get('question') or ''
            # 答案字段多样
            answer = ''
            ans = ex.get('answer')
            if isinstance(ans, dict):
                answer = ans.get('value') or (ans.get('normalized_value') if isinstance(ans.get('normalized_value'), str) else '')
                if not answer:
                    aliases = ans.get('aliases')
                    if isinstance(aliases, list) and aliases:
                        answer = aliases[0]
            elif isinstance(ans, str):
                answer = ans
            if not answer:
                answer = ex.get('answer_text') or ''
            if not (question and answer):
                return None
            context = ex.get('context') or ''
            return {
                'id': ex.get('id') or ex.get('question_id') or '',
                'question': question,
                'answer': answer,
                'context': context if isinstance(context, str) else str(context)[:2000],
                'document_title': ''
            }
        except Exception as e:
            logger.warning(f"处理TriviaQA样本失败: {e}")
            return None

    def _download_triviaqa_once(self):
        # TriviaQA 使用正确的数据集名称
        try:
            return self._download_generic_once(
                dataset_id="mandarjoshi/trivia_qa/rc",
                out_subdir="trivia_qa",
                process_fn_name="_process_triviaqa_example",
                train_limit=100,
                val_limit=50,
            )
        except Exception:
            try:
                return self._download_generic_once(
                    dataset_id="mandarjoshi/trivia_qa/unfiltered",
                    out_subdir="trivia_qa",
                    process_fn_name="_process_triviaqa_example",
                    train_limit=100,
                    val_limit=50,
                )
            except Exception:
                # 如果都失败，使用原始数据下载方法
                self._download_triviaqa_raw()
                return self._process_dataset("trivia_qa", self._process_triviaqa_example, 100, 50)



    # ---- MS MARCO ----
    def _process_msmarco_example(self, ex: Dict) -> Dict:
        try:
            question = ex.get('query') or ex.get('question') or ''
            # answers可以是list
            answer = ''
            answers = ex.get('answers')
            if isinstance(answers, list) and answers:
                # 选择第一个非空
                for a in answers:
                    if isinstance(a, str) and a.strip():
                        answer = a
                        break
            if not answer:
                answer = ex.get('answer') or ''
            if not (question and answer):
                return None
            context = ''
            passages = ex.get('passages')
            
            # 处理不同的passages结构
            if isinstance(passages, list) and passages:
                # 旧版本格式: list of dicts
                texts = []
                for p in passages[:5]:
                    if isinstance(p, dict) and p.get('passage_text'):
                        texts.append(p['passage_text'])
                context = ' '.join(texts)[:4000]
            elif isinstance(passages, dict):
                # 新版本格式: dict with passage_text list
                passage_texts = passages.get('passage_text', [])
                is_selected = passages.get('is_selected', [])
                
                if isinstance(passage_texts, list):
                    # 优先选择被标记为选中的段落
                    selected_texts = []
                    other_texts = []
                    
                    for i, text in enumerate(passage_texts[:10]):  # 限制处理数量
                        if isinstance(text, str) and text.strip():
                            if i < len(is_selected) and is_selected[i] == 1:
                                selected_texts.append(text)
                            else:
                                other_texts.append(text)
                    
                    # 先用选中的段落，不够再补充其他段落
                    all_texts = selected_texts + other_texts[:5-len(selected_texts)]
                    context = ' '.join(all_texts)[:4000]
            
            return {
                'id': ex.get('query_id') or ex.get('id') or '',
                'question': question,
                'answer': answer,
                'context': context,
                'document_title': ''
            }
        except Exception as e:
            logger.warning(f"处理MS MARCO样本失败: {e}")
            return None

    def _download_msmarco_once(self):
        # MS MARCO 使用 microsoft/ms_marco 配置 v1.1
        try:
            return self._download_generic_once(
                dataset_id="microsoft/ms_marco",
                out_subdir="ms_marco",
                process_fn_name="_process_msmarco_example",
                train_limit=100,
                val_limit=50,
                config_name="v1.1"
            )
        except Exception as e:
            logger.error(f"加载 microsoft/ms_marco v1.1 失败: {e}")
            raise RuntimeError(f"MS MARCO 数据集加载失败: {e}")
    
    # 原始数据下载方法（不进行处理和切分）
    def _download_nq_raw(self):
        """下载 Natural Questions 原始数据（仅下载，不转换格式）"""
        logger.info("下载 Natural Questions 原始数据...")
        
        try:
            # 仅下载数据集到缓存，不进行格式转换和保存
            dataset = load_dataset("google-research-datasets/natural_questions", trust_remote_code=True)
            logger.info(f"Natural Questions 数据集下载完成，缓存到系统")
        except Exception as e:
            logger.error(f"下载 Natural Questions 数据集失败: {e}")
            raise Exception(f"无法下载 Natural Questions 数据集: {e}")
    

    def _download_hotpotqa_raw(self):
        """下载 HotpotQA 原始数据（仅下载，不转换格式）"""
        logger.info("下载 HotpotQA 原始数据...")
        
        # 仅下载数据集到缓存，不进行格式转换和保存
        dataset = load_dataset("hotpot_qa", "fullwiki", trust_remote_code=True)
        logger.info(f"HotpotQA 数据集下载完成，缓存到系统")
    
    def _download_triviaqa_raw(self):
        """下载 TriviaQA 原始数据（仅下载，不转换格式）"""
        logger.info("下载 TriviaQA 原始数据...")
        
        try:
            # 仅下载数据集到缓存，不进行格式转换和保存
            dataset = load_dataset("mandarjoshi/trivia_qa", "rc", trust_remote_code=True)
            logger.info(f"TriviaQA 数据集下载完成，缓存到系统")
        except Exception as e:
            logger.warning(f"使用 mandarjoshi/trivia_qa 失败: {e}，尝试其他方式...")
            try:
                dataset = load_dataset("mandarjoshi/trivia_qa", "unfiltered", trust_remote_code=True)
                logger.info(f"TriviaQA 数据集下载完成，缓存到系统")
            except Exception as e2:
                # 所有下载尝试都失败时直接抛出异常
                raise Exception(f"TriviaQA数据集下载失败: {e2}")
    
    
    def _download_msmarco_raw(self):
        """下载 MS MARCO 原始数据（仅下载，不转换格式）"""
        logger.info("下载 MS MARCO 原始数据...")
        
        try:
            # 仅下载数据集到缓存，不进行格式转换和保存
            dataset = load_dataset("microsoft/ms_marco", "v2.1", trust_remote_code=True)
            logger.info(f"MS MARCO 数据集下载完成，缓存到系统")
        except:
            try:
                logger.warning("MS MARCO 'v2.1' 配置失败，尝试 'v1.1'...")
                dataset = load_dataset("microsoft/ms_marco", "v1.1", trust_remote_code=True)
                logger.info(f"MS MARCO 数据集下载完成，缓存到系统")
            except:
                logger.warning("MS MARCO 'v1.1' 配置失败，尝试默认配置...")
                dataset = load_dataset("microsoft/ms_marco", trust_remote_code=True)
                logger.info(f"MS MARCO 数据集下载完成，缓存到系统")
    
    def _process_dataset(self, dataset_name: str, process_fn, train_limit: int = 100, val_limit: int = 50):
        """通用数据集处理方法"""
        raw_dir = self.data_dir / "raw" / dataset_name
        out_dir = self.data_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 读取原始数据
            import pandas as pd
            train_df = pd.read_json(raw_dir / "train_raw.json", lines=True)
            val_df = pd.read_json(raw_dir / "validation_raw.json", lines=True)
            
            # 采样
            if len(train_df) > train_limit:
                train_df = train_df.sample(n=train_limit, random_state=42)
            if len(val_df) > val_limit:
                val_df = val_df.sample(n=val_limit, random_state=42)
            
            # 处理数据
            train_processed = []
            val_processed = []
            
            for _, row in train_df.iterrows():
                try:
                    processed = process_fn(row.to_dict())
                    if processed:
                        train_processed.append(processed)
                except Exception as e:
                    logger.warning(f"处理训练样本失败: {e}")
            
            for _, row in val_df.iterrows():
                try:
                    processed = process_fn(row.to_dict())
                    if processed:
                        val_processed.append(processed)
                except Exception as e:
                    logger.warning(f"处理验证样本失败: {e}")
            
            # 保存处理后的数据
            with open(out_dir / "train.json", "w", encoding="utf-8") as f:
                json.dump(train_processed, f, ensure_ascii=False, indent=2)
            
            with open(out_dir / "validation.json", "w", encoding="utf-8") as f:
                json.dump(val_processed, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ {dataset_name} 处理完成: 训练集 {len(train_processed)} 条, 验证集 {len(val_processed)} 条")
            return {"train_size": len(train_processed), "val_size": len(val_processed), "status": "success", "path": str(out_dir)}
            
        except Exception as e:
            logger.error(f"处理 {dataset_name} 失败: {e}")
            return {"train_size": 0, "val_size": 0, "status": "failed", "path": str(out_dir)}

    # 更新下载入口，改用专用 once 函数（以便传递配置）
    def download_hotpotqa(self) -> Dict[str, Any]:
        logger.info("开始处理HotpotQA数据集...")
        self._download_hotpotqa_once()
        out_dir = self.data_dir / "hotpot_qa"
        with open(out_dir / "train.json", "r", encoding="utf-8") as f:
            train = json.load(f)
        with open(out_dir / "validation.json", "r", encoding="utf-8") as f:
            val = json.load(f)
        logger.info(f"✅ HotpotQA完成: 训练集 {len(train)} 条, 验证集 {len(val)} 条")
        return {"train_size": len(train), "val_size": len(val), "status": "success", "path": str(out_dir)}

    def download_triviaqa(self) -> Dict[str, Any]:
        logger.info("开始处理TriviaQA数据集...")
        self._download_triviaqa_once()
        out_dir = self.data_dir / "trivia_qa"
        with open(out_dir / "train.json", "r", encoding="utf-8") as f:
            train = json.load(f)
        with open(out_dir / "validation.json", "r", encoding="utf-8") as f:
            val = json.load(f)
        logger.info(f"✅ TriviaQA完成: 训练集 {len(train)} 条, 验证集 {len(val)} 条")
        return {"train_size": len(train), "val_size": len(val), "status": "success", "path": str(out_dir)}



    def download_msmarco(self) -> Dict[str, Any]:
        logger.info("开始处理MS MARCO数据集...")
        self._download_msmarco_once()
        out_dir = self.data_dir / "ms_marco"
        with open(out_dir / "train.json", "r", encoding="utf-8") as f:
            train = json.load(f)
        with open(out_dir / "validation.json", "r", encoding="utf-8") as f:
            val = json.load(f)
        logger.info(f"✅ MS MARCO完成: 训练集 {len(train)} 条, 验证集 {len(val)} 条")
        return {"train_size": len(train), "val_size": len(val), "status": "success", "path": str(out_dir)}

    def download_all_datasets(self):
        """
        第一阶段：下载所有数据集原始数据（不进行切分处理）
        """
        logger.info("=== 第一阶段：开始下载所有数据集原始数据 ===")
        
        download_results = {}
        
        # 下载 Natural Questions
        logger.info("下载 Natural Questions...")
        try:
            self._watchdog_run(
                target=self._download_nq_raw,
                args=(),
                min_speed_bytes=1_048_576,
                window_seconds=60,
                check_interval=1.0,
                max_retries=3,
            )
            download_results["natural_questions"] = "success"
        except Exception as e:
            logger.error(f"Natural Questions 下载失败: {e}")
            download_results["natural_questions"] = "failed"
        
        # 下载 HotpotQA
        logger.info("下载 HotpotQA...")
        try:
            self._watchdog_run(
                target=self._download_hotpotqa_raw,
                args=(),
                min_speed_bytes=1_048_576,
                window_seconds=60,
                check_interval=1.0,
                max_retries=3,
            )
            download_results["hotpot_qa"] = "success"
        except Exception as e:
            logger.error(f"HotpotQA 下载失败: {e}")
            download_results["hotpot_qa"] = "failed"
        
        # 下载 TriviaQA
        logger.info("下载 TriviaQA...")
        try:
            self._watchdog_run(
                target=self._download_triviaqa_raw,
                args=(),
                min_speed_bytes=1_048_576,
                window_seconds=60,
                check_interval=1.0,
                max_retries=3,
            )
            download_results["trivia_qa"] = "success"
        except Exception as e:
            logger.error(f"TriviaQA 下载失败: {e}")
            download_results["trivia_qa"] = "failed"
        

        
        # 下载 MS MARCO
        logger.info("下载 MS MARCO...")
        try:
            self._watchdog_run(
                target=self._download_msmarco_raw,
                args=(),
                min_speed_bytes=1_048_576,
                window_seconds=60,
                check_interval=1.0,
                max_retries=3,
            )
            download_results["ms_marco"] = "success"
        except Exception as e:
            logger.error(f"MS MARCO 下载失败: {e}")
            download_results["ms_marco"] = "failed"
        
        logger.info("=== 第一阶段：所有数据集下载完成 ===")
        return download_results
    
    def process_all_datasets(self):
        """
        第二阶段：处理和切分所有已下载的数据集
        """
        logger.info("=== 第二阶段：开始处理和切分数据集 ===")
        
        results = {}
        
        # 处理 Natural Questions - 使用专门的处理方法
        logger.info("处理 Natural Questions...")
        try:
            self._process_nq_data()
            # 读取处理结果
            nq_dir = self.data_dir / "natural_questions"
            train_fp = nq_dir / "train.json"
            val_fp = nq_dir / "validation.json"
            if train_fp.exists() and val_fp.exists():
                with open(train_fp, "r", encoding="utf-8") as f:
                    train_data = json.load(f)
                with open(val_fp, "r", encoding="utf-8") as f:
                    val_data = json.load(f)
                results["natural_questions"] = {
                    "train_size": len(train_data),
                    "val_size": len(val_data),
                    "status": "success",
                    "path": str(nq_dir)
                }
            else:
                results["natural_questions"] = {"train_size": 0, "val_size": 0, "status": "failed"}
        except Exception as e:
            logger.error(f"处理 Natural Questions 失败: {e}")
            results["natural_questions"] = {"train_size": 0, "val_size": 0, "status": "failed"}
        
        # 处理 HotpotQA
        if (self.data_dir / "raw" / "hotpot_qa").exists():
            logger.info("处理 HotpotQA...")
            results["hotpot_qa"] = self._process_dataset("hotpot_qa", self._process_hotpotqa_example)
        
        # 处理 TriviaQA
        if (self.data_dir / "raw" / "trivia_qa").exists():
            logger.info("处理 TriviaQA...")
            results["trivia_qa"] = self._process_dataset("trivia_qa", self._process_triviaqa_example)
        

        
        # 处理 MS MARCO
        if (self.data_dir / "raw" / "ms_marco").exists():
            logger.info("处理 MS MARCO...")
            results["ms_marco"] = self._process_dataset("ms_marco", self._process_msmarco_example)
        
        # 创建评测分割
        self.create_evaluation_splits()
        
        # 保存处理结果
        with open(self.data_dir / "preparation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info("=== 第二阶段：所有数据集处理完成 ===")
        return results
    
    def prepare_all_datasets(self):
        """
        完整的数据集准备流程：先下载，再处理切分
        """
        logger.info("开始完整的数据集准备流程...")
        
        # 第一阶段：下载所有数据集
        download_results = self.download_all_datasets()
        
        # 第二阶段：处理和切分数据集
        process_results = self.process_all_datasets()
        
        logger.info("完整的数据集准备流程完成")
        return process_results

if __name__ == "__main__":
    # 创建数据集准备器
    preparator = DatasetPreparator()
    
    # 准备所有数据集
    results = preparator.prepare_all_datasets()
    
    print("\n=== 数据集准备完成 ===")
    for dataset_name, result in results.items():
        if result:
            print(f"{dataset_name}: 训练集 {result.get('train_size', 0)} 条, 验证集 {result.get('val_size', 0)} 条")
        else:
            print(f"{dataset_name}: 准备失败")