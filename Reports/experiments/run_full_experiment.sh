#!/bin/bash

# 一键运行完整实验流程脚本
# 流程：converter -> chunk -> enhance -> index -> evaluation -> advanced_evaluation

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "一键运行完整实验流程脚本"
    echo ""
    echo "用法:"
    echo "  $0 [选项] <样本量>"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -s, --skip-stage    跳过指定阶段 (converter|chunk|enhance|index|evaluation|advanced)"
    echo "  --test              启用测试模式（用于index阶段）"
    echo "  --no-filter         不过滤没有答案的数据（用于converter阶段）"
    echo "  --enhance-mode      增强模式 (optimize|metadata|pipeline，默认：pipeline)"
    echo "  --batch-size        评估最大并发数（默认：10）"
    echo "  --verbose           启用详细日志输出"
    echo ""
    echo "消融实验选项（用于evaluation阶段）:"
    echo "  --no-query-rewriter     禁用查询重写模块"
    echo "  --no-dense-chunks       禁用密集块检索路径"
    echo "  --no-dense-keywords     禁用密集关键字检索路径"
    echo "  --no-dense-questions    禁用密集问题检索路径"
    echo "  --no-usefulness-judger  禁用有用性判断模块"
    echo ""
    echo "参数:"
    echo "  样本量              每个数据集的最大样本数量（必需）"
    echo ""
    echo "实验流程:"
    echo "  1. converter  - 转换数据集格式"
    echo "  2. chunk      - 对数据进行分块处理"
    echo "  3. enhance    - 使用LLM增强数据"
    echo "  4. index      - 构建搜索索引"
    echo "  5. evaluation - 运行RAG评估"
    echo "  6. advanced   - 运行高级评估分析"
    echo ""
    echo "示例:"
    echo "  $0 1000                           # 使用1000个样本运行完整流程"
    echo "  $0 --skip-stage converter 1000    # 跳过converter阶段"
    echo "  $0 --test --verbose 100           # 测试模式，详细日志，100个样本"
    echo "  $0 --enhance-mode optimize 500    # 仅优化模式，500个样本"
    echo ""
    echo "消融实验示例:"
    echo "  $0 --no-query-rewriter 100        # 禁用查询重写，测试其影响"
    echo "  $0 --no-usefulness-judger 100     # 禁用有用性判断，测试其影响"
    echo "  $0 --no-dense-chunks --no-dense-keywords 100  # 仅使用问题检索路径"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装或不在PATH中"
        return 1
    fi
    
    # 检查各阶段脚本是否存在
    local scripts=(
        "dataset_converters/convert_all.py"
        "dataset_chunk/chunk_datasets.py"
        "dataset_enhance/enhance_datasets.py"
        "dataset_index/build_dataset_indexes.py"
        "evaluation/evaluate_datasets.py"
        "run_advanced_evaluation.sh"
    )
    
    for script in "${scripts[@]}"; do
        if [ ! -f "$SCRIPT_DIR/$script" ]; then
            print_error "脚本不存在: $SCRIPT_DIR/$script"
            return 1
        fi
    done
    
    print_success "依赖检查通过"
    return 0
}

# 运行converter阶段
run_converter() {
    local max_samples=$1
    local no_filter=$2
    
    print_info "========== 阶段1: 数据集转换 =========="
    
    local cmd="python3 $SCRIPT_DIR/dataset_converters/convert_all.py --max-samples $max_samples"
    if [ "$no_filter" = true ]; then
        cmd="$cmd --no-filter"
    fi
    
    print_info "执行命令: $cmd"
    if eval $cmd; then
        print_success "数据集转换完成"
        return 0
    else
        print_error "数据集转换失败"
        return 1
    fi
}

# 运行chunk阶段
run_chunk() {
    print_info "========== 阶段2: 数据分块 =========="
    
    local cmd="python3 $SCRIPT_DIR/dataset_chunk/chunk_datasets.py"
    
    print_info "执行命令: $cmd"
    if eval $cmd; then
        print_success "数据分块完成"
        return 0
    else
        print_error "数据分块失败"
        return 1
    fi
}

# 运行enhance阶段
run_enhance() {
    local enhance_mode=$1
    local max_samples=$2
    local verbose=$3
    
    print_info "========== 阶段3: 数据增强 =========="
    
    local cmd="python3 $SCRIPT_DIR/dataset_enhance/enhance_datasets.py $enhance_mode"

    if [ "$verbose" = true ]; then
        cmd="$cmd --verbose"
    fi
    
    print_info "执行命令: $cmd"
    if eval $cmd; then
        print_success "数据增强完成"
        return 0
    else
        print_error "数据增强失败"
        return 1
    fi
}

# 运行index阶段
run_index() {
    local test_mode=$1
    local max_samples=$2
    
    print_info "========== 阶段4: 索引构建 =========="
    
    local cmd="python3 $SCRIPT_DIR/dataset_index/build_dataset_indexes.py"
    if [ "$test_mode" = true ]; then
        cmd="$cmd --test --test-limit $max_samples"
    fi
    
    print_info "执行命令: $cmd"
    if eval $cmd; then
        print_success "索引构建完成"
        return 0
    else
        print_error "索引构建失败"
        return 1
    fi
}

# 运行evaluation阶段
run_evaluation() {
    local max_samples=$1
    local batch_size=$2
    local ablation_args=$3
    
    print_info "========== 阶段5: RAG评估 =========="
    
    # 切换到项目根目录执行评估脚本，确保路径和环境一致
    local cmd="cd $PROJECT_ROOT && python3 $SCRIPT_DIR/evaluation/evaluate_datasets.py --dataset all --max-questions $max_samples --batch-size $batch_size $ablation_args"
    
    print_info "执行命令: $cmd"
    if eval $cmd; then
        print_success "RAG评估完成"
        return 0
    else
        print_error "RAG评估失败"
        return 1
    fi
}

# 运行advanced evaluation阶段
run_advanced_evaluation() {
    print_info "========== 阶段6: 高级评估分析 =========="
    
    # 修正路径：evaluation结果应该在rag_evaluation_results目录中
    local results_dir="/home/pushihao/RAG/Reports/experiments/datasets/rag_evaluation_results"
    
    # 检查评估结果是否存在
    if [ ! -d "$results_dir" ]; then
        print_error "评估结果目录不存在: $results_dir"
        return 1
    fi
    
    # 运行高级评估脚本，评估所有数据集
    local cmd="bash $SCRIPT_DIR/run_advanced_evaluation.sh -a"
    
    print_info "执行命令: $cmd"
    if eval $cmd; then
        print_success "高级评估分析完成"
        return 0
    else
        print_error "高级评估分析失败"
        return 1
    fi
}

# 主函数
main() {
    local max_samples=""
    local skip_stages=()
    local test_mode=false
    local no_filter=false
    local enhance_mode="pipeline"
    local batch_size=10
    local verbose=false
    local ablation_args=""
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -s|--skip-stage)
                skip_stages+=("$2")
                shift 2
                ;;
            --test)
                test_mode=true
                shift
                ;;
            --no-filter)
                no_filter=true
                shift
                ;;
            --enhance-mode)
                enhance_mode="$2"
                if [[ ! "$enhance_mode" =~ ^(optimize|metadata|pipeline)$ ]]; then
                    print_error "无效的增强模式: $enhance_mode"
                    exit 1
                fi
                shift 2
                ;;
            --batch-size)
                batch_size="$2"
                if ! [[ "$batch_size" =~ ^[0-9]+$ ]]; then
                    print_error "批处理大小必须是正整数: $batch_size"
                    exit 1
                fi
                shift 2
                ;;
            --verbose)
                verbose=true
                shift
                ;;
            --no-query-rewriter)
                ablation_args="$ablation_args --no-query-rewriter"
                shift
                ;;
            --no-dense-chunks)
                ablation_args="$ablation_args --no-dense-chunks"
                shift
                ;;
            --no-dense-keywords)
                ablation_args="$ablation_args --no-dense-keywords"
                shift
                ;;
            --no-dense-questions)
                ablation_args="$ablation_args --no-dense-questions"
                shift
                ;;
            --no-usefulness-judger)
                ablation_args="$ablation_args --no-usefulness-judger"
                shift
                ;;
            -*)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
            *)
                if [ -z "$max_samples" ]; then
                    max_samples="$1"
                    if ! [[ "$max_samples" =~ ^[0-9]+$ ]]; then
                        print_error "样本量必须是正整数: $max_samples"
                        exit 1
                    fi
                else
                    print_error "多余的参数: $1"
                    show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # 检查必需参数
    if [ -z "$max_samples" ]; then
        print_error "缺少必需参数: 样本量"
        show_help
        exit 1
    fi
    
    # 检查依赖
    if ! check_dependencies; then
        exit 1
    fi
    
    print_info "========== 实验配置 =========="
    print_info "样本量: $max_samples"
    print_info "跳过阶段: ${skip_stages[*]:-无}"
    print_info "测试模式: $([ "$test_mode" = true ] && echo "是" || echo "否")"
    print_info "过滤无答案数据: $([ "$no_filter" = true ] && echo "否" || echo "是")"
    print_info "增强模式: $enhance_mode"
    print_info "并发数: $batch_size"
    print_info "详细日志: $([ "$verbose" = true ] && echo "是" || echo "否")"
    print_info "工作目录: $SCRIPT_DIR"
    
    # 生成运行ID与运行目录（放在 Reports/experiments/datasets/runs 下）
    local ts=$(date +%Y%m%d-%H%M%S)
    local ablation_tag=""
    [[ "$ablation_args" == *"--no-query-rewriter"* ]] && ablation_tag+="-noqr"
    [[ "$ablation_args" == *"--no-dense-chunks"* ]] && ablation_tag+="-ndc"
    [[ "$ablation_args" == *"--no-dense-keywords"* ]] && ablation_tag+="-ndk"
    [[ "$ablation_args" == *"--no-dense-questions"* ]] && ablation_tag+="-ndq"
    [[ "$ablation_args" == *"--no-usefulness-judger"* ]] && ablation_tag+="-nuj"
    local run_id="${ts}-enh=${enhance_mode}-bs=${batch_size}${ablation_tag}"
    local RUNS_BASE_DIR="$SCRIPT_DIR/datasets/runs"
    local RUN_DIR="$RUNS_BASE_DIR/$run_id"
    
    # 计算各阶段是否执行
    stage_skipped() { local s="$1"; for st in "${skip_stages[@]}"; do [[ "$st" == "$s" ]] && return 0; done; return 1; }
    local do_converter=true; stage_skipped converter && do_converter=false
    local do_chunk=true; stage_skipped chunk && do_chunk=false
    local do_enhance=true; stage_skipped enhance && do_enhance=false
    local do_index=true; stage_skipped index && do_index=false
    local do_evaluation=true; stage_skipped evaluation && do_evaluation=false
    local do_advanced=true; stage_skipped advanced && do_advanced=false
    
    # 共享目录（默认路径）
    local SHARED_CONVERTED="$SCRIPT_DIR/datasets/converted"
    local SHARED_CHUNKED="$SCRIPT_DIR/datasets/chunked"
    local SHARED_ENHANCED="$SCRIPT_DIR/datasets/enhanced"
    local SHARED_KB="$SCRIPT_DIR/datasets/knowledge_bases"
    local SHARED_EVAL_RESULTS="$SCRIPT_DIR/datasets/rag_evaluation_results"
    local SHARED_ADV_RESULTS="$SCRIPT_DIR/datasets/advanced_evaluation_results"
    
    # 运行目录的子路径
    local RUN_CONVERTED="$RUN_DIR/converted"
    local RUN_CHUNKED="$RUN_DIR/chunked"
    local RUN_ENHANCED="$RUN_DIR/enhanced"
    local RUN_KB="$RUN_DIR/knowledge_bases"
    local RUN_EVAL_RESULTS="$RUN_DIR/rag_evaluation_results"
    local RUN_ADV_RESULTS="$RUN_DIR/advanced_evaluation_results"
    
    # 创建运行根目录
    mkdir -p "$RUN_DIR"
    print_info "运行ID: $run_id"
    print_info "运行目录: $RUN_DIR"
    
    # 导出环境变量以隔离本次运行的输入/输出
    # converter 输出目录
    export CONVERT_OUTPUT_DIR="$RUN_CONVERTED"
    
    # chunk 输入/输出目录
    if [ "$do_converter" = true ]; then
        export CHUNK_INPUT_DIR="$RUN_CONVERTED"
    else
        export CHUNK_INPUT_DIR="$SHARED_CONVERTED"
    fi
    export CHUNK_OUTPUT_DIR="$RUN_CHUNKED"
    
    # enhance 输入/输出目录
    if [ "$do_chunk" = true ]; then
        export ENHANCE_INPUT_DIR="$RUN_CHUNKED"
    else
        export ENHANCE_INPUT_DIR="$SHARED_CHUNKED"
    fi
    export ENHANCE_OUTPUT_DIR="$RUN_ENHANCED"
    
    # index 输入（增强数据）与输出（知识库基目录）
    if [ "$do_enhance" = true ]; then
        export INDEX_ENHANCED_DIR="$RUN_ENHANCED"
    else
        export INDEX_ENHANCED_DIR="$SHARED_ENHANCED"
    fi
    export INDEX_OUTPUT_BASE_DIR="$RUN_KB"
    
    # evaluation 问题文件目录、索引基目录、输出目录
    if [ "$do_converter" = true ]; then
        export EVAL_QUESTIONS_DIR="$RUN_CONVERTED"
    else
        export EVAL_QUESTIONS_DIR="$SHARED_CONVERTED"
    fi
    if [ "$do_index" = true ]; then
        export EVAL_INDEX_BASE_DIR="$RUN_KB"
    else
        export EVAL_INDEX_BASE_DIR="$SHARED_KB"
    fi
    export EVAL_OUTPUT_BASE_DIR="$RUN_EVAL_RESULTS"
    
    # advanced evaluation 输入/输出与KB基目录
    export RUN_RESULTS_DIR="$RUN_EVAL_RESULTS"
    export RUN_ADV_OUTPUT_DIR="$RUN_ADV_RESULTS"
    if [ "$do_index" = true ]; then
        export KB_BASE_DIR="$RUN_KB"
    else
        export KB_BASE_DIR="$SHARED_KB"
    fi
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 执行各阶段
    local stages=("converter" "chunk" "enhance" "index" "evaluation" "advanced")
    local failed_stages=()
    
    for stage in "${stages[@]}"; do
        # 检查是否跳过该阶段
        if [[ " ${skip_stages[@]} " =~ " ${stage} " ]]; then
            print_warning "跳过阶段: $stage"
            continue
        fi
        
        case $stage in
            converter)
                if ! run_converter "$max_samples" "$no_filter"; then
                    failed_stages+=("$stage")
                fi
                ;;
            chunk)
                if ! run_chunk; then
                    failed_stages+=("$stage")
                fi
                ;;
            enhance)
                if ! run_enhance "$enhance_mode" "$max_samples" "$verbose"; then
                    failed_stages+=("$stage")
                fi
                ;;
            index)
                if ! run_index "$test_mode" "$max_samples"; then
                    failed_stages+=("$stage")
                fi
                ;;
            evaluation)
                if ! run_evaluation "$max_samples" "$batch_size" "$ablation_args"; then
                    failed_stages+=("$stage")
                fi
                ;;
            advanced)
                if ! run_advanced_evaluation; then
                    failed_stages+=("$stage")
                fi
                ;;
        esac
        
        echo "" # 添加空行分隔
    done
    
    # 计算总耗时
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    local hours=$((total_time / 3600))
    local minutes=$(((total_time % 3600) / 60))
    local seconds=$((total_time % 60))
    
    # 总结
    print_info "========== 实验完成 =========="
    print_info "总耗时: ${hours}h ${minutes}m ${seconds}s"
    
    if [ ${#failed_stages[@]} -eq 0 ]; then
        print_success "所有阶段执行成功！"
        print_info "实验结果位置："
        print_info "  - 转换数据: /home/pushihao/RAG/Reports/experiments/datasets/converted/"
        print_info "  - 分块数据: /home/pushihao/RAG/Reports/experiments/datasets/chunked/"
        print_info "  - 增强数据: /home/pushihao/RAG/Reports/experiments/datasets/enhanced/"
        print_info "  - 索引文件: /home/pushihao/RAG/Reports/experiments/datasets/knowledge_bases/"
        print_info "  - 评估结果: /home/pushihao/RAG/Reports/experiments/datasets/rag_evaluation_results/"
        print_info "  - 高级分析: /home/pushihao/RAG/Reports/experiments/datasets/advanced_evaluation_results/"
        exit 0
    else
        print_error "以下阶段执行失败: ${failed_stages[*]}"
        exit 1
    fi
}

# 运行主函数
main "$@"