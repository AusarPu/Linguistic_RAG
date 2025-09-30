#!/bin/bash

# 高级评估运行脚本
# 用于运行advanced_evaluation.py对RAG评估结果进行高级分析

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADVANCED_EVAL_SCRIPT="$SCRIPT_DIR/evaluation/advanced_evaluation.py"
RESULTS_DIR="$SCRIPT_DIR/rag_evaluation_results"
OUTPUT_DIR="$SCRIPT_DIR/advanced_evaluation_results"

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
    echo "高级评估运行脚本"
    echo ""
    echo "用法:"
    echo "  $0 [选项] [数据集名称]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -s, --sample        运行样本评估（默认）"
    echo "  -f, --full          运行完整评估"
    echo "  -a, --all           评估所有数据集"
    echo "  -l, --limit N       限制处理的结果数量（用于测试）"
    echo "  --list              列出可用的数据集"
    echo ""
    echo "数据集名称:"
    echo "  hotpotqa           HotpotQA数据集"
    echo "  ms_marco           MS MARCO数据集"
    echo "  natural_questions  Natural Questions数据集"
    echo "  triviaqa           TriviaQA数据集"
    echo ""
    echo "示例:"
    echo "  $0                              # 评估所有数据集的样本结果"
    echo "  $0 -s hotpotqa                  # 评估HotpotQA的样本结果"
    echo "  $0 -f natural_questions         # 评估Natural Questions的完整结果"
    echo "  $0 -a -f                        # 评估所有数据集的完整结果"
    echo "  $0 -l 10 hotpotqa               # 仅评估HotpotQA的前10个结果"
}

# 列出可用数据集
list_datasets() {
    print_info "可用的数据集:"
    if [ -d "$RESULTS_DIR" ]; then
        for dataset_dir in "$RESULTS_DIR"/*; do
            if [ -d "$dataset_dir" ]; then
                dataset_name=$(basename "$dataset_dir")
                echo "  - $dataset_name"
                
                # 检查可用的结果文件
                sample_file="$dataset_dir/sample_results.json"
                full_file="$dataset_dir/full_results.json"
                
                if [ -f "$sample_file" ]; then
                    echo "    ✓ 样本结果可用"
                fi
                if [ -f "$full_file" ]; then
                    echo "    ✓ 完整结果可用"
                fi
                if [ ! -f "$sample_file" ] && [ ! -f "$full_file" ]; then
                    echo "    ✗ 无可用结果文件"
                fi
            fi
        done
    else
        print_error "结果目录不存在: $RESULTS_DIR"
    fi
}

# 检查依赖
check_dependencies() {
    # 检查Python脚本是否存在
    if [ ! -f "$ADVANCED_EVAL_SCRIPT" ]; then
        print_error "高级评估脚本不存在: $ADVANCED_EVAL_SCRIPT"
        return 1
    fi
    
    # 检查结果目录是否存在
    if [ ! -d "$RESULTS_DIR" ]; then
        print_error "评估结果目录不存在: $RESULTS_DIR"
        return 1
    fi
    
    # 检查Python是否可用
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装或不在PATH中"
        return 1
    fi
    
    return 0
}

# 运行单个数据集的评估
run_evaluation() {
    local dataset_name=$1
    local eval_type=$2  # "sample" 或 "full"
    local limit=$3      # 可选的限制数量
    
    print_info "开始评估数据集: $dataset_name ($eval_type)"
    
    # 确定输入文件
    local input_file
    if [ "$eval_type" = "sample" ]; then
        input_file="$RESULTS_DIR/$dataset_name/sample_results.json"
    else
        input_file="$RESULTS_DIR/$dataset_name/full_results.json"
    fi
    
    # 检查输入文件是否存在
    if [ ! -f "$input_file" ]; then
        print_error "输入文件不存在: $input_file"
        return 1
    fi
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR/$dataset_name"
    
    # 确定输出文件
    local output_file="$OUTPUT_DIR/$dataset_name/advanced_${eval_type}_results.json"
    
    # 构建命令
    local cmd="python3 \"$ADVANCED_EVAL_SCRIPT\" \"$input_file\" \"$output_file\""
    
    # 添加限制参数
    if [ -n "$limit" ] && [ "$limit" -gt 0 ]; then
        cmd="$cmd --limit $limit"
        print_info "限制处理数量: $limit"
    fi
    
    print_info "执行命令: $cmd"
    
    # 执行评估
    if eval $cmd; then
        print_success "数据集 $dataset_name ($eval_type) 评估完成"
        print_info "结果保存到: $output_file"
        return 0
    else
        print_error "数据集 $dataset_name ($eval_type) 评估失败"
        return 1
    fi
}

# 主函数
main() {
    local eval_type="sample"  # 默认为样本评估
    local eval_all=false
    local limit=""
    local datasets=()
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -s|--sample)
                eval_type="sample"
                shift
                ;;
            -f|--full)
                eval_type="full"
                shift
                ;;
            -a|--all)
                eval_all=true
                shift
                ;;
            -l|--limit)
                limit="$2"
                if ! [[ "$limit" =~ ^[0-9]+$ ]]; then
                    print_error "限制数量必须是正整数: $limit"
                    exit 1
                fi
                shift 2
                ;;
            --list)
                list_datasets
                exit 0
                ;;
            -*)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
            *)
                datasets+=("$1")
                shift
                ;;
        esac
    done
    
    # 检查依赖
    if ! check_dependencies; then
        exit 1
    fi
    
    # 确定要评估的数据集
    if [ "$eval_all" = true ]; then
        # 评估所有数据集
        datasets=()
        for dataset_dir in "$RESULTS_DIR"/*; do
            if [ -d "$dataset_dir" ]; then
                datasets+=($(basename "$dataset_dir"))
            fi
        done
        
        if [ ${#datasets[@]} -eq 0 ]; then
            print_error "未找到任何数据集"
            exit 1
        fi
        
        print_info "将评估所有数据集: ${datasets[*]}"
    elif [ ${#datasets[@]} -eq 0 ]; then
        # 如果没有指定数据集，默认评估所有
        print_warning "未指定数据集，将评估所有可用数据集"
        for dataset_dir in "$RESULTS_DIR"/*; do
            if [ -d "$dataset_dir" ]; then
                datasets+=($(basename "$dataset_dir"))
            fi
        done
    fi
    
    # 验证数据集名称
    for dataset in "${datasets[@]}"; do
        if [ ! -d "$RESULTS_DIR/$dataset" ]; then
            print_error "数据集目录不存在: $RESULTS_DIR/$dataset"
            exit 1
        fi
    done
    
    print_info "评估配置:"
    print_info "  评估类型: $eval_type"
    print_info "  数据集: ${datasets[*]}"
    if [ -n "$limit" ]; then
        print_info "  限制数量: $limit"
    fi
    print_info "  输出目录: $OUTPUT_DIR"
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    
    # 运行评估
    local success_count=0
    local total_count=${#datasets[@]}
    
    for dataset in "${datasets[@]}"; do
        print_info "正在处理数据集: $dataset (${success_count}/${total_count})"
        
        if run_evaluation "$dataset" "$eval_type" "$limit"; then
            ((success_count++))
        fi
        
        echo "" # 添加空行分隔
    done
    
    # 总结
    print_info "评估完成!"
    print_success "成功评估: $success_count/$total_count 个数据集"
    
    if [ $success_count -lt $total_count ]; then
        print_warning "部分数据集评估失败，请检查日志"
        exit 1
    else
        print_success "所有数据集评估成功完成!"
        print_info "结果保存在: $OUTPUT_DIR"
    fi
}

# 运行主函数
main "$@"