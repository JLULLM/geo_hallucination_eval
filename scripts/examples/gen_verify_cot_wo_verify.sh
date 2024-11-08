cd /workspace/linjh/CoT_Factory

set -x
emojis=(":fire:" ":star:" ":heart:" ":smile:" ":tada:" ":sparkles:" ":confetti_ball:" ":balloon:" ":cake:" ":gift:" ":sun_with_face:" ":cloud_with_lightning_and_rain:" ":snowman:" ":umbrella:" ":leaves:" ":rose:" ":tulip:" ":four_leaf_clover:" ":cherry_blossom:" ":bouquet:")
richf() {
    local random_emoji=${emojis[$RANDOM % ${#emojis[@]}]}
    rich -p -a heavy -j "$random_emoji $@"
}

# geo3k_meta_data=/workspace/image_sft/datav20240920/BenchmarksQA/en/geo3k/MetaFiles/benchmark_geo3k_000000000.jsonl 
# ai2d_meta_data=/workspace/image_sft/datav20240920/BenchmarksQA/en/ai2d/MetaFiles/benchmark_ai2d_000000000.jsonl
# input_meta_data=${geo3k_meta_data}

# input_meta_data=${1:-/workspace/image_sft/datav20240920/SFT/Subject/xueke_0927/MetaFiles/xueke_000001.jsonl}



### setting-1
# prompt_class=only_q
# prompt_version=v1017
# prompt_version=v1021_top1
# prompt_version=${3:-v1021_top1}

prompt_class=both_qa
# prompt_version=${2:-v1017_zh_multi_ques}
prompt_version=${2:-v1017_zh_multi_ques_tune}
richf 'prompt_class: ' ${prompt_class}
richf 'prompt_version: ' ${prompt_version}
gen_model_name=gpt-4o-2024-08-06


# 遍历 0927 0917 0911 0902
# subdir=xueke_0917
# Chemistry  Code  Math  Physics  tiku_0729  tiku_0820  xingce_0806  xingce_0822  xueke_0902  xueke_0911  xueke_0917  xueke_0927  xueke_1016  xueke_1023
# for subdir in Chemistry/tiku-chemistry-large Math/tiku-math Physics/tiku-physics-large; do
for subdir in Physics/tiku-physics-large; do
    input_dir="/workspace/image_sft/datav20240920/SFT/Subject/${subdir}/MetaFiles/"
    output_dir=./assets/output/cot_subject_1026/${subdir}

    # output_dir=${output_dir}/${prompt_class}/${prompt_version}
    if [ ! -d ${output_dir} ]; then
        mkdir -p ${output_dir}
    fi

    for input_meta_data in ${input_dir}*.jsonl; do
        # get file stem
        file_stem=$(basename ${input_meta_data} .jsonl)

        # 保存文件的逻辑更改为：output_dir/{input_file}_{model_name}_{output_suffix}
        # 如果output_dir设定为原来的input_meta_data所在的目录，就可以实现了原来的modify_dir的逻辑
        
                # --do_show_html True \
        python src/cot_1_generate.py \
            --input_file ${input_meta_data} \
            --output_dir ${output_dir} \
            --output_file stick_add_model \
            --output_suffix _cot.json \
            --prompt_class ${prompt_class} \
            --prompt_version ${prompt_version} \
            --max_worker 100 \
            --model_name ${gen_model_name} \
            2>&1 | tee ${output_dir}/cot_1_generate_${file_stem}.log
        
        if output_file=stick_add_model; then
            output_file=${output_dir}/${file_stem}_${gen_model_name}_cot.json
        fi
        richf 'output_file: ' ${output_file}
        # 为了不大幅修改之前的普通的cot，将parse过程单独拎出来，这样生成和验证之间需要各种花样操作，就可以在这里定义
        
        python src/cot_15_parse.py \
            --input_file ${output_file} \
            2>&1 | tee ${output_dir}/cot_15_parse_${file_stem}.log
    done

done
# verifiy_model_name=gpt-4o-2024-08-06
# modify_dir=${output_dir}
# verifiy_prompt_class=verify_by_a
# verifiy_prompt_version=v1017
# python src/cot_2_verify.py \
#     --modify_dir ${modify_dir} \
#     --prompt_class ${verifiy_prompt_class} \
#     --prompt_version ${verifiy_prompt_version} \
#     --model_name ${verifiy_model_name} \
#     --max_worker 100 \
#     2>&1 | tee ${modify_dir}/cot_2_verify_${file_stem}.log
