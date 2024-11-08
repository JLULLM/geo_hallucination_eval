cd /workspace/linjh/CoT_Factory

set -x

emojis=(":fire:" ":star:" ":heart:" ":smile:" ":tada:" ":sparkles:" ":confetti_ball:" ":balloon:" ":cake:" ":gift:" ":sun_with_face:" ":cloud_with_lightning_and_rain:" ":snowman:" ":umbrella:" ":leaves:" ":rose:" ":tulip:" ":four_leaf_clover:" ":cherry_blossom:" ":bouquet:")
richf() {
    local random_emoji=${emojis[$RANDOM % ${#emojis[@]}]}
    rich -p -a heavy -j "$random_emoji $@"
}


input_meta_data=${1}
output_subdir_name=${2}

output_dir=./assets/output/cot_1021/${output_subdir_name}
richf "output_dir: ${output_dir}"

### setting-1
prompt_class=only_q
# prompt_version=v1017
# prompt_version=v1021_top1
prompt_version=${3:-v1021_top1}
echo "prompt_version: ${prompt_version}"
# output_dir=${output_dir}/${prompt_class}/${prompt_version}
if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi

# get file stem
file_stem=$(basename ${input_meta_data} .jsonl)

# do_filter_multi_img=True
# do_filter_single_question=False
# do_filter_img2img=False

gen_model_name=gpt-4o-2024-08-06
python src/cot_1_generate.py \
    --input_file ${input_meta_data} \
    --output_dir ${output_dir} \
    --prompt_class ${prompt_class} \
    --prompt_version ${prompt_version} \
    --max_worker 100 \
    --model_name ${gen_model_name} \
    2>&1 | tee ${output_dir}/cot_1_generate_${file_stem}.log
    # --do_filter_multi_img ${do_filter_multi_img} \
    # --do_filter_single_question ${do_filter_single_question} \
    # --do_filter_img2img ${do_filter_img2img} \


verifiy_model_name=gpt-4o-2024-08-06
modify_dir=${output_dir}
verifiy_prompt_class=verify_by_a
verifiy_prompt_version=v1017
python src/cot_2_verify.py \
    --modify_dir ${modify_dir} \
    --prompt_class ${verifiy_prompt_class} \
    --prompt_version ${verifiy_prompt_version} \
    --model_name ${verifiy_model_name} \
    --max_worker 100 \
    2>&1 | tee ${modify_dir}/cot_2_verify_${file_stem}.log