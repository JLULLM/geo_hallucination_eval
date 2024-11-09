cd /workspace/linjh/CoT_Factory

set -x

### setting-1
model_name='gpt-4o-2024-08-06'
max_workers=100
sample_test_size=200
prompt_class='verify_by_a'
prompt_version='v1017'
beam_seach_size=2
bfs_size=2
output_dir=./assets/output/optim_prompt/

### modify following lines ###
# exp_id=ai2d  
# init_prompt_evaluation_record='./assets/output/cot_1021/ai2d/benchmark_ai2d_000000010_gpt-4o-2024-08-06_cot_verify.json'

exp_id=raven  
init_prompt_evaluation_record='./assets/output/cot_1021/raven/benchmark_raven_000000005_gpt-4o-2024-08-06_cot_verify.json'
### modify above lines ###

optimize_step=5
init_prompt_template=./assets/output/optim_prompt/${exp_id}/init_prompt.txt

output_dir=${output_dir}/${exp_id}
if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi


python src/optimize_prompt.py \
    --optimize_step ${optimize_step} \
    --init_prompt_template ${init_prompt_template} \
    --init_prompt_evaluation_record ${init_prompt_evaluation_record} \
    --prompt_class ${prompt_class} \
    --prompt_version ${prompt_version} \
    --max_worker 100 \
    --sample_size 300 \
    --model_name ${model_name} \
    --sample_test_size ${sample_test_size} \
    2>&1 | tee ${output_dir}/optim.log