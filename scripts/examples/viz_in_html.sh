#!/bin/bash
set -x

cd /path/to/geo_hallucination_eval


emojis=(":fire:" ":star:" ":heart:" ":smile:" ":tada:" ":sparkles:" ":confetti_ball:" ":balloon:" ":cake:" ":gift:" ":sun_with_face:" ":cloud_with_lightning_and_rain:" ":snowman:" ":umbrella:" ":leaves:" ":rose:" ":tulip:" ":four_leaf_clover:" ":cherry_blossom:" ":bouquet:")
richf() {
    local random_emoji=${emojis[$RANDOM % ${#emojis[@]}]}
    rich -p -a heavy -j "$random_emoji $@"
}

input_file_or_dir="/secret/secret_sub/MetaFiles/secret_sub.jsonl",
output_dir="/workspace/linjh/CoT_Factory/assets/anal/subject"

richf "input_file_or_dir: $input_file_or_dir"
richf "output_dir: $output_dir"

python src/utils/to_html.py \
  --input_file_or_dir $input_file_or_dir \
  --output_dir $output_dir \