from pathlib import Path
import json

folder_path = Path("./token_consumption_build_memory_locomo_fusionrag/")  # 替换为你的文件夹路径

# 递归获取所有 .json 文件（包括子目录）
json_files = list(folder_path.rglob("*.json"))

all_common_prefix = []
all_reuse_raw = []
all_reuse_memory = []
all_first_occur = []


all_filed = 0
# 打印文件路径（Path 对象）
for file in json_files:
    try:
        with open(file, "r") as f:
            all_prompt_len = 0
            json_data = json.load(f)
            fusionrag_stats = json_data["fusionrag_stats"]
            all_len = 0
            for stat in fusionrag_stats:
                all_len += stat["query_len"] + stat["system_len"] + stat["origin_text_list_len"]

                all_common_prefix.append(stat["system_len"])
                if stat["reuse_type"] == "reuse_prefill":
                    all_reuse_raw.append(stat["origin_text_list_len"])
                elif stat["reuse_type"] == "reuse_decode":
                    all_reuse_memory.append(stat["origin_text_list_len"])
                elif stat["reuse_type"] == "reuse_mix":
                    all_reuse_memory.append(stat["reuse_type_detail_decode"])
                    all_reuse_raw.append(stat["reuse_type_detail_prefill"])
                all_first_occur.append(stat["query_len"])
            all_filed += 1
            print(f"all_len: {all_len}")
            print(f"total_llm_tokens: {json_data['prompt_tokens']}")
            print(f"="*100)
    except Exception as e:
        print(f"skip file: {file}")

print(f"common_prefix: {sum(all_common_prefix)/all_filed}")
print(f"reuse raw: {sum(all_reuse_raw)/all_filed}")
print(f"reuse memory {sum(all_reuse_memory)/all_filed}")
print(f"first_occur: {sum(all_first_occur)/all_filed}")

print(f"build tokens: {(sum(all_reuse_raw)+sum(all_reuse_memory)+sum(all_first_occur))/all_filed:,.0f}")
print(f"build tokens fusionrag: {((sum(all_reuse_raw)+sum(all_reuse_memory))*0.3+sum(all_first_occur))/all_filed:,.0f}")

print(f"all token count: {(sum(all_common_prefix) + sum(all_reuse_raw) + sum(all_reuse_memory) + sum(all_first_occur))/all_filed}")