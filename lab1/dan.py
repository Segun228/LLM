import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import re
import os
from kaggle_secrets import UserSecretsClient

# === НАСТРОЙКИ ===
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN") if user_secrets else None

if not torch.cuda.is_available():
    raise RuntimeError("⚠️ GPU не обнаружен! Включи 'Accelerator T4 x2' в настройках.")

model_id = "Qwen/Qwen3-14B" 
input_file = "/kaggle/input/maindata/LR1.csv" 
submission_file = "/kaggle/working/submission.csv"
reasoning_file = "/kaggle/working/reasoning_log.csv"

# === ЗАГРУЗКА МОДЕЛИ ===
print(f"🚀 Загрузка {model_id}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token,
    trust_remote_code=True
)

# === ФУНКЦИЯ ОБРАБОТКИ ===
def chat_with_model(category, question, options_str):
    options_list = options_str.split(";;;")
    num_options = len(options_list)

    system_role = f"""<system>
  <role>Ты эксперт в области {category}.</role>

  <task>
    <goal>Выбери единственный правильный вариант ответа.</goal>
    <method>Сначала сделай пошаговое рассуждение: пройди по каждому варианту и объясни, почему он подходит или не подходит.</method>
  </task>

  <constraints>
    <numbering>Варианты нумеруются с 0.</numbering>
    <answer_format>В самом конце выведи ТОЛЬКО индекс правильного варианта в двойных квадратных скобках: [[N]]. Никакого текста после.</answer_format>
    <options_note>Если вариант содержит запятую (например, "яблоко, банан"), это ОДИН вариант ответа, а не несколько.</options_note>
  </constraints>

  <example>
    <question>Какая планета ближе всего к Солнцу?</question>
    <options>
      <option index="0">Венера</option>
      <option index="1">Марс</option>
      <option index="2">Земля</option>
      <option index="3">Меркурий</option>
    </options>
    <reasoning>Венера — вторая планета от Солнца, Марс — четвертая, Земля — третья. Меркурий находится ближе всех к Солнцу.</reasoning>
    <final_answer>[[3]]</final_answer>
  </example>
</system>"""

    formatted_options = "\n".join([f"{i}. {opt}" for i, opt in enumerate(options_list)])
    prompt = f"Вопрос: {question}\n\nВарианты:\n{formatted_options}\n\nРассуждение и ответ:"

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2500,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    # Парсинг ответа
    match = re.search(r'\[\[(\d+)\]\]', response)
    ans_index = int(match.group(1)) if match else None
    
    if ans_index is None:
        match = re.search(r'(?:Ответ|Answer)[:\s\-]+(\d+)', response[-500:], re.I)
        ans_index = int(match.group(1)) if match else None
    
    torch.cuda.empty_cache()
    return ans_index, response

# === ОБРАБОТКА ===
df = pd.read_csv(input_file)
df['id'] = df.get('Unnamed: 0', df.index)
df['options'] = df['options'].apply(lambda x: ";;;".join(re.findall(r"'([^']*)'", x)))

# Создаём пустые файлы
pd.DataFrame(columns=['id', 'answer']).to_csv(submission_file, index=False)
pd.DataFrame(columns=['question_id', 'model_reasoning']).to_csv(reasoning_file, index=False)

print(f"🏁 Обработка {len(df)} вопросов...\n")

for _, row in df.iterrows():
    try:
        ans, reasoning = chat_with_model(row['category'], row['question'], row['options'])
        
        print(f"Вопрос {row['id']}: {ans if ans is not None else 0}")
        
        pd.DataFrame([{'id': row['id'], 'answer': ans or 0}]).to_csv(
            submission_file, mode='a', header=False, index=False
        )
        pd.DataFrame([{'question_id': row['id'], 'model_reasoning': reasoning.replace('\n', ' ')}]).to_csv(
            reasoning_file, mode='a', header=False, index=False
        )
        
    except Exception as e:
        print(f"Вопрос {row['id']}: ОШИБКА - {e}")

print(f"\n✅ Готово!\n📄 {submission_file}\n📄 {reasoning_file}")