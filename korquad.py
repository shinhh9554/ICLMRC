import os
import json


def json2jsonl(input_path, output_dir):
    with open(input_path, 'r', encoding='utf-8') as rf:
        raw_file = json.load(rf)

    examples = []

    for data in raw_file['data']:
        title = data['title']
        for paragraphs in data['paragraphs']:
            context = paragraphs['context']
            for qa in paragraphs['qas']:
                qas_id = qa['id']
                question = qa['question']
                answers = qa['answers']
                answer_text = str(answers[0]['text'])
                answer_start = int(answers[0]['answer_start'])

                examples.append({
                    'id': qas_id,
                    'title': title,
                    'context': context,
                    'question': question,
                    'answers': { "text": [answer_text], "answer_start": [answer_start] }
                })

    with open(os.path.join(output_dir, input_path.split('/')[-1]), 'w', encoding='utf-8') as wf:
        for example in examples:
            json.dump(example, wf, ensure_ascii=False)
            wf.write('\n')


if __name__ == '__main__':
    input_path = 'data/KorQuAD/nal2021_2_squad_dev.json'
    output_dir = 'data/validation'
    json2jsonl(input_path, output_dir)
