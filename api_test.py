import os, json, re
from string import Template
import openai  # https://platform.openai.com/docs/api-reference

openai.api_key = "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"
model = "text-davinci-003"
data_dir_path = "中文wiki测试数据"


# completion = openai.ChatCompletion.create(  # test api
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "user", "content": "Hello!"}
#     ]
# )
def get_data(data_dir_path):
    file_paths = os.listdir(data_dir_path)
    for test_file in file_paths[1:]:  # the 0st one is th json demo file
        content = json.loads(open(os.path.join(data_dir_path, test_file), "r").read())
        log_path = open(os.path.join("log", test_file), "a")
        yield content, log_path


def get_ent_predicate_set(segments):
    total_ent_types, total_predicates_types = set(), set()
    for seg in segments:
        entity_list, relation_list = seg.get("entity_list"), seg.get("relation_list")
        for ent_info in entity_list: total_ent_types.update(ent_info['type'])
        for rel_info in relation_list: total_predicates_types.add(rel_info['predicate'])
    return total_ent_types, total_predicates_types


if __name__ == "__main__":
    entity_template = Template(
        "你是一个自然语言标注工程师，需要标注文中出现的实体和对应的实体类型，关系和对应的关系类型，和出现的所有的事件。\n"
        "现在给你的需要标注的文本为：$input，"
        "可能出现的实体类型为：$entity_set，"
        "可能出现的关系类型为：$rel_set。"
        '一个实体可以属于多个实体类型，标注结果使用数组结构的json字符串，'
        '每个事件使用一个字典表示，每个事件字典中包含了 event_type（事件类型）和argument_list（论元列表）。'
        '同时，argument_list 中的每个元素为一个论元字典，其中包含了 text 字段，char_span字段（表示起始和终止位置）和type（论元类型）。'
        '一个例子为：'
        '{"entity_list": [{"text": "中国","type": [ "territorial entity type", ] } ],'
        '"relation_list": [{"subject": "张峻","predicate": "participant","object": "协和组织"}],'
        '"event_list": {[{"event_type": "EquityOverweight", "argument_list": [{"text": "张峻", "char_span": [17, 18], "type": "EquityHolder"}, {"text": "50000股", "char_span": [39, 43], "type": "TradedShares"}, ...]}'
        '}'
        "严格按照上面的标注结果进行标注，你的 json 标注结果为：")
    
    # event_template = Template(
    #     '假设你是一个自然语言标注工程师，需要标注文中出现的事件。'
    #     '尽管没有告诉你可能出现的事件类型，需要抽取出文中出现的所有的事件，每个事件使用一个字典表示，每个事件字典中包含了 event_type（事件类型）和argument_list（论元列表）。'
    #     '同时，argument_list 中的每个元素为一个论元字典，其中包含了 text 字段，char_span字段（表示起始和终止位置）和type（论元类型）'
    #     '一个例子为：输入为： "2008年12月31日，公司监事长张峻根据原计划在当日从二级市场购入本公司股票50000股，..." '
    #     '标注结果为：{[{"event_type": "EquityOverweight", "argument_list": [{"text": "张峻", "char_span": [17, 18], "type": "EquityHolder"}, {"text": "50000股", "char_span": [39, 43], "type": "TradedShares"}, {"text": "2008年12月31日", "char_span": [0, 11], "type": "StartDate"}, {"text": "2008年12月31日", "char_span": [0, 11], "type": "EndDate"}]}\''
    #     '现在给你的需要标注的文本为：$input ，'
    #     '严格按照上面的标注结果进行标注，你的 json 标注结果为：'
    # )
    
    for content, log_writer in get_data(data_dir_path):
        _id, doc_id = content.get("_id"), content.get("doc_id")
        segments = content.get("segments")
        ent_types, predicates_types = get_ent_predicate_set(segments)
        
        for seg in segments:
            para_id, paragraph, entity_list, relation_list = seg.get("para_id"), seg.get("paragraph"), seg.get(
                "entity_list"), seg.get("relation_list")
            prompt = entity_template.substitute(input=paragraph, entity_set=ent_types, rel_set=predicates_types)
            
            completion = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=2048,
                # top_p=0.9,
                # temperature=0.2
            )
            
            predications = re.sub("\n|`", "", completion.choices[0].text)
            seg[f"complete-{model}"] = predications
            log_writer.write(json.dumps(seg, ensure_ascii=False) + "\n")  # 先存模型的输出
            try:
                predications = json.loads(predications)
                predications = [[i['text'] + j for j in i['type']] for i in predications]  # 输出合法的话存json
                labels = [[i['text'] + j for j in i['type']] for i in entity_list]
            
            except:
                print(f"{para_id} completion could not save as json.")
    
    # print(completion.choices[0].text)
