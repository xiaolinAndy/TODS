import json
import os
import random
import jieba
import sys

random.seed(2021)
def get_data(data_pth):
    with open(data_pth, 'r') as f:
        data = json.load(f)
    return data

def del_repeat(topics):
    new_topics = []
    for s in topics:
        if s not in new_topics:
            new_topics.append(s)
    return new_topics

def get_topics(data):
    topics = []
    for sample in data:
        for qa in sample['QA']:
            if qa['Topic'] == '' or qa['Topic'] is not None:
                topic = qa['Topic']
                if topic not in topics:
                    topics.append(topic)
    return topics

RANDOM = True

def convert_example_seg(data, new_name):
    new_data = []
    for sample in data:
        context = []
        for turn in sample['Dialogue']:
            tmp_utt = ''
            if turn['speaker'] == 'Q':
                tmp_utt += sample['QRole'] + ':'
            else:
                tmp_utt += '客服' + ':'
            tmp_utt += ''.join(turn['utterance'].split())
            tmp_utt = tmp_utt[:30]
            context.append(tmp_utt)
        top_sum = {}
        top_index = {}
        all_topics = []
        top_index_all = []

        for qa in sample['QA']:
            top_index_all.append(qa['QueSummUttIDs'] + qa['AnsSummLongUttIDs'])
            if qa['Topic'] != '' and qa['Topic'] is not None:
                all_topics.append(qa['Topic'])
                if qa['Topic'] not in top_sum:
                    top_sum[qa['Topic']] = qa['QASumm']
                    top_index[qa['Topic']] = qa['QueSummUttIDs'] + qa['AnsSummLongUttIDs']
                else:
                    top_sum[qa['Topic']] += qa['QASumm']
                    top_index[qa['Topic']] += qa['QueSummUttIDs'] + qa['AnsSummLongUttIDs']
        for topic, sum in top_sum.items():
            min_index = min(top_index[topic])
            max_index = max(top_index[topic])
            new_data.append({'Dialogue': context[min_index:max_index+1],
                             'Topic': topic,
                             'QASumm': sum,
                             'AllTopic': '，'.join(all_topics),
                             'AttIndexMin': min_index,
                             'AttIndexMax': max_index,
                             'AttIndexRange': list(range(min(top_index[topic]), max(top_index[topic]) + 1))})

    with open(new_name, 'w') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)


def convert_shannon_example_sync(data, new_name):
    new_data = []
    for sample in data:
        context = []
        for turn in sample['Dialogue']:
            tmp_utt = ''
            if turn['speaker'] == 'Q':
                tmp_utt += sample['QRole'] + ':'
            else:
                tmp_utt += '客服' + ':'
            tmp_utt += ''.join(turn['utterance'].split())
            tmp_utt = tmp_utt[:30]
            context.append(tmp_utt)
        top_sum = {}
        top_index = {}
        all_topics = []
        top_index_all = []

        for qa in sample['QA']:
            top_index_all.append(qa['QueSummUttIDs'] + qa['AnsSummLongUttIDs'])
            if qa['Topic'] != '' and qa['Topic'] is not None:
                all_topics.append(qa['Topic'])
                if qa['Topic'] not in top_sum:
                    top_sum[qa['Topic']] = qa['QASumm']
                    top_index[qa['Topic']] = qa['QueSummUttIDs'] + qa['AnsSummLongUttIDs']
                else:
                    top_sum[qa['Topic']] += qa['QASumm']
                    top_index[qa['Topic']] += qa['QueSummUttIDs'] + qa['AnsSummLongUttIDs']
        if len(top_sum) > 1:
            for topic, sum in top_sum.items():
                other_topics = list(top_sum.keys())
                other_topics.remove(topic)
                choose = random.randint(0, len(other_topics)-1)
                new_data.append({'Dialogue': context,
                                 'TrueTopic': topic,
                                 'FalseTopic': other_topics[choose],
                                 'QASumm': sum,
                                 'FalseQASumm': top_sum[other_topics[choose]],
                                 'AllTopic': '，'.join(all_topics),
                                 'AttIndexMin': min(top_index[topic]),
                                 'AttIndexMax': max(top_index[topic]),
                                 'AttShannonMask': 1})
        else:
            for topic, sum in top_sum.items():
                new_data.append({'Dialogue': context,
                                 'TrueTopic': topic,
                                 'FalseTopic': topic,
                                 'QASumm': sum,
                                 'FalseQASumm': sum,
                                 'AllTopic': '，'.join(all_topics),
                                 'AttIndexMin': min(top_index[topic]),
                                 'AttIndexMax': max(top_index[topic]),
                                 'AttShannonMask': 0})

    with open(new_name, 'w') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)






if __name__ == '__main__':
    if sys.argv[1] == 'aux2':
        if not os.path.exists('../data/CSDS/aux2/'):
            os.mkdir('../data/CSDS/aux2/')
        data = get_data('../data/CSDS/topic/train.json')
        topics = get_topics(data)
        for name in ['train', 'val', 'test']:
            data = get_data('../data/CSDS/topic/' + name + '.json')
            convert_example_seg(data, '../data/CSDS/aux2/' + name + '.json')
    elif sys.argv[1] == 'aux3':
        if not os.path.exists('../data/CSDS/aux3/'):
            os.mkdir('../data/CSDS/aux3/')
        data = get_data('../data/CSDS/topic/train.json')
        topics = get_topics(data)
        for name in ['train', 'val', 'test']:
            data = get_data('../data/CSDS/topic/' + name + '.json')
            convert_shannon_example_sync(data, '../data/CSDS/aux3/' + name + '.json')