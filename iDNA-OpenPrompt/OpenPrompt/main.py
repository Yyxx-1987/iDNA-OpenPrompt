# step1:
import pandas as pd
import torch
from openprompt.data_utils import InputExample
import torch.nn as nn
import torch.nn.functional as F
from util import util_metric
from openprompt.prompts import ManualVerbalizer
from openprompt.plms import load_plm
from openprompt.prompts.manual_template import ManualTemplate
from transformers import BertTokenizer
from openprompt.prompts.manual_verbalizer import ManualVerbalizer
from openprompt.pipeline_base import PromptForClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from openprompt.pipeline_base import PromptDataLoader
import csv
classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "positive"
]

# 讀取數據放入到dataset中
def load_dna_sequences(lines):
    dataset = []
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()  # 一次性读取所有行

    for idx, line in enumerate(lines):
        line = line.strip()  # 去除可能的空白字符
        if line:  # 确保行不为空
            example = InputExample(guid=idx,
                                   text_a=line)
            dataset.append(example)
            # if idx < len(lines) - 1:
            #     dataset.append(",")  # 在最后一个元素之前添加逗号
    return dataset

# 示例：使用函数读取文件并生成数据集
# dna_file_path = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/txt/4mC/4mC_C.equisetifolia/test_neg.txt'  # 替换为您的文件路径
# dataset = load_dna_sequences(dna_file_path)
# # 20240105
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/test.tsv', sep='\t')
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/tsv/4mC/4mC_F.vesca/test.tsv', sep='\t')
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/tsv/4mC/4mC_S.cerevisiae/test.tsv', sep='\t')
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/tsv/4mC/4mC_Tolypocladium/test.tsv', sep='\t')
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/tsv/5hmC/5hmC_H.sapiens/test.tsv', sep='\t')
data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/tsv/5hmC/5hmC_M.musculus/test.tsv', sep='\t')

# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/tsv/6mA/6mA_A.thaliana/test.tsv', sep='\t')
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/tsv/6mA/6mA_C.elegans/test.tsv', sep='\t')
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/tsv//6mA/6mA_C.equisetifolia/test.tsv', sep='\t')
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/tsv/6mA/6mA_D.melanogaster/test.tsv', sep='\t')
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/tsv/6mA/6mA_F.vesca/test.tsv', sep='\t')
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/tsv/6mA/6mA_H.sapiens/test.tsv', sep='\t')
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/tsv/6mA/6mA_R.chinensis/test.tsv', sep='\t')
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/tsv/6mA/6mA_S.cerevisiae/test.tsv', sep='\t')
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/tsv/6mA/6mA_T.thermophile/test.tsv', sep='\t')
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/tsv/6mA/6mA_Tolypocladium/test.tsv', sep='\t')
# data = pd.read_csv('D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/tsv/6mA/6mA_Xoc BLS256/test.tsv', sep='\t')

# file_path1 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/txt/4mC/4mC_S.cerevisiae/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/txt/4mC/4mC_S.cerevisiae/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/txt/4mC/4mC_F.vesca/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/txt/4mC/4mC_F.vesca/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/txt/4mC/4mC_C.equisetifolia/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/txt/4mC/4mC_C.equisetifolia/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/txt/4mC/4mC_Tolypocladium/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/txt/4mC/4mC_Tolypocladium/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/txt/5hmC/5hmC_H.sapiens/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/txt/5hmC/5hmC_H.sapiens/train_neg.txt'  # 替换为你的文件路径
file_path1 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/txt/5hmC/5hmC_M.musculus/train_pos.txt'  # 替换为你的文件路径
file_path2 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS_1/txt/5hmC/5hmC_M.musculus/train_neg.txt'  # 替换为你的文件路径

# file_path1 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/txt/4mC/4mC_S.cerevisiae/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/txt/4mC/4mC_S.cerevisiae/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/txt/4mC/4mC_F.vesca/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/txt/4mC/4mC_F.vesca/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/txt/4mC/4mC_C.equisetifolia/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/txt/4mC/4mC_C.equisetifolia/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/txt/4mC/4mC_Tolypocladium/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA_OpenPrompt/data/DNA_MS/txt/4mC/4mC_Tolypocladium/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS/txt/5hmC/5hmC_H.sapiens/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS/txt/5hmC/5hmC_H.sapiens/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS/txt/5hmC/5hmC_M.musculus/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS/txt/5hmC/5hmC_M.musculus/train_neg.txt'  # 替换为你的文件路径
# #
# file_path1 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_A.thaliana/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_A.thaliana/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS/txt/6mA/6mA_C.elegans/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS/txt/6mA/6mA_C.elegans/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_C.equisetifolia/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_C.equisetifolia/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_D.melanogaster/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_D.melanogaster/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS/txt/6mA/6mA_F.vesca/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS/txt/6mA/6mA_F.vesca/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_H.sapiens/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_H.sapiens/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_R.chinensis/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_R.chinensis/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_S.cerevisiae/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_S.cerevisiae/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_T.thermophile/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_T.thermophile/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS/txt/6mA/6mA_Tolypocladium/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS/txt/6mA/6mA_Tolypocladium/train_neg.txt'  # 替换为你的文件路径
# file_path1 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_Xoc BLS256/train_pos.txt'  # 替换为你的文件路径
# file_path2 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS_1/txt/6mA/6mA_Xoc BLS256/train_neg.txt'  # 替换为你的文件路径

# # 提取label列和text列 由于TSV文件使用制表符作为字段分隔符，因此指定sep='\t'
labels = data['label']
lines = data['text']
dataset = load_dna_sequences(lines)
BERT_PATH = '../bert_base_uncased'
plm, tokenizer, model_config, WrapperClass = load_plm("bert", BERT_PATH)
#step3
# 创建分词器，指定自定义词汇表文件
tokenizer = BertTokenizer(vocab_file='dna_vocab.txt', do_lower_case=False)
promptTemplate = ManualTemplate(
    text = '{"placeholder":"text_a"} It was {"mask"}',
    tokenizer = tokenizer,
)
#step4
def split_into_6mers(dna_sequence):
    """将给定的 DNA 序列切分成 6-mer 序列。"""
    all_mers = set()
    mers=[dna_sequence[i:i+6] for i in range(0, len(dna_sequence)-21, 6)]
    all_mers.update(mers)
    mers = [dna_sequence[i:i + 6] for i in range(20, len(dna_sequence), 6)]
    return all_mers

def get_unique_6mers_from_file(file_path):
    """从文件中读取 DNA 序列，并获取唯一的 6-mer 集合。"""
    sequences = pd.read_csv(file_path, header=None)[0]  # 假设序列在第一列
    all_6mers = set()
    for sequence in sequences:
        sixmers = split_into_6mers(sequence)
        all_6mers.update(sixmers)
    return list(all_6mers)

# file_path1 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS/txt/4mC/4mC_C.equisetifolia/train_pos.txt'  # 替换为你的文件路径
unique_6mers_p = get_unique_6mers_from_file(file_path1)

# file_path2 = 'D:/yuxia/iDNA-OpenPrompt/data/DNA_MS/txt/4mC/4mC_C.equisetifolia/train_neg.txt'  # 替换为你的文件路径
unique_6mers_n = get_unique_6mers_from_file(file_path2)

# 假设我们有两个类别 'negative' 和 'positive'
# 你可以根据你的需要来分配 6-mer tokens 到不同的类别
label_words = {
    "negative": [unique_6mers_n[i] for i in range(len(unique_6mers_n))],
    "positive": [unique_6mers_p[i] for i in range(len(unique_6mers_p))],
}
promptVerbalizer = ManualVerbalizer(
    classes = ['negative', 'positive'],
    label_words = label_words,
    tokenizer = tokenizer,  # 之前定义的分词器
)
promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
)
#step 6
data_loader = PromptDataLoader(
    dataset=dataset,
    # labels=labels,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=2,
)
# step 7
# making zero-shot inference using pretrained MLM with prompt
# promptModel.eval()
def get_loss(logits, label, criterion):
    loss = criterion(logits.view(-1, 2), label)
    loss = (loss.float()).mean()
    loss = (loss - 0.06).abs() + 0.06
    return loss

if __name__ == '__main__':
    promptModel = promptModel.to(device)
    promptModel.train()
    # actual_labels = 1
    actual_labels = 0
    correct_predictions = 0
    total_predictions = 0
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(promptModel.parameters(), lr=0.0001, weight_decay=0.0025)
    optimizer = torch.optim.AdamW(promptModel.parameters(), lr=0.0004, weight_decay=0.0025)

    # 定义损失函数和优化器

    best_performance = 0
    total_loss = 0  # 跟踪总损失
    loss = 0
    correct_predictions = 0
    total_predictions = 0

    num_epochs = 1
    # 20240104
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)
    AUC0 = 0
    iter_size, corrects, avg_loss = 0, 0, 0
    repres_list = []
    label_list = []

    # 20240104
    bath_size = 2
    labels_tensor = torch.tensor(labels.values)
    labels_tensor.to(device)
    for epoch in range(1, num_epochs + 1):
        repres_list = []
        label_list = []
        i=0
        corrects =0
        iter_size = 0
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for batch in data_loader:
            batch.to(device)
            label_1 = labels_tensor[i:i+bath_size]
            i = i+bath_size
            logits = promptModel(batch)
            # # 20240108将提取的特征存入文件，以便于可视化展示
            # if epoch == 1:
            #     features = logits
            #     features1 = features.cpu()
            #     features2 = features1.detach().numpy()
            #     label_11 = label_1.cpu()
            #     label_12 = label_11.detach().numpy()
            #     output_file = "D:/yuxia/iDNA_OpenPrompt/OpenPrompt/result/umap/6mA_Tolypocladium_features_and_labels.csv"
            #
            #     # 将特征和标签写入CSV文件
            #     with open(output_file, mode='a', newline='') as file:
            #         writer = csv.writer(file)
            #
            #         # 写入文件的表头，如果需要的话
            #         # writer.writerow(['Feature1', 'Feature2', 'Feature3', 'Label'])
            #
            #         # 将特征和标签一行一行地写入文件
            #         for feature3, label2 in zip(features2, label_12):
            #             # feature = features2.numpy()
            #             writer.writerow([feature3, label2])
            #
            #     # print(f"特征和标签已写入文件: {output_file}")
            # # 20240108

            preds = torch.argmax(logits, dim=-1)
           # print(classes[preds])

            logits_shape = logits.size()
            # if actual_labels == 1:
            #     label = torch.ones(1, dtype=torch.long)
            # else:
            #     label = torch.zeros(1, dtype=torch.long)
            # correct_predictions += (classes[preds] == label)
            label_1 = label_1.to(device)
            loss = get_loss(logits, label_1, criterion)
            # 20240106
            # comparison_results = torch.gt(preds, label)
            # A = comparison_results.tolist()
            # 遍历数组A中的每个元素
            label_2 = label_1.tolist()
            preds_2 = preds.tolist()

            for value in range(len(preds)):
                if preds_2[value] == label_2[value]:
                # 更新总预测计数 # 如果值为1（或True），则增加正确预测计数
                    correct_predictions += 1
                total_predictions += 1
            # 20240106
            # if preds == label:
            #     correct_predictions += 1
            #     total_predictions += 1
           # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           # steps += 1
            total_loss += loss.item()
           # 20240104
            pred_prob_all = F.softmax(logits, dim=1)
            # Prediction probability [batch_size, class_num]
            pred_prob_positive = pred_prob_all[:, 1]
            # Probability of predicting positive classes [batch_size]
            pred_prob_sort = torch.max(pred_prob_all, 1)
            # The maximum probability of prediction in each sample [batch_size]
            pred_class = pred_prob_sort[1]
            # The location (class) of the predicted maximum probability in each sample [batch_size]
            corrects += (pred_class == label_1).sum()
            iter_size += label_1.shape[0]
            label_pred = torch.cat([label_pred, pred_class.float()])
            label_real = torch.cat([label_real, label_1.float()])
            pred_prob = torch.cat([pred_prob, pred_prob_positive])
            # label_pred.append(pred_class.float())
            # label_real.append(label_1.float())
            # pred_prob.append(pred_prob_positive.float())
        # label_pred1 = label_pred.view(-1)
        metric, roc_data, prc_data, AUC, fpr, tpr = util_metric.caculate_metric(label_pred, label_real, pred_prob)
        if AUC > AUC0:
            AUC0 = AUC
            fpr1 = fpr
            tpr1 = tpr
            util_metric.ROC(fpr1, tpr1, AUC0)
         #    with open("D:/yuxia/iDNA_OpenPrompt/OpenPrompt/result/roc/4mC_F.vesca_roc_curve_data.txt", "w") as file:
         #        # 遍历 fpr 和 tpr 列表，将数据写入文件
         #        for i in range(len(fpr)):
         #            file.write(f"FPR: {fpr[i]}, TPR: {tpr[i]}\n")
         # # 20240104

         # 打印平均损失
        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_predictions
        # accuracy = metric[0] ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC
        Precision = metric[1]
        Sensitivity = metric[2]
        Specificity = metric[3]
        F1 = metric[4]
        AUC = metric[5]
        MCC = metric[5]

        # print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(avg_loss,
        #                                                               accuracy,
        #                                                               correct_predictions,
        #                                                               iter_size))
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%,'
              f'Specificity: {Specificity * 100:.2f}%, Sensitivity: {Sensitivity * 100:.2f}%,'
              f' AUC: {AUC * 100:.2f}%, MCC: {MCC * 100:.2f}%')
