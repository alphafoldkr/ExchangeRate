import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from dateutil.relativedelta import relativedelta
from datetime import *

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tqdm import tqdm
import os
import data_preprocessing
import argparse


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dimension = args.dimension
        self.productcode_embeddings = nn.Embedding(279, dimension)
        self.chargetypeid_embeddings = nn.Embedding(33, dimension)
        self.promo_100_embeddings = nn.Embedding(2, dimension)
        self.coinReceived_embeddings = nn.Embedding(2, dimension)
        self.devicetypeid_embeddings = nn.Embedding(23, dimension)
        self.isauth_embeddings = nn.Embedding(2, dimension)
        self.gender_embeddings = nn.Embedding(3, dimension) 
        self.agegroup_embeddings = nn.Embedding(26, dimension)
        self.concurrentwatchcount_embeddings = nn.Embedding(4, dimension)
        self.REG_CNT_embeddings = nn.Embedding(5, dimension)
        self.PRD_CNT_embeddings = nn.Embedding(4, dimension)
        self.BM_CNT_embeddings = nn.Embedding(1098, dimension)
        self.CH_CNT_embeddings = nn.Embedding(5, dimension)
        self.PRG_CNT_embeddings = nn.Embedding(138, dimension)
        self.DEV_CNT_embeddings = nn.Embedding(12, dimension)
        self.DATE_CNT_embeddings = nn.Embedding(32, dimension)
        self.COIN_CNT_embeddings = nn.Embedding(16, dimension)
        self.v100_embeddings = nn.Embedding(434, dimension)
        self.v300_embeddings = nn.Embedding(599, dimension)
        self.vn_embeddings = nn.Embedding(596, dimension)

        self.dropout = nn.Dropout(args.dropout)
        
        self.LayerNorm = nn.LayerNorm([args.dimension*21], eps=1e-05, elementwise_affine=True)
        self.bottom_fc1 = nn.Linear(16, 2048)  
        self.bottom_fc2 = nn.Linear(2048, dimension)

        self.top_fc1 = nn.Linear(args.dimension*21, 512)
        self.top_fc2 = nn.Linear(512, 256)
        self.top_fc3 = nn.Linear(256, 128)
        self.top_fc4 = nn.Linear(128, 64)
        self.top_fc5 = nn.Linear(64, 1)

    def forward(self, emb_input, dense_input):
        #Embedding Layer
        emb0 = self.productcode_embeddings(emb_input[:, 0])
        emb1 = self.chargetypeid_embeddings(emb_input[:, 1])
        emb2 = self.promo_100_embeddings(emb_input[:, 2])
        emb3 = self.coinReceived_embeddings(emb_input[:, 3])
        emb4 = self.devicetypeid_embeddings(emb_input[:, 4])
        emb5 = self.isauth_embeddings(emb_input[:, 5])
        emb6 = self.gender_embeddings(emb_input[:, 6])
        emb7 = self.agegroup_embeddings(emb_input[:, 7])
        emb8 = self.concurrentwatchcount_embeddings(emb_input[:, 8])
        emb9 = self.REG_CNT_embeddings(emb_input[:, 9])
        emb10 = self.PRD_CNT_embeddings(emb_input[:, 10])
        emb11 = self.BM_CNT_embeddings(emb_input[:, 11])
        emb12 = self.CH_CNT_embeddings(emb_input[:, 12])
        emb13 = self.PRG_CNT_embeddings(emb_input[:, 13])
        emb14 = self.DEV_CNT_embeddings(emb_input[:, 14])
        emb15 = self.DATE_CNT_embeddings(emb_input[:, 15])
        emb16 = self.COIN_CNT_embeddings(emb_input[:, 16])
        emb17 = self.v100_embeddings(emb_input[:, 17])
        emb18 = self.v300_embeddings(emb_input[:, 18])
        emb19 = self.vn_embeddings(emb_input[:, 19])
                
        #Bottom FC Layer
        bottom = F.gelu(self.bottom_fc1(dense_input))
        bottom = self.dropout(bottom)
        bottom = F.gelu(self.bottom_fc2(bottom))
        
        #Interact Features 
        top_input = torch.cat((bottom,emb0,emb1,emb2,emb3,emb4,emb5,emb6,emb7,emb8,emb9,emb10,emb11,emb12,emb13,emb14,emb15,emb16,emb17,emb18,emb19), dim=1)
        top_input = self.LayerNorm(top_input)
        # ##### Dot Product #####
        # T = top_input.view((args.batch_size, -1, args.dimension))
        # Z = torch.bmm(T, torch.transpose(T, 1, 2))
        # _, ni, nj = Z.shape
        # li = torch.tensor([i for i in range(ni) for j in range(i)])
        # lj = torch.tensor([j for i in range(nj) for j in range(i)])
        # Zflat = Z[:, li, lj]
        # top_input = torch.cat([bottom] + [Zflat], dim=1)

        #Top FC Layer
        output = F.gelu(self.top_fc1(top_input))
        output = self.dropout(output)
        output = F.gelu(self.top_fc2(output))
        output = self.dropout(output)
        output = F.gelu(self.top_fc3(output))
        output = self.dropout(output)
        output = F.gelu(self.top_fc4(output))
        output = F.gelu(self.top_fc5(output))

        return output

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--trained_model', type=str, default="weight_decay=0.21,dropout=0.3,threshold=0.25,e=20,b=64,lr=0.001,dim=64_v3_adamP.pt")
    args = parser.parse_args()
    print(args)

    if args.device == "cpu":
        device = torch.device('cpu')
        print("The targe device is CPU")
    elif args.device == "cuda":
        device = torch.device('cuda')
        print("The target device is GPU")

    # Data Load
    if not os.path.isfile("predict_dataset.csv"):
        data_preprocessing.save_predict_dataset()
        print("Predict Dataset is saved")
    df_P = pd.read_csv("predict_dataset.csv")
    
    # Split Train/Test Data 
    X_predict = df_P.drop(["uno","registerdate","enddate"], axis=1)
    X_predict = torch.tensor(X_predict.values.tolist(), dtype=torch.float32).to(device)
    
    net = Net().to(device)
    net = torch.load('./model/'+args.trained_model)
    net.eval()
    y_pred = []
    with torch.no_grad():
        for inputs in X_predict:
            inputs.resize_(1, list(inputs.size())[0])
            emb_input = inputs[:, 0:20].type(torch.long)
            dense_input = inputs[:, 20:]
            outputs = net(emb_input,dense_input)
            outputs = torch.where(outputs.double() >= args.threshold, 1., 0.)
            y_pred.append(outputs.item())
    
    # 결과 제출 답안지 불러오기
    ds_sheet = "./Submission/CDS_submission.csv"
    df_sheet = pd.read_csv(ds_sheet)
    df_sheet.drop('CHURN', axis=1, inplace=True)
    
    # 답안지에 답안 표기
    df_productCode = pd.read_csv("Productcode.csv")
    df_productCode = df_productCode[:279]
    productcode = dict(zip(df_productCode.index, df_productCode.Productcode))
    df_P.rename(columns = {'productcode' : 'productcode'}, inplace = True)
    df_P["productcode"]= df_P['productcode'].map(productcode)

    df_result = df_P.loc[:,('uno','registerdate','productcode')]
    df_result['registerdate'] = pd.to_datetime(df_result['registerdate'])
    df_result['KEY']  = df_result['uno'] + '|' + df_result['registerdate'].dt.strftime('%y-%m-%d %I:%M:%S') + '|' + df_result['productcode']
    df_result['CHURN'] = pd.DataFrame(y_pred)
    df_result = df_result.loc[:,('KEY','CHURN')]
    df_answer_sheet = pd.merge(df_sheet, df_result, on='KEY', how='left')

    # 답안지 제출 파일 생성하기
    ds_answer_sheet = "./Submission/CDS_submission_inposter_210727.csv"
    df_answer_sheet.to_csv(ds_answer_sheet, index=False, encoding='utf8')
