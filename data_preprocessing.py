import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from dateutil.relativedelta import relativedelta
import datetime as dt

def save_train_dataset():
    #서비스 데이터 읽기 : 전체 69,708 rows
    #고객 상품(이용권)별 재결제(해지) 이력 정보
    ds_service = "./Train/train_service.csv"
    df_service = pd.read_csv(ds_service, parse_dates=['registerdate','enddate'], infer_datetime_format=True)
    df_service.info()
    df_service_regist = df_service[['uno','registerdate']]

    #시청 이력(train_bookmark) 데이터 읽기 : 7,987,609 rows
    ds_bookmark = "./Train/train_bookmark.csv"
    df_bookmark = pd.read_csv(ds_bookmark, parse_dates=['dates'], infer_datetime_format=True)
    df_bookmark = pd.merge(df_bookmark, df_service_regist, on='uno', how='left')
    df_bookmark['dates2'] = df_bookmark['dates']-df_bookmark['registerdate']
    df_bookmark['dates2'] = df_bookmark['dates2'].dt.floor('D')
    df_bookmark_2 = df_bookmark


    # 유저별, 일일 총 시청시간의 '평균' '표준편차' 'max' 'min' 'skewness' 'cretfactor' 'kurtosis''median'
    df_f1 = df_bookmark_2.groupby(['uno','dates2']).viewtime.sum()  #일일 총 시청시간
    df_f1 = df_f1.to_frame().reset_index() # Series to Dataframe

    df_f1_std = df_f1.groupby(by='uno', as_index=False).viewtime.std()
    df_f1_max = df_f1.groupby(by='uno', as_index=False).viewtime.max()
    df_f1_min = df_f1.groupby(by='uno', as_index=False).viewtime.min()
    df_f1_sum = df_f1.groupby(by='uno', as_index=False).viewtime.sum()
    df_f1_mean = df_f1.groupby(by='uno', as_index=False).viewtime.mean()


    # Feature 추가 항목
    # 5일이상
    df_f1_d5 = df_f1[df_f1.dates2 > dt.timedelta(days=5)]
    df_f1_d5_sum = df_f1_d5.groupby(by='uno', as_index=False).viewtime.sum()
    df_f1_d5_sum = pd.merge(df_f1_d5_sum, df_f1_sum, on='uno', how='left')
    df_f1_d5_sum['CUS_V5'] = df_f1_d5_sum['viewtime_x']/df_f1_d5_sum['viewtime_y']

    # 10일이상
    df_f1_d10 = df_f1[df_f1.dates2 > dt.timedelta(days=10)]
    df_f1_d10_sum = df_f1_d10.groupby(by='uno', as_index=False).viewtime.sum()
    df_f1_d10_sum = pd.merge(df_f1_d10_sum, df_f1_sum, on='uno', how='left')
    df_f1_d10_sum['CUS_V10'] = df_f1_d10_sum['viewtime_x']/df_f1_d10_sum['viewtime_y']

    # 15일 이상
    df_f1_d15 = df_f1[df_f1.dates2 > dt.timedelta(days=15)]
    df_f1_d15_sum = df_f1_d15.groupby(by='uno', as_index=False).viewtime.sum()
    df_f1_d15_sum = pd.merge(df_f1_d15_sum, df_f1_sum, on='uno', how='left')
    df_f1_d15_sum['CUS_V15'] = df_f1_d15_sum['viewtime_x']/df_f1_d15_sum['viewtime_y']


    # 새로운 피쳐 또 만들기
    df_bookmark_3 = df_bookmark

    # 새로운 피쳐 또 만들기
    # (1) 평균시간보정
    vh_avg = df_bookmark_3[['uno','hour']]
    vh_avg.loc[(vh_avg['hour'] <  5), 'hour'] = vh_avg['hour'] + 24
    vh_avg.hour.value_counts()


    #  고객별 시청 평균 시간대역
    nvh_avg = vh_avg.groupby(by='uno', as_index=False).hour.mean()
    nvh_avg.rename(columns={'hour':'NVH_AVG'}, inplace=True)

    view = df_bookmark_3[['uno','viewtime']]
    view_100 = view[view.viewtime < 100]
    v100 = view_100.groupby(['uno']).viewtime.count()
    v100=v100.to_frame().reset_index()
    v100.rename(columns={'viewtime':'v100'}, inplace=True)


    view_300 = view[view.viewtime < 300]
    v300 = view_300.groupby(['uno']).viewtime.count()
    v300=v300.to_frame().reset_index()
    v300.rename(columns={'viewtime':'v300'}, inplace=True)

    view_n = view[view.viewtime > 900]
    vn = view_n.groupby(['uno']).viewtime.count()
    vn=vn.to_frame().reset_index()
    vn.rename(columns={'viewtime':'vn'}, inplace=True)


    #COIN 이벤트 참가 이력 데이터 읽기
    ds_coin = "./Train/coin.csv"
    df_coin = pd.read_csv(ds_coin, parse_dates=['registerdate'], infer_datetime_format=True)

    # gender 값 및 분포 확인
    df_service.gender.value_counts()

    # gender null 값을 N 으로 변경 후 확인
    df_service['gender'] = df_service['gender'].fillna('N')
    df_service.gender.value_counts()

    # agegroup 값 및 분포 확인
    df_service.agegroup.value_counts()

    # agegroup 950 값을 0 으로 변환
    df_service['agegroup'] = df_service['agegroup'].replace(950, 0)
    df_service.agegroup.value_counts()

    # pgamount 값 및 분포 확인
    df_service.pgamount.value_counts()

    # pgamount 금액 중에 달러로 결제된 것 원화로 변경 (pgamount 100원 미만인 건은 Appstore에서 달러 결제 건임)
    df_service.loc[(df_service['pgamount'] <  100), 'pgamount'] = df_service['pgamount'] * 1120
    df_service.pgamount.value_counts()

    # 기타 컬럼들의 결측치 처리
    df_service = df_service.fillna('X')

    # 해지 예측 대상 서비스 추출
    # 즉, 예측시점(가입일+3주)에 해지하지 않은 서비스 추출
    df_svc_target = df_service[df_service.enddate.dt.date > (df_service.registerdate + pd.DateOffset(weeks=3)).dt.date]

    # (1) 고객별 서비스 가입 이력 수 #
    df_feature_1 = df_service.groupby(by='uno', as_index=False).registerdate.count()
    df_feature_1.rename(columns={'registerdate':'REG_CNT'}, inplace=True)

    # (2) 고객별 서비스 가입 이력 상품 수 #
    df_feature_2 = df_service[['uno','productcode']]
    df_feature_2 = df_feature_2.drop_duplicates() # 고객별 동일 상품 제거
    df_feature_2 = df_feature_2.groupby(by='uno', as_index=False).productcode.count()
    df_feature_2.rename(columns={'productcode':'PRD_CNT'}, inplace=True)

    # (3) 고객별 시청 건수 (1시간 단위) #
    df_feature_3 = df_bookmark.groupby(by='uno', as_index=False).dates.count()
    df_feature_3.rename(columns={'dates':'BM_CNT'}, inplace=True)

    # (4) 고객별 시청 총 시간
    df_feature_4 = df_bookmark.groupby(by='uno', as_index=False).viewtime.sum()
    df_feature_4.rename(columns={'viewtime':'VT_TOT'}, inplace=True)

    # (5) 고객별 시청 평균 시간
    df_feature_5 = df_bookmark.groupby(by='uno', as_index=False).viewtime.mean()
    df_feature_5.rename(columns={'viewtime':'VT_AVG'}, inplace=True)

    # (6) 고객별 시청 채널 수
    df_feature_6 = df_bookmark[['uno','channeltype']]
    df_feature_6 = df_feature_6.drop_duplicates() # 고객별 동일 채널 제거
    df_feature_6 = df_feature_6.groupby(by='uno', as_index=False).channeltype.count()
    df_feature_6.rename(columns={'channeltype':'CH_CNT'}, inplace=True)

    # (7) 고객별 시청 프로그램 수
    df_feature_7 = df_bookmark[['uno','programid']]
    df_feature_7 = df_feature_7.drop_duplicates() # 고객별 동일 프로그램 제거
    df_feature_7 = df_feature_7.groupby(by='uno', as_index=False).programid.count()
    df_feature_7.rename(columns={'programid':'PRG_CNT'}, inplace=True)

    # (8) 고객별 시청 디바이스 수
    df_feature_8 = df_bookmark[['uno','devicetype']]
    df_feature_8 = df_feature_8.drop_duplicates() # 고객별 동일 프로그램 제거
    df_feature_8 = df_feature_8.groupby(by='uno', as_index=False).devicetype.count()
    df_feature_8.rename(columns={'devicetype':'DEV_CNT'}, inplace=True)

    # (9) 고객별 시청 일자의 수
    df_feature_9 = df_bookmark[['uno','dates']]
    df_feature_9 = df_feature_9.drop_duplicates() # 고객별 동일 시청 일자 제거
    df_feature_9 = df_feature_9.groupby(by='uno', as_index=False).dates.count()
    df_feature_9.rename(columns={'dates':'DATE_CNT'}, inplace=True)

    # (10) 고객별 시청 평균 시간대역
    df_feature_10 = df_bookmark.groupby(by='uno', as_index=False).hour.mean()
    df_feature_10.rename(columns={'hour':'VHOUR_AVG'}, inplace=True)

    # (11) 고객별 코인 관련 거래 횟수
    df_feature_11 = df_coin.groupby(by='uno', as_index=False).registerdate.count()
    df_feature_11.rename(columns={'registerdate':'COIN_CNT'}, inplace=True)

    # (12) 고객별 총 실제 결제 금액
    df_feature_12 = df_coin.groupby(by='uno', as_index=False).pgamount.sum()
    df_feature_12.rename(columns={'pgamount':'PG_TOT'}, inplace=True)

    # (13) 고객별 총 유료코인 결제 금액
    df_feature_13 = df_coin.groupby(by='uno', as_index=False).coinamount.sum()
    df_feature_13.rename(columns={'coinamount':'COIN_TOT'}, inplace=True)

    # (14) 고객별 총 무료코인 결제 금액
    df_feature_14 = df_coin.groupby(by='uno', as_index=False).bonusamount.sum()
    df_feature_14.rename(columns={'bonusamount':'BONUS_TOT'}, inplace=True)

    # (16) 고객별 일일 총 시청시간의 '표준편차'
    df_feature_16 = df_f1_std
    df_feature_16.rename(columns={'viewtime':'V_STD'}, inplace=True)

    # (17) 고객별 일일 총 시청시간의 'MAX'
    df_feature_17 = df_f1_max
    df_feature_17.rename(columns={'viewtime':'V_MAX'}, inplace=True)

    # (18) 고객별 일일 총 시청시간의 'min'
    df_feature_18 = df_f1_min
    df_feature_18.rename(columns={'viewtime':'V_MIN'}, inplace=True)

    # (19) 고객별 일일 총 시청시간의 'skewness'
    df_feature_19 = df_f1_sum
    df_feature_19.rename(columns={'viewtime':'V_SUM'}, inplace=True)

    # (20) 고객별 일일 총 시청시간의 'kurtosis'
    df_feature_20 = df_f1_mean
    df_feature_20.rename(columns={'viewtime':'V_MEAN'}, inplace=True)

    # (21) 고객별 5일 후 이용 시간 / 총이용시간
    df_feature_21 = df_f1_d5_sum[['uno','CUS_V5']]

    # (22) 고객별 5일 후 이용 시간 / 총이용시간
    df_feature_22 = df_f1_d10_sum[['uno','CUS_V10']]

    # (23) 고객별 5일 후 이용 시간 / 총이용시간
    df_feature_23 = df_f1_d15_sum[['uno','CUS_V15']]

    # (24) 평균시간보정
    df_feature_24 = nvh_avg[['uno','NVH_AVG']]

    # (25) View time 100 미만 갯수
    df_feature_25 = v100[['uno','v100']]

    # (26) View time 300 미만 갯수
    df_feature_26 = v300[['uno','v300']]

    #(27) View time 900 이상 갯수
    df_feature_27 = vn[['uno','vn']]


    # 해지 예측 대상 서비스에 생성한 변수 연결
    df_svc_target = pd.merge(df_svc_target, df_feature_1, on='uno', how='left') #REG_CNT
    df_svc_target = pd.merge(df_svc_target, df_feature_2, on='uno', how='left') #PRD_CNT
    df_svc_target = pd.merge(df_svc_target, df_feature_3, on='uno', how='left') #BM_CNT
    df_svc_target = pd.merge(df_svc_target, df_feature_4, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_5, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_6, on='uno', how='left') #CH_CNT
    df_svc_target = pd.merge(df_svc_target, df_feature_7, on='uno', how='left') #PRG_CNT
    df_svc_target = pd.merge(df_svc_target, df_feature_8, on='uno', how='left') #DEV_CNT
    df_svc_target = pd.merge(df_svc_target, df_feature_9, on='uno', how='left') #DATE_CNT
    df_svc_target = pd.merge(df_svc_target, df_feature_10, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_11, on='uno', how='left') #COIN_CNT
    df_svc_target = pd.merge(df_svc_target, df_feature_12, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_13, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_14, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_16, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_17, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_18, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_19, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_20, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_21, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_22, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_23, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_24, on='uno', how='left')
    df_svc_target = pd.merge(df_svc_target, df_feature_25, on='uno', how='left') #v100
    df_svc_target = pd.merge(df_svc_target, df_feature_26, on='uno', how='left') #v300
    df_svc_target = pd.merge(df_svc_target, df_feature_27, on='uno', how='left') #vn


    # 추가 생성 컬럼들의 결측치 처리
    df_svc_target = df_svc_target.fillna(0)
    df_svc_target = df_svc_target.astype({'BM_CNT':'int', 'VT_TOT':'int', 'CH_CNT':'int', 'PRG_CNT':'int', 'DEV_CNT':'int', 'DATE_CNT':'int','VHOUR_AVG':'float','COIN_CNT':'int','PG_TOT':'int','COIN_TOT':'int','BONUS_TOT':'int','V_STD':'float','V_MAX':'float','V_MIN':'float','V_SUM':'float','V_MEAN':'float','CUS_V5':'float','CUS_V10':'float','CUS_V15':'float','NVH_AVG':'float','v100':'int','v300':'int','vn':'int'})

    # 최종 분석 대상 데이터
    df_T = df_svc_target.copy()
    df_T['CHURN'] = np.where(df_T.Repurchase == 'X', 1, 0)

    # df_temp = pd.DataFrame()
    # df_temp = pd.DataFrame({'REG_CNT': df_T['REG_CNT'].unique(),'value': np.arange(len(df_T['REG_CNT'].unique()))})
    # df_temp.to_csv('REG_CNT.csv', index=False)
    # df_temp = pd.DataFrame({'PRD_CNT': df_T['PRD_CNT'].unique(),'value': np.arange(len(df_T['PRD_CNT'].unique()))})
    # df_temp.to_csv('PRD_CNT.csv', index=False)
    # df_temp = pd.DataFrame({'BM_CNT': df_T['BM_CNT'].unique(),'value': np.arange(len(df_T['BM_CNT'].unique()))})
    # df_temp.to_csv('BM_CNT.csv', index=False)
    # df_temp = pd.DataFrame({'CH_CNT': df_T['CH_CNT'].unique(),'value': np.arange(len(df_T['CH_CNT'].unique()))})
    # df_temp.to_csv('CH_CNT.csv', index=False)
    # df_temp = pd.DataFrame({'PRG_CNT': df_T['PRG_CNT'].unique(),'value': np.arange(len(df_T['PRG_CNT'].unique()))})
    # df_temp.to_csv('PRG_CNT.csv', index=False)
    # df_temp = pd.DataFrame({'DEV_CNT': df_T['DEV_CNT'].unique(),'value': np.arange(len(df_T['DEV_CNT'].unique()))})
    # df_temp.to_csv('DEV_CNT.csv', index=False)
    # df_temp = pd.DataFrame({'DATE_CNT': df_T['DATE_CNT'].unique(),'value': np.arange(len(df_T['DATE_CNT'].unique()))})
    # df_temp.to_csv('DATE_CNT.csv', index=False)
    # df_temp = pd.DataFrame({'COIN_CNT': df_T['COIN_CNT'].unique(),'value': np.arange(len(df_T['COIN_CNT'].unique()))})
    # df_temp.to_csv('COIN_CNT.csv', index=False)
    # df_temp = pd.DataFrame({'v100': df_T['v100'].unique(),'value': np.arange(len(df_T['v100'].unique()))})
    # df_temp.to_csv('v100.csv', index=False)
    # df_temp = pd.DataFrame({'v300': df_T['v300'].unique(),'value': np.arange(len(df_T['v300'].unique()))})
    # df_temp.to_csv('v300.csv', index=False)
    # df_temp = pd.DataFrame({'vn': df_T['vn'].unique(),'value': np.arange(len(df_T['vn'].unique()))})
    # df_temp.to_csv('vn.csv', index=False)

    df_REG_CNT = pd.read_csv("REG_CNT.csv")
    df_REG_CNT = df_REG_CNT[:5]
    REG_CNT = dict(zip(df_REG_CNT.REG_CNT,df_REG_CNT.value))
    df_T["REG_CNT"]= df_T['REG_CNT'].map(REG_CNT)

    df_PRD_CNT = pd.read_csv("PRD_CNT.csv")
    df_PRD_CNT = df_PRD_CNT[:4]
    PRD_CNT = dict(zip(df_PRD_CNT.PRD_CNT,df_PRD_CNT.value))
    df_T["PRD_CNT"]= df_T['PRD_CNT'].map(PRD_CNT)

    df_BM_CNT = pd.read_csv("BM_CNT.csv")
    df_BM_CNT = df_BM_CNT[:1098]
    BM_CNT = dict(zip(df_BM_CNT.BM_CNT,df_BM_CNT.value))
    df_T["BM_CNT"]= df_T['BM_CNT'].map(BM_CNT)

    df_CH_CNT = pd.read_csv("CH_CNT.csv")
    df_CH_CNT = df_CH_CNT[:5]
    CH_CNT = dict(zip(df_CH_CNT.CH_CNT,df_CH_CNT.value))
    df_T["CH_CNT"]= df_T['CH_CNT'].map(CH_CNT)

    df_PRG_CNT = pd.read_csv("PRG_CNT.csv")
    df_PRG_CNT = df_PRG_CNT[:138]
    PRG_CNT = dict(zip(df_PRG_CNT.PRG_CNT,df_PRG_CNT.value))
    df_T["PRG_CNT"]= df_T['PRG_CNT'].map(PRG_CNT)

    df_DEV_CNT = pd.read_csv("DEV_CNT.csv")
    df_DEV_CNT = df_DEV_CNT[:12]
    DEV_CNT = dict(zip(df_DEV_CNT.DEV_CNT,df_DEV_CNT.value))
    df_T["DEV_CNT"]= df_T['DEV_CNT'].map(DEV_CNT)

    df_DATE_CNT = pd.read_csv("DATE_CNT.csv")
    df_DATE_CNT = df_DATE_CNT[:32]
    DATE_CNT = dict(zip(df_DATE_CNT.DATE_CNT,df_DATE_CNT.value))
    df_T["DATE_CNT"]= df_T['DATE_CNT'].map(DATE_CNT)

    df_COIN_CNT = pd.read_csv("COIN_CNT.csv")
    df_COIN_CNT = df_COIN_CNT[:16]
    COIN_CNT = dict(zip(df_COIN_CNT.COIN_CNT,df_COIN_CNT.value))
    df_T["COIN_CNT"]= df_T['COIN_CNT'].map(COIN_CNT)

    df_v100 = pd.read_csv("v100.csv")
    df_v100 = df_v100[:434]
    v100 = dict(zip(df_v100.v100,df_v100.value))
    df_T["v100"]= df_T['v100'].map(v100)

    df_v300 = pd.read_csv("v300.csv")
    df_v300 = df_v300[:599]
    v300 = dict(zip(df_v300.v300,df_v300.value))
    df_T["v300"]= df_T['v300'].map(v300)

    df_vn = pd.read_csv("vn.csv")
    df_vn = df_vn[:596]
    vn = dict(zip(df_vn.vn,df_vn.value))
    df_T["vn"]= df_T['vn'].map(vn)

    # 상품 productcode Indexing / coin productcode 넣어지면서 앞에서 product_x로 변경됨
    df_productCode = pd.read_csv("Productcode.csv")
    df_productCode = df_productCode[:279]
    productcode = dict(zip(df_productCode.Productcode,df_productCode.value))
    df_T["productcode"]= df_T['productcode'].map(productcode)

    # 상품 chargetypeid indexing
    df_chargetypeid = pd.read_csv("chargetypeid.csv")
    df_chargetypeid = df_chargetypeid[:33]
    chargetypeid = dict(zip(df_chargetypeid.chargetypeid,df_chargetypeid.value))
    df_T["chargetypeid"]= df_T['chargetypeid'].map(chargetypeid)

    #Age 정보 indexing
    df_agegroup = pd.read_csv("agegroup.csv")
    df_agegroup = df_agegroup[:27]
    agegroup = dict(zip(df_agegroup.agegroup,df_agegroup.value))
    df_T["agegroup"]= df_T['agegroup'].map(agegroup)

    df_T.loc[df_T.promo_100=="X","promo_100"]=0
    df_T.loc[df_T.promo_100=="O","promo_100"]=1

    df_T.loc[df_T.coinReceived=="X","coinReceived"]=0
    df_T.loc[df_T.coinReceived=="O","coinReceived"]=1

    df_T.loc[df_T.devicetypeid=="pc","devicetypeid"]=0
    df_T.loc[df_T.devicetypeid=="ios","devicetypeid"]=1
    df_T.loc[df_T.devicetypeid=="android","devicetypeid"]=2
    df_T.loc[df_T.devicetypeid=="smarttv","devicetypeid"]=3
    df_T.loc[df_T.devicetypeid=="ott","devicetypeid"]=4
    df_T.loc[df_T.devicetypeid=="mobile","devicetypeid"]=5
    df_T.loc[df_T.devicetypeid=="lgchplus","devicetypeid"]=6
    df_T.loc[df_T.devicetypeid=="ibb","devicetypeid"]=7
    df_T.loc[df_T.devicetypeid=="chromecast","devicetypeid"]=8
    df_T.loc[df_T.devicetypeid=="sstv","devicetypeid"]=9
    df_T.loc[df_T.devicetypeid=="ssfamilyhub","devicetypeid"]=10
    df_T.loc[df_T.devicetypeid=="lgtv","devicetypeid"]=11
    df_T.loc[df_T.devicetypeid=="tiviva","devicetypeid"]=12
    df_T.loc[df_T.devicetypeid=="ott_unocube","devicetypeid"]=13
    df_T.loc[df_T.devicetypeid=="unocube_live","devicetypeid"]=14
    df_T.loc[df_T.devicetypeid=="ott_skylife","devicetypeid"]=15
    df_T.loc[df_T.devicetypeid=="ott_dlive","devicetypeid"]=16
    df_T.loc[df_T.devicetypeid=="ott_hanliutv","devicetypeid"]=17
    df_T.loc[df_T.devicetypeid=="ott_titan","devicetypeid"]=18
    df_T.loc[df_T.devicetypeid=="ott_cjhello","devicetypeid"]=19
    df_T.loc[df_T.devicetypeid=="ott_telebee","devicetypeid"]=20
    df_T.loc[df_T.devicetypeid=="ott_kraizer","devicetypeid"]=21
    df_T.loc[df_T.devicetypeid=="ott_skya","devicetypeid"]=22
    df_T.loc[df_T.devicetypeid=="ott_unocube2","devicetypeid"]=23

    df_T.loc[df_T.isauth=="X","isauth"]=0
    df_T.loc[df_T.isauth=="Y","isauth"]=1

    df_T.loc[df_T.gender=="N","gender"]=0
    df_T.loc[df_T.gender=="F","gender"]=1
    df_T.loc[df_T.gender=="M","gender"]=2

    df_T.loc[df_T.concurrentwatchcount==1,"concurrentwatchcount"]=0
    df_T.loc[df_T.concurrentwatchcount==2,"concurrentwatchcount"]=1
    df_T.loc[df_T.concurrentwatchcount==3,"concurrentwatchcount"]=2
    df_T.loc[df_T.concurrentwatchcount==4,"concurrentwatchcount"]=3

    # df_T 에 들어어가고도 결측치 data 처리
    df_T = df_T[["uno","registerdate","enddate","Repurchase","CHURN",'productcode', 'chargetypeid', 'promo_100', 'coinReceived', 'devicetypeid', 'isauth', 'gender', 'agegroup', 'concurrentwatchcount','REG_CNT', 'PRD_CNT','BM_CNT','CH_CNT','PRG_CNT','DEV_CNT', 'DATE_CNT', 'COIN_CNT','v100','v300','vn','pgamount', 'VT_TOT','VT_AVG','VHOUR_AVG','PG_TOT','COIN_TOT','BONUS_TOT','V_STD','V_MAX','V_MIN','V_SUM','V_MEAN','CUS_V5','CUS_V10','CUS_V15','NVH_AVG']]
    
    df_T.to_csv("Training_dataset2_v3.csv", index=False, encoding='utf8')

    # Standardization
    from sklearn.preprocessing import RobustScaler
    transformer = RobustScaler()
    Standard_X = df_T.drop(["uno","registerdate","enddate","Repurchase","CHURN",'productcode', 'chargetypeid', 'promo_100', 'coinReceived', 'devicetypeid', 'isauth', 'gender', 'agegroup', 'concurrentwatchcount','REG_CNT', 'PRD_CNT','BM_CNT','CH_CNT','PRG_CNT','DEV_CNT', 'DATE_CNT', 'COIN_CNT','v100','v300','vn'], axis=1)
    transformer.fit(Standard_X)
    Standard_X = transformer.transform(Standard_X)

    df_T['pgamount'] = Standard_X[:,0]
    df_T['VT_TOT'] = Standard_X[:,1]
    df_T['VT_AVG'] = Standard_X[:,2]
    df_T['VHOUR_AVG'] = Standard_X[:,3]
    df_T['PG_TOT'] = Standard_X[:,4]
    df_T['COIN_TOT'] = Standard_X[:,5]
    df_T['BONUS_TOT'] = Standard_X[:,6]
    df_T['V_STD'] = Standard_X[:,7]
    df_T['V_MAX'] = Standard_X[:,8]
    df_T['V_MIN'] = Standard_X[:,9]
    df_T['V_SUM'] = Standard_X[:,10]
    df_T['V_MEAN'] = Standard_X[:,11]
    df_T['CUS_V5'] = Standard_X[:,12]
    df_T['CUS_V10'] = Standard_X[:,13]
    df_T['CUS_V15'] = Standard_X[:,14]
    df_T['NVH_AVG'] = Standard_X[:,15]
    df_T.to_csv("Training_dataset_v3.csv", index=False, encoding='utf8')

def save_predict_dataset():
    #서비스 데이터 읽기 :
    #고객 상품(이용권)별 재결제(해지) 이력 정보
    pds_service = "./Predict/predict_service.csv"
    pdf_service = pd.read_csv(pds_service, parse_dates=['registerdate','enddate'], infer_datetime_format=True)
    pdf_service = pdf_service.drop(['Repurchase'],axis=1)
    pdf_service_regist = pdf_service[['uno','registerdate']]

    #시청 이력(train_bookmark) 데이터 읽기 : 7,987,609 rows
    pds_bookmark = "./Predict/predict_bookmark.csv"
    pdf_bookmark = pd.read_csv(pds_bookmark, parse_dates=['dates'], infer_datetime_format=True)
    pdf_bookmark = pd.merge(pdf_bookmark, pdf_service_regist, on='uno', how='left')
    pdf_bookmark['dates2'] = pdf_bookmark['dates']-pdf_bookmark['registerdate']
    pdf_bookmark['dates2'] = pdf_bookmark['dates2'].dt.floor('D')
    pdf_bookmark_2 = pdf_bookmark


    # 유저별, 일일 총 시청시간의 '평균' '표준편차' 'max' 'min' 'skewness' 'cretfactor' 'kurtosis''median'
    pdf_f1 = pdf_bookmark_2.groupby(['uno','dates2']).viewtime.sum()  #일일 총 시청시간
    pdf_f1 = pdf_f1.to_frame().reset_index() # Series to Dataframe

    pdf_f1_std = pdf_f1.groupby(by='uno', as_index=False).viewtime.std()
    pdf_f1_max = pdf_f1.groupby(by='uno', as_index=False).viewtime.max()
    pdf_f1_min = pdf_f1.groupby(by='uno', as_index=False).viewtime.min()
    pdf_f1_sum = pdf_f1.groupby(by='uno', as_index=False).viewtime.sum()
    pdf_f1_mean = pdf_f1.groupby(by='uno', as_index=False).viewtime.mean()


    # Feature 추가 항목
    # 5일이상
    pdf_f1_d5 = pdf_f1[pdf_f1.dates2 > dt.timedelta(days=5)]
    pdf_f1_d5_sum = pdf_f1_d5.groupby(by='uno', as_index=False).viewtime.sum()
    pdf_f1_d5_sum = pd.merge(pdf_f1_d5_sum, pdf_f1_sum, on='uno', how='left')
    pdf_f1_d5_sum['CUS_V5'] = pdf_f1_d5_sum['viewtime_x']/pdf_f1_d5_sum['viewtime_y']

    # 10일이상
    pdf_f1_d10 = pdf_f1[pdf_f1.dates2 > dt.timedelta(days=10)]
    pdf_f1_d10_sum = pdf_f1_d10.groupby(by='uno', as_index=False).viewtime.sum()
    pdf_f1_d10_sum = pd.merge(pdf_f1_d10_sum, pdf_f1_sum, on='uno', how='left')
    pdf_f1_d10_sum['CUS_V10'] = pdf_f1_d10_sum['viewtime_x']/pdf_f1_d10_sum['viewtime_y']

    # 15일 이상
    pdf_f1_d15 = pdf_f1[pdf_f1.dates2 > dt.timedelta(days=15)]
    pdf_f1_d15_sum = pdf_f1_d15.groupby(by='uno', as_index=False).viewtime.sum()
    pdf_f1_d15_sum = pd.merge(pdf_f1_d15_sum, pdf_f1_sum, on='uno', how='left')
    pdf_f1_d15_sum['CUS_V15'] = pdf_f1_d15_sum['viewtime_x']/pdf_f1_d15_sum['viewtime_y']


    # 새로운 피쳐 또 만들기
    pdf_bookmark_3 = pdf_bookmark

    # 새로운 피쳐 또 만들기
    # (1) 평균시간보정
    pvh_avg = pdf_bookmark_3[['uno','hour']]
    pvh_avg.loc[(pvh_avg['hour'] <  5), 'hour'] = pvh_avg['hour'] + 24
    pvh_avg.hour.value_counts()


    #  고객별 시청 평균 시간대역
    pnvh_avg = pvh_avg.groupby(by='uno', as_index=False).hour.mean()
    pnvh_avg.rename(columns={'hour':'NVH_AVG'}, inplace=True)

    pview = pdf_bookmark_3[['uno','viewtime']]
    pview_100 = pview[pview.viewtime < 100]
    pv100 = pview_100.groupby(['uno']).viewtime.count()
    pv100=pv100.to_frame().reset_index()
    pv100.rename(columns={'viewtime':'v100'}, inplace=True)

    ###
    pview_300 = pview[pview.viewtime < 300]
    pv300 = pview_300.groupby(['uno']).viewtime.count()
    pv300=pv300.to_frame().reset_index()
    pv300.rename(columns={'viewtime':'v300'}, inplace=True)

    pview_n = pview[pview.viewtime > 900]
    pvn = pview_n.groupby(['uno']).viewtime.count()
    pvn=pvn.to_frame().reset_index()
    pvn.rename(columns={'viewtime':'vn'}, inplace=True)


    #COIN 이벤트 참가 이력 데이터 읽기
    pds_coin = "./Train/coin.csv"
    pdf_coin = pd.read_csv(pds_coin, parse_dates=['registerdate'], infer_datetime_format=True)

    # gender 값 및 분포 확인
    pdf_service.gender.value_counts()

    # gender null 값을 N 으로 변경 후 확인
    pdf_service['gender'] = pdf_service['gender'].fillna('N')
    pdf_service.gender.value_counts()

    # agegroup 값 및 분포 확인
    pdf_service.agegroup.value_counts()

    # agegroup 950 값을 0 으로 변환
    pdf_service['agegroup'] = pdf_service['agegroup'].replace(950, 0)
    pdf_service.agegroup.value_counts()

    # pgamount 값 및 분포 확인
    pdf_service.pgamount.value_counts()

    # pgamount 금액 중에 달러로 결제된 것 원화로 변경 (pgamount 100원 미만인 건은 Appstore에서 달러 결제 건임)
    pdf_service.loc[(pdf_service['pgamount'] <  100), 'pgamount'] = pdf_service['pgamount'] * 1120
    pdf_service.pgamount.value_counts()

    # 기타 컬럼들의 결측치 처리
    pdf_service = pdf_service.fillna('X')

    # 해지 예측 대상 서비스 추출
    # 즉, 예측시점(가입일+3주)에 해지하지 않은 서비스 추출
    pdf_svc_target = pdf_service[pdf_service.enddate.dt.date > (pdf_service.registerdate + pd.DateOffset(weeks=3)).dt.date]

    # (1) 고객별 서비스 가입 이력 수
    pdf_feature_1 = pdf_service.groupby(by='uno', as_index=False).registerdate.count()
    pdf_feature_1.rename(columns={'registerdate':'REG_CNT'}, inplace=True)

    # (2) 고객별 서비스 가입 이력 상품 수
    pdf_feature_2 = pdf_service[['uno','productcode']]
    pdf_feature_2 = pdf_feature_2.drop_duplicates() # 고객별 동일 상품 제거
    pdf_feature_2 = pdf_feature_2.groupby(by='uno', as_index=False).productcode.count()
    pdf_feature_2.rename(columns={'productcode':'PRD_CNT'}, inplace=True)

    # (3) 고객별 시청 건수 (1시간 단위)
    pdf_feature_3 = pdf_bookmark.groupby(by='uno', as_index=False).dates.count()
    pdf_feature_3.rename(columns={'dates':'BM_CNT'}, inplace=True)

    # (4) 고객별 시청 총 시간
    pdf_feature_4 = pdf_bookmark.groupby(by='uno', as_index=False).viewtime.sum()
    pdf_feature_4.rename(columns={'viewtime':'VT_TOT'}, inplace=True)

    # (5) 고객별 시청 평균 시간
    pdf_feature_5 = pdf_bookmark.groupby(by='uno', as_index=False).viewtime.mean()
    pdf_feature_5.rename(columns={'viewtime':'VT_AVG'}, inplace=True)

    # (6) 고객별 시청 채널 수
    pdf_feature_6 = pdf_bookmark[['uno','channeltype']]
    pdf_feature_6 = pdf_feature_6.drop_duplicates() # 고객별 동일 채널 제거
    pdf_feature_6 = pdf_feature_6.groupby(by='uno', as_index=False).channeltype.count()
    pdf_feature_6.rename(columns={'channeltype':'CH_CNT'}, inplace=True)

    # (7) 고객별 시청 프로그램 수
    pdf_feature_7 = pdf_bookmark[['uno','programid']]
    pdf_feature_7 = pdf_feature_7.drop_duplicates() # 고객별 동일 프로그램 제거
    pdf_feature_7 = pdf_feature_7.groupby(by='uno', as_index=False).programid.count()
    pdf_feature_7.rename(columns={'programid':'PRG_CNT'}, inplace=True)

    # (8) 고객별 시청 디바이스 수
    pdf_feature_8 = pdf_bookmark[['uno','devicetype']]
    pdf_feature_8 = pdf_feature_8.drop_duplicates() # 고객별 동일 프로그램 제거
    pdf_feature_8 = pdf_feature_8.groupby(by='uno', as_index=False).devicetype.count()
    pdf_feature_8.rename(columns={'devicetype':'DEV_CNT'}, inplace=True)

    # (9) 고객별 시청 일자의 수
    pdf_feature_9 = pdf_bookmark[['uno','dates']]
    pdf_feature_9 = pdf_feature_9.drop_duplicates() # 고객별 동일 시청 일자 제거
    pdf_feature_9 = pdf_feature_9.groupby(by='uno', as_index=False).dates.count()
    pdf_feature_9.rename(columns={'dates':'DATE_CNT'}, inplace=True)

    # (10) 고객별 시청 평균 시간대역
    pdf_feature_10 = pdf_bookmark.groupby(by='uno', as_index=False).hour.mean()
    pdf_feature_10.rename(columns={'hour':'VHOUR_AVG'}, inplace=True)

    # (11) 고객별 코인 관련 거래 횟수
    pdf_feature_11 = pdf_coin.groupby(by='uno', as_index=False).registerdate.count()
    pdf_feature_11.rename(columns={'registerdate':'COIN_CNT'}, inplace=True)

    # (12) 고객별 총 실제 결제 금액
    pdf_feature_12 = pdf_coin.groupby(by='uno', as_index=False).pgamount.sum()
    pdf_feature_12.rename(columns={'pgamount':'PG_TOT'}, inplace=True)

    # (13) 고객별 총 유료코인 결제 금액
    pdf_feature_13 = pdf_coin.groupby(by='uno', as_index=False).coinamount.sum()
    pdf_feature_13.rename(columns={'coinamount':'COIN_TOT'}, inplace=True)

    # (14) 고객별 총 무료코인 결제 금액
    pdf_feature_14 = pdf_coin.groupby(by='uno', as_index=False).bonusamount.sum()
    pdf_feature_14.rename(columns={'bonusamount':'BONUS_TOT'}, inplace=True)

    # (16) 고객별 일일 총 시청시간의 '표준편차'
    pdf_feature_16 = pdf_f1_std
    pdf_feature_16.rename(columns={'viewtime':'V_STD'}, inplace=True)

    # (17) 고객별 일일 총 시청시간의 'MAX'
    pdf_feature_17 = pdf_f1_max
    pdf_feature_17.rename(columns={'viewtime':'V_MAX'}, inplace=True)

    # (18) 고객별 일일 총 시청시간의 'min'
    pdf_feature_18 = pdf_f1_min
    pdf_feature_18.rename(columns={'viewtime':'V_MIN'}, inplace=True)

    # (19) 고객별 일일 총 시청시간의 'skewness'
    pdf_feature_19 = pdf_f1_sum
    pdf_feature_19.rename(columns={'viewtime':'V_SUM'}, inplace=True)

    # (20) 고객별 일일 총 시청시간의 'kurtosis'
    pdf_feature_20 = pdf_f1_mean
    pdf_feature_20.rename(columns={'viewtime':'V_MEAN'}, inplace=True)

    # (21) 고객별 5일 후 이용 시간 / 총이용시간
    pdf_feature_21 = pdf_f1_d5_sum[['uno','CUS_V5']]

    # (22) 고객별 5일 후 이용 시간 / 총이용시간
    pdf_feature_22 = pdf_f1_d10_sum[['uno','CUS_V10']]

    # (23) 고객별 5일 후 이용 시간 / 총이용시간
    pdf_feature_23 = pdf_f1_d15_sum[['uno','CUS_V15']]

    # (24) 평균시간보정
    pdf_feature_24 = pnvh_avg[['uno','NVH_AVG']]

    # (25) View time 100 미만 갯수
    pdf_feature_25 = pv100[['uno','v100']]

    # (26) View time 300 미만 갯수
    pdf_feature_26 = pv300[['uno','v300']]

    #(27) View time 900 이상 갯수
    pdf_feature_27 = pvn[['uno','vn']]


    # 해지 예측 대상 서비스에 생성한 변수 연결
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_1, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_2, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_3, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_4, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_5, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_6, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_7, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_8, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_9, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_10, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_11, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_12, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_13, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_14, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_16, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_17, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_18, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_19, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_20, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_21, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_22, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_23, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_24, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_25, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_26, on='uno', how='left')
    pdf_svc_target = pd.merge(pdf_svc_target, pdf_feature_27, on='uno', how='left')


    # 추가 생성 컬럼들의 결측치 처리
    pdf_svc_target = pdf_svc_target.fillna(0)
    pdf_svc_target = pdf_svc_target.astype({'BM_CNT':'int', 'VT_TOT':'int', 'CH_CNT':'int', 'PRG_CNT':'int', 'DEV_CNT':'int', 'DATE_CNT':'int','VHOUR_AVG':'float','COIN_CNT':'int','PG_TOT':'int','COIN_TOT':'int','BONUS_TOT':'int','V_STD':'float','V_MAX':'float','V_MIN':'float','V_SUM':'float','V_MEAN':'float','CUS_V5':'float','CUS_V10':'float','CUS_V15':'float','NVH_AVG':'float','v100':'int','v300':'int','vn':'int'})

    # 최종 분석 대상 데이터
    pdf_T = pdf_svc_target.copy()

    # 상품 productcode Indexing / coin productcode 넣어지면서 앞에서 product_x로 변경됨
    pdf_productCode = pd.read_csv("Productcode.csv")
    pdf_productCode = pdf_productCode[:279]
    print(pdf_productCode)
    productcode = dict(zip(pdf_productCode.Productcode,pdf_productCode.index))
    pdf_T["productcode"]= pdf_T['productcode'].map(productcode)

    df_REG_CNT = pd.read_csv("REG_CNT.csv")
    df_REG_CNT = df_REG_CNT[:5]
    REG_CNT = dict(zip(df_REG_CNT.REG_CNT,df_REG_CNT.value))
    pdf_T["REG_CNT"]= pdf_T['REG_CNT'].map(REG_CNT)

    df_PRD_CNT = pd.read_csv("PRD_CNT.csv")
    df_PRD_CNT = df_PRD_CNT[:4]
    PRD_CNT = dict(zip(df_PRD_CNT.PRD_CNT,df_PRD_CNT.value))
    pdf_T["PRD_CNT"]= pdf_T['PRD_CNT'].map(PRD_CNT)

    df_BM_CNT = pd.read_csv("BM_CNT.csv")
    df_BM_CNT = df_BM_CNT[:1098]
    BM_CNT = dict(zip(df_BM_CNT.BM_CNT,df_BM_CNT.value))
    pdf_T["BM_CNT"]= pdf_T['BM_CNT'].map(BM_CNT)

    df_CH_CNT = pd.read_csv("CH_CNT.csv")
    df_CH_CNT = df_CH_CNT[:5]
    CH_CNT = dict(zip(df_CH_CNT.CH_CNT,df_CH_CNT.value))
    pdf_T["CH_CNT"]= pdf_T['CH_CNT'].map(CH_CNT)

    df_PRG_CNT = pd.read_csv("PRG_CNT.csv")
    df_PRG_CNT = df_PRG_CNT[:138]
    PRG_CNT = dict(zip(df_PRG_CNT.PRG_CNT,df_PRG_CNT.value))
    pdf_T["PRG_CNT"]= pdf_T['PRG_CNT'].map(PRG_CNT)

    df_DEV_CNT = pd.read_csv("DEV_CNT.csv")
    df_DEV_CNT = df_DEV_CNT[:12]
    DEV_CNT = dict(zip(df_DEV_CNT.DEV_CNT,df_DEV_CNT.value))
    pdf_T["DEV_CNT"]= pdf_T['DEV_CNT'].map(DEV_CNT)

    df_DATE_CNT = pd.read_csv("DATE_CNT.csv")
    df_DATE_CNT = df_DATE_CNT[:32]
    DATE_CNT = dict(zip(df_DATE_CNT.DATE_CNT,df_DATE_CNT.value))
    pdf_T["DATE_CNT"]= pdf_T['DATE_CNT'].map(DATE_CNT)

    df_COIN_CNT = pd.read_csv("COIN_CNT.csv")
    df_COIN_CNT = df_COIN_CNT[:16]
    COIN_CNT = dict(zip(df_COIN_CNT.COIN_CNT,df_COIN_CNT.value))
    pdf_T["COIN_CNT"]= pdf_T['COIN_CNT'].map(COIN_CNT)

    df_v100 = pd.read_csv("v100.csv")
    df_v100 = df_v100[:434]
    v100 = dict(zip(df_v100.v100,df_v100.value))
    pdf_T["v100"]= pdf_T['v100'].map(v100)

    df_v300 = pd.read_csv("v300.csv")
    df_v300 = df_v300[:599]
    v300 = dict(zip(df_v300.v300,df_v300.value))
    pdf_T["v300"]= pdf_T['v300'].map(v300)

    df_vn = pd.read_csv("vn.csv")
    df_vn = df_vn[:596]
    vn = dict(zip(df_vn.vn,df_vn.value))
    pdf_T["vn"]= pdf_T['vn'].map(vn)

    # 상품 chargetypeid indexing
    pdf_chargetypeid = pd.read_csv("chargetypeid.csv")
    pdf_chargetypeid = pdf_chargetypeid[:33]
    chargetypeid = dict(zip(pdf_chargetypeid.chargetypeid,pdf_chargetypeid.index))
    pdf_T["chargetypeid"]= pdf_T['chargetypeid'].map(chargetypeid)

    #Age 정보 indexing
    pdf_agegroup = pd.read_csv("agegroup.csv")
    pdf_agegroup = pdf_agegroup[:27]
    agegroup = dict(zip(pdf_agegroup.agegroup,pdf_agegroup.index))
    pdf_T["agegroup"]= pdf_T['agegroup'].map(agegroup)

    pdf_T.loc[pdf_T.promo_100=="X","promo_100"]=0
    pdf_T.loc[pdf_T.promo_100=="O","promo_100"]=1

    pdf_T.loc[pdf_T.coinReceived=="X","coinReceived"]=0
    pdf_T.loc[pdf_T.coinReceived=="O","coinReceived"]=1

    pdf_T.loc[pdf_T.devicetypeid=="pc","devicetypeid"]=0
    pdf_T.loc[pdf_T.devicetypeid=="ios","devicetypeid"]=1
    pdf_T.loc[pdf_T.devicetypeid=="android","devicetypeid"]=2
    pdf_T.loc[pdf_T.devicetypeid=="smarttv","devicetypeid"]=3
    pdf_T.loc[pdf_T.devicetypeid=="ott","devicetypeid"]=4
    pdf_T.loc[pdf_T.devicetypeid=="mobile","devicetypeid"]=5
    pdf_T.loc[pdf_T.devicetypeid=="lgchplus","devicetypeid"]=6
    pdf_T.loc[pdf_T.devicetypeid=="ibb","devicetypeid"]=7
    pdf_T.loc[pdf_T.devicetypeid=="chromecast","devicetypeid"]=8
    pdf_T.loc[pdf_T.devicetypeid=="sstv","devicetypeid"]=9
    pdf_T.loc[pdf_T.devicetypeid=="ssfamilyhub","devicetypeid"]=10
    pdf_T.loc[pdf_T.devicetypeid=="lgtv","devicetypeid"]=11
    pdf_T.loc[pdf_T.devicetypeid=="tiviva","devicetypeid"]=12
    pdf_T.loc[pdf_T.devicetypeid=="ott_unocube","devicetypeid"]=13
    pdf_T.loc[pdf_T.devicetypeid=="unocube_live","devicetypeid"]=14
    pdf_T.loc[pdf_T.devicetypeid=="ott_skylife","devicetypeid"]=15
    pdf_T.loc[pdf_T.devicetypeid=="ott_dlive","devicetypeid"]=16
    pdf_T.loc[pdf_T.devicetypeid=="ott_hanliutv","devicetypeid"]=17
    pdf_T.loc[pdf_T.devicetypeid=="ott_titan","devicetypeid"]=18
    pdf_T.loc[pdf_T.devicetypeid=="ott_cjhello","devicetypeid"]=19
    pdf_T.loc[pdf_T.devicetypeid=="ott_telebee","devicetypeid"]=20
    pdf_T.loc[pdf_T.devicetypeid=="ott_kraizer","devicetypeid"]=21
    pdf_T.loc[pdf_T.devicetypeid=="ott_skya","devicetypeid"]=22
    pdf_T.loc[pdf_T.devicetypeid=="ott_unocube2","devicetypeid"]=23

    pdf_T.loc[pdf_T.isauth=="X","isauth"]=0
    pdf_T.loc[pdf_T.isauth=="Y","isauth"]=1

    pdf_T.loc[pdf_T.gender=="N","gender"]=0
    pdf_T.loc[pdf_T.gender=="F","gender"]=1
    pdf_T.loc[pdf_T.gender=="M","gender"]=2


    pdf_T.loc[pdf_T.concurrentwatchcount==1,"concurrentwatchcount"]=0
    pdf_T.loc[pdf_T.concurrentwatchcount==2,"concurrentwatchcount"]=1
    pdf_T.loc[pdf_T.concurrentwatchcount==3,"concurrentwatchcount"]=2
    pdf_T.loc[pdf_T.concurrentwatchcount==4,"concurrentwatchcount"]=3

    # df_T 에 들어어가고도 결측치 data 처리
    pdf_T = pdf_T[["uno","registerdate","enddate",'productcode', 'chargetypeid', 'promo_100', 'coinReceived', 'devicetypeid', 'isauth', 'gender', 'agegroup', 'concurrentwatchcount','REG_CNT', 'PRD_CNT','BM_CNT','CH_CNT','PRG_CNT','DEV_CNT', 'DATE_CNT', 'COIN_CNT','v100','v300','vn','pgamount', 'VT_TOT','VT_AVG','VHOUR_AVG','PG_TOT','COIN_TOT','BONUS_TOT','V_STD','V_MAX','V_MIN','V_SUM','V_MEAN','CUS_V5','CUS_V10','CUS_V15','NVH_AVG']]
    
    # Standardization

    from sklearn.preprocessing import RobustScaler
    transformer = RobustScaler()
    df_T = pd.read_csv("Training_dataset2_v3.csv")
    Standard_T = df_T.drop(["uno","registerdate","enddate","Repurchase","CHURN",'productcode', 'chargetypeid', 'promo_100', 'coinReceived', 'devicetypeid', 'isauth', 'gender', 'agegroup', 'concurrentwatchcount','REG_CNT', 'PRD_CNT','BM_CNT','CH_CNT','PRG_CNT','DEV_CNT', 'DATE_CNT', 'COIN_CNT','v100','v300','vn'], axis=1)
   
    Standard_P = pdf_T.drop(["uno","registerdate","enddate",'chargetypeid', 'productcode','promo_100', 'coinReceived', 'devicetypeid', 'isauth', 'gender', 'agegroup', 'concurrentwatchcount','REG_CNT', 'PRD_CNT','BM_CNT','CH_CNT','PRG_CNT','DEV_CNT', 'DATE_CNT', 'COIN_CNT','v100','v300','vn'], axis=1)
   
    transformer.fit(Standard_T)
    Standard_P = transformer.transform(Standard_P)

    pdf_T['pgamount'] = Standard_P[:,0]
    pdf_T['VT_TOT'] = Standard_P[:,1]
    pdf_T['VT_AVG'] = Standard_P[:,2]
    pdf_T['VHOUR_AVG'] = Standard_P[:,3]
    pdf_T['PG_TOT'] = Standard_P[:,4]
    pdf_T['COIN_TOT'] = Standard_P[:,5]
    pdf_T['BONUS_TOT'] = Standard_P[:,6]
    pdf_T['V_STD'] = Standard_P[:,7]
    pdf_T['V_MAX'] = Standard_P[:,8]
    pdf_T['V_MIN'] = Standard_P[:,9]
    pdf_T['V_SUM'] = Standard_P[:,10]
    pdf_T['V_MEAN'] = Standard_P[:,11]
    pdf_T['CUS_V5'] = Standard_P[:,12]
    pdf_T['CUS_V10'] = Standard_P[:,13]
    pdf_T['CUS_V15'] = Standard_P[:,14]
    pdf_T['NVH_AVG'] = Standard_P[:,15]

    pdf_T.to_csv("predict_dataset.csv", index=False, encoding='utf8')

if __name__ == "__main__":
    save_predict_dataset()