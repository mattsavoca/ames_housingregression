import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn as skl
import statsmodels.api as sm
import re
from scipy.stats import norm, skew
from scipy.special import boxcox1p

# %matplotlib inline
sns.set()
warnings.filterwarnings('ignore')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12,8)

def custom_pre_proc(df, bctransform = False):
    df.columns = [x.lower() for x in df.columns]
    try:
        df['log_price'] = np.log(df.saleprice)
    except:
        pass
    df['plannedunitdev'] = df.mssubclass.isin(['180', '160', '150', '120'])
    df['remodhome'] = df['yearremodadd'] > df['yearbuilt']
    df['remodgarage'] = df['garageyrblt'] > df['yearremodadd']
    df['totalbath'] = df.fullbath + df.halfbath
    df['bsmttotalbath'] = df.bsmtfullbath + df.bsmthalfbath
    df['splitmulti'] = df.mssubclass.isin(['85', '80', '180'])
    df['seassold'] = df['mosold'].replace([12,1,2],"Winter").replace([3,4,5], "Spring").replace([6,7,8], "Summer").replace([9,10,11], "Fall")
    df['residentzone'] = df.mszoning.isin(['FV', 'RL', 'RM', 'RH', 'RP'])
    df['reszonehidens'] = df.mszoning == 'RH'
    df['publicutil'] = df.utilities.apply(lambda x: 1 if x == 'AllPub' else 0)
    df['lotconfig'] = df.lotconfig.apply(lambda x: 'FR2P' if x in ['FR2', 'FR3'] else x)
    df['isartery'] = (df.condition1 == 'Artery') | (df.condition2 == 'Artery')
    df['isfeeder'] = (df.condition1 == 'Feedr') | (df.condition2 == 'Feedr')
    df['normcond'] = (df.condition1 == 'Norm') | (df.condition2 == 'Norm')
    df['rrnearby'] = (df.condition1.isin(['RRNn', 'RRNe'])) | (df.condition2.isin(['RRNn', 'RRNe']))
    df['rradj'] = (df.condition1.isin(['RRAn', 'RRAe'])) | (df.condition2.isin(['RRAn', 'RRAe']))
    df['positnear'] = (df.condition1 == 'PosN') | (df.condition2 == 'PosN')
    df['positadj'] = (df.condition1 == 'PosA') | (df.condition2 == 'PosA')
    df['lotshapereg'] = df.lotshape == 'Reg'
    df['roofstyle'] = df.roofstyle.apply(lambda x: 'Other' if x in ['Shed', 'Flat', 'Mansard', 'Gambrel'] else x)
    df['foundation'] = df.roofstyle.apply(lambda x: 'Other' if x in ['Wood', 'Stone' 'Slab'] else x)
    df['compshgroof'] = df.roofmatl == 'CompShg'
    finstories = {'1.5Fin': 1.5, '1.5Unf': 1, '1Story': 1, '2.5Fin': 2.5, '2.5Unf': 2, '2Story': 2, 'SFoyer': 1.5, 'SLvl': 2}
    df['finishedstories'] = df.housestyle.map(finstories)
    df['asbestos'] = (df.exterior1st == 'AsbShng') | (df.exterior2nd == 'AsbShng')
    df.totalbsmtsf.fillna(0, inplace = True)
    df.bsmtunfsf.fillna(0, inplace = True)
    df['1stflrsf'].fillna(0, inplace = True)
    df['2ndflrsf'].fillna(0, inplace = True)
    df['totalsf'] = df.totalbsmtsf + df['1stflrsf'] + df['2ndflrsf']
    df['enc_2ndfloor'] = df['2ndflrsf'] > 0
    df['finishsf'] = df.totalsf - df.bsmtunfsf
    df.lowqualfinsf.fillna(0, inplace = True)
    df['lqfinishpct'] = df.lowqualfinsf/df.finishsf
    df['bsmtlivqtr'] = (df.bsmtfintype1.isin(['ALQ','GLQ'])) | (df.bsmtfintype2.isin(['ALQ','GLQ']))
    df['bsmtlivqtrlq'] = (df.bsmtfintype1.isin(['BLQ', 'LwQ'])) | (df.bsmtfintype2.isin(['BLQ', 'LwQ']))
    df['bsmtunf'] = (df.bsmtfintype1.isin(['BLQ', 'LwQ'])) | (df.bsmtfintype2.isin(['BLQ', 'LwQ']))
    df['nobsmt'] = (pd.isna(df.bsmtfintype1)) & (pd.isna(df.bsmtfintype2))
    df['gasheat'] = df.heating.isin(['GasA', 'GasW'])
    df['centralair'] = df.centralair == 'Y'
    df['3ssnporch'].fillna(0, inplace = True)
    df.screenporch.fillna(0, inplace = True)
    df.openporchsf.fillna(0, inplace = True)
    df.wooddecksf.fillna(0, inplace = True)
    df.lotarea.fillna(0, inplace = True)
    df.garagearea.fillna(0, inplace = True)
    df.grlivarea.fillna(0, inplace = True)
    df['porchdecksf'] = df['3ssnporch'] + df.screenporch + df.openporchsf + df.wooddecksf
    df['yardsf'] = df['lotarea'] - df['grlivarea'] - df.porchdecksf - df.poolarea
    df['sbreaker'] = df.electrical == 'Sbreaker'
    df['lqfuse'] = df.electrical.isin(['FuseF', 'FuseP'])
    df['typfunc'] = df.functional == 'Typ'
    df['attachdgarage'] = df.garagetype.isin(['BuiltIn','Attchd', 'Basment', '2Types'])
    df['nogarage'] = pd.isna(df.garagetype)
    df['detachdgarage'] = df.garagetype.isin(['Detchd'])
    df['finishgarage'] = df.garagefinish == 'Fin'
    df['paveddrive'] = df.paveddrive == 'Y'
    df['saletype'] = df.saletype.apply(lambda x: 'Other' if x not in ['New', 'WD'] else x)
    df['salecondition'] = df.salecondition.apply(lambda x: 'Other' if x not in ['Partial', 'Normal'] else x)
    df.exterior1st = df.exterior1st.str.replace("Brk Cmn", "BrkComm")
    df.exterior2nd = df.exterior2nd.str.replace("Brk Cmn", "BrkComm")
    df.exterior1st = df.exterior1st.str.replace("CmentBd", "CemntBd")
    df.exterior2nd = df.exterior2nd.str.replace("CmentBd", "CemntBd")
    df.exterior1st = df.exterior1st.str.replace("Wd Shng", "WdShing")
    df.exterior2nd = df.exterior2nd.str.replace("Wd Shng", "WdShing")
    df.exterior1st.fillna('Unk', inplace = True)
    df.exterior2nd.fillna(df.exterior1st,  inplace = True)
    df['onext'] = df.exterior1st == df.exterior2nd
    df['ext'] = df.loc[:,'exterior1st'] + '_' + df.loc[:,'exterior2nd']
    df.ext = np.where(df.onext, df.exterior1st, df.ext)
    df['ext_uniqueness'] = df.groupby('ext')['id'].transform('count')/df.shape[0]
    df['unique_ext'] = np.where(df.ext_uniqueness < .05, True, False)
    df['overallcond'] = df.overallcond.apply(lambda x: 1 if x <= 5 else x - 4)
    df['enc_pool'] = df.poolarea.apply(lambda x: 1 if x > 0 else 0)
    df['enc_alley'] = df.alley.apply(lambda x: 0 if pd.isna(x) else 1)
    df['enc_fence'] = df.fence.apply(lambda x: 0 if pd.isna(x) else 1)
    df.masvnrarea.interpolate()
    df['lotfrontage'] = df.lotfrontage.interpolate()
    df.masvnrarea = df.masvnrarea.apply(lambda x: 1 if x > 0 else 0)
    df.garagecond.fillna('None', inplace = True)
    df.garagefinish.fillna('None', inplace = True)
    df.fireplacequ.fillna('None', inplace = True)
    df.garagetype.fillna('None', inplace = True)
    df.garagequal.fillna('None', inplace = True)
    df.bsmtexposure.fillna('None', inplace = True)
    df.bsmtcond.fillna('None', inplace = True)
    df.bsmtqual.fillna('None', inplace = True)
    df.kitchenqual.fillna('None', inplace = True)
    df.garageyrblt.fillna(df.yearremodadd, inplace= True)
    df['extrarooms'] = df.totrmsabvgrd - df.bedroomabvgr - df.kitchenabvgr
    df['finishbsmtsf'] = df.bsmtfinsf1 + df.bsmtfinsf2
    df['finishbsmt'] = df.finishbsmtsf > 0
    df['multfireplaces'] = df.fireplaces > 1
    df.drop(['poolqc', 'poolarea', 'miscfeature', 'alley', 'fence', \
    'mssubclass', 'condition1', 'condition2', 'masvnrtype',\
    '2ndflrsf', 'exterior1st','bsmtfintype1', 'bsmtfintype2', \
    'bsmtfinsf1', 'bsmtfinsf2', 'fireplaces', \
    'lotshape', 'electrical', 'exterior1st', 'exterior2nd', \
     'housestyle', 'mszoning', 'roofmatl', 'heating', 'functional',\
    'garagetype', 'garagefinish', '3ssnporch', 'screenporch', 'openporchsf', 'wooddecksf',\
    'fullbath', 'halfbath', 'bsmtfullbath', 'bsmthalfbath', 'utilities', 'ext'
    ], axis = 1, inplace = True)
    exterqual_ord = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
    bsmtexp_ord = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}
    qual_ord = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'None': 0}
    qual_ord_None1 = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'None': 1}
    df['exterqual'] = df.exterqual.map(exterqual_ord)
    df['extercond'] = df.extercond.map(exterqual_ord)
    df['bsmtqual'] = df.bsmtqual.map(qual_ord)
    df['bsmtcond'] = df.bsmtcond.map(qual_ord_None1)
    df['bsmtexposure'] = df.bsmtexposure.map(bsmtexp_ord)
    df['heatingqc'] = df.heatingqc.map(qual_ord)
    df['kitchenqual'] = df.kitchenqual.map(qual_ord)
    df['fireplacequ'] = df.fireplacequ.map(qual_ord_None1)
    df['garagequal'] = df.garagequal.map(qual_ord)
    df['garagecond'] = df.garagecond.map(qual_ord)
    df['overallcond'] = df['overallcond']
    df['yrsold'] = df['yrsold'].astype(str)
    df['mosold'] = df['mosold'].astype(str)
    if bctransform:
        num_data = df.dtypes[(df.dtypes == 'int64') | (df.dtypes == 'float64')].index
        skewed_data = df[num_data].apply(lambda x: skew(x))
        skewed = skewed_data[abs(skewed_data) > 1.0]
        for i in skewed.index:
            df[i] = boxcox1p(df[i], 0.20)
    else:
        pass
    df.finishbsmtsf.fillna(0, inplace = True)
    df.garagecars.fillna(0, inplace = True)
    df['yardsf'].fillna(0, inplace = True)
    df['bsmttotalbath'].fillna(0.0, inplace = True)
    df['bsmtunfsf'].fillna(0, inplace = True)