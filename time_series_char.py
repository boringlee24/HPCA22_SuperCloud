import pandas as pd
import time
import os
import sys
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pickle

home = str(Path.home())
start = time.time()
gpu_ts_main=pd.read_csv("/some_path/nvidia_smi.csv")
end = time.time()
print (f"Sampling freq : {gpu_ts_main.iloc[1]['timestamp']-gpu_ts_main.iloc[0]['timestamp']}")
# gpu_ts_main.shape #(477606994, 12)
gpu_ts_mini=gpu_ts_main[['Node', 'gpu_index','pcie_link_width_current','timestamp','id_job','utilization_memory_pct','utilization_gpu_pct','power_draw_W','memory_used_MiB']]
start = time.time()
gpu_ts_mini=gpu_ts_mini.set_index('id_job')
end = time.time()
print(end - start)
start = time.time()
hash_ts_length=Counter(gpu_ts_mini.index)
end = time.time()
print(end - start)
print (len(hash_ts_length))

# read user info
user_df=pd.read_csv("2021-03-03/anon_sched.csv")
user_df['run_time_from_user_df']=user_df['time_end']-user_df['time_start']


#user_df has been renamed to user_df_no_dup. be careful of its usage later
if (user_df.shape[0]!=len(user_df['id_job'].unique())): #if there are duplicates
    tmp_df=user_df[user_df['id_job'].duplicated(keep=False)][['id_job', 'derived_ec', 'state', 'derived_es']] #may need to rewrite for 2021-IAP. it doesnt have these columns
    idx_to_drop=tmp_df.index[tmp_df['state']!=3]
    user_df_no_dup=user_df.drop(idx_to_drop)
    assert (user_df_no_dup.shape[0]==len(user_df_no_dup['id_job'].unique())) #all unique now
else:
    user_df_no_dup=user_df

user_df_no_dup.shape

set_ts=set(gpu_ts_mini.index)

start = time.time()
jobs_to_drop=[job for job in hash_ts_length if hash_ts_length[job]<300] #drop small length jobs. sample freq seems 100ms
end = time.time()
print(end - start)
print (len(jobs_to_drop))

start = time.time()
gpu_ts=gpu_ts_mini.drop(pd.Index(jobs_to_drop))
end = time.time()
print(end - start)


hash_ts_length=Counter(gpu_ts.index)
len(hash_ts_length)
start = time.time()

gpu_ts=gpu_ts.dropna() #removes NaN timestamps if any

hash_ts_length=Counter(gpu_ts.index)
len(hash_ts_length)

gpu_ts_rst_idx=gpu_ts.reset_index()

def find(series1):
    ans=[]
    current_len=0 #current_len acts like prev_zero_flag flag
    for val in series1:
        if val<1:
            current_len+=1
        elif val>=1 and current_len!=0:
            ans.append(current_len)
            current_len=0
    if current_len!=0:
        ans.append(current_len)
    return ans

def find_active(series1):
    ans=[]
    current_len=0 #current_len acts like prev_zero_flag flag
    for val in series1:
        if val>=1:
            current_len+=1
        elif val==0 and current_len!=0:
            ans.append(current_len)
            current_len=0
    if current_len!=0:
        ans.append(current_len)
    return ans

def find_all(series1):
    ans=[]
    current_len_idle=0 #current_len acts like prev_zero_flag flag
    current_len_active=0
    for val in series1:
        if val>=1:
            current_len_active+=1
            if current_len_idle!=0:
                ans.append(current_len_idle)
                current_len_idle=0
        if val==0:
            current_len_idle+=1
            if current_len_active!=0:
                ans.append(current_len_active)
                current_len_active=0
    if current_len_idle!=0:
        ans.append(current_len_idle)
    if current_len_active!=0:
        ans.append(current_len_active)
    return ans

dict_jobs_to_rle_sm=dict()
dict_jobs_to_rle_sm_active=dict()
dict_jobs_to_rle_sm_all=dict()

jobs_set=list(set(gpu_ts.index))

start=time.time()
for job in jobs_set:
    series1=gpu_ts.loc[job]['utilization_gpu_pct']
    dict_jobs_to_rle_sm_active[job]=find_active(series1)
end=time.time()


start=time.time()
for job in jobs_set:
    series1=gpu_ts.loc[job]['utilization_gpu_pct']
    dict_jobs_to_rle_sm_all[job]=find_all(series1)
end=time.time()

start=time.time()
for job in jobs_set:
    series1=gpu_ts.loc[job]['utilization_gpu_pct']
    dict_jobs_to_rle_sm[job]=find(series1)
end=time.time()
end-start

_object = dict_jobs_to_rle_sm_all
_file = open('dict_jobs_to_rle_sm_all.obj', 'wb')
pickle.dump(_object, _file) #https://www.thoughtco.com/using-pickle-to-save-objects-2813661
_file.close()

_object = dict_jobs_to_rle_sm_active
_file = open('dict_active.obj', 'wb')
pickle.dump(_object, _file) #https://www.thoughtco.com/using-pickle-to-save-objects-2813661
_file.close()


_object = dict_jobs_to_rle_sm
_file = open('dict.obj', 'wb')
pickle.dump(_object, _file) #https://www.thoughtco.com/using-pickle-to-save-objects-2813661
_file.close()


_file = open('dict_active.obj', 'rb')
dict_jobs_to_rle_sm_active=pickle.load(_file) #https://www.thoughtco.com/using-pickle-to-save-objects-2813661
_file.close()


_file = open('dict.obj', 'rb')
dict_jobs_to_rle_sm=pickle.load(_file) #https://www.thoughtco.com/using-pickle-to-save-objects-2813661
_file.close()

df=pd.DataFrame()
df['Idle_steps']=pd.Series(dtype=object)
df['Active_steps_list']=pd.Series(dtype=object)
df['All_steps_list']=pd.Series(dtype=object)
df['job_id']=(dict_jobs_to_rle_sm.keys())
df=df.set_index('job_id')
for job in df.index:
    df.loc[job,'Idle_steps']=dict_jobs_to_rle_sm[job]
    df.loc[job,'Active_steps_list']=dict_jobs_to_rle_sm_active[job]
    df.loc[job,'All_steps_list']=dict_jobs_to_rle_sm_all[job]

df_ck1=df


df_ck1['t_list']=None

# for job in df_ck1.index:
#     t_list=[]
#     for i,j in zip(df_ck1.loc[job]['Idle_steps'],df_ck1.loc[job]['Active_steps_list']):
#         t_list=t_list+[i,j]
#     df_ck1.loc[job]['t_list']=t_list



df_ck1['total_idle_steps']=df_ck1['Idle_steps'].apply(lambda x: sum(x))
df_ck1['total_active_steps_sanity']=df_ck1['Active_steps_list'].apply(lambda x: sum(x))
for job in df_ck1.index:
    df_ck1.loc[job,'total_steps']=hash_ts_length[job]
df_ck1['Idle_steps_rm1']=df_ck1['Idle_steps'].apply(lambda x: x[1:])
df_ck1['count_intervals_rm1']=df_ck1['Idle_steps_rm1'].apply(lambda x: len(x))
df_ck1['idle_time_pct']=df_ck1['total_idle_steps']/df_ck1['total_steps']*100

df_ck1['Active_Steps']=df_ck1['total_steps']-df_ck1['total_idle_steps']
df_ck1['Active_Steps_pct']=df_ck1['Active_Steps']/df_ck1['total_steps']*100

df_ck1=df_ck1.sort_values('Active_Steps_pct')

df1=df_ck1[['total_steps','Active_Steps_pct']]

df1=df1.sort_values('Active_Steps_pct')

plt.rcParams["figure.figsize"] = (3.1,2.6)
fig, axs = plt.subplots()
x=(np.arange(len(df1))/(len(df1)-1))*100
axs.plot(df1['Active_Steps_pct'],x,label="% Active Time",linestyle='--',linewidth=2,color='blue')
axs.set_xlabel('% Active Time',fontsize=18)
axs.set_ylabel('Empirical CDF \n(% of Jobs)',fontsize=18)
# axs.spines["left"].set_color("white")
# axs.spines["right"].set_color("orange")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# axs.legend(fontsize=15,labelspacing=0.25,handlelength=1,borderaxespad=0.2,handletextpad=0.4,borderpad=0.2)
# plt.stem([60],[26])
# plt.hlines([26], 0, [60])
plt.ylim(0,100)
plt.xlim(0,100)

# plt.text(62,20,"60,22",fontsize=14,weight='bold');
# axs2=axs.twinx()
# axs2.bar(x,df1['total_steps'],alpha=0.05,color='orange')
# axs2.set_yscale('log')
# axs2.set_ylabel('Total Runtime of Job',fontsize=15);
# # axs2.set_xlabel('Percentile of Jobs',fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15);
# axs2.spines["left"].set_color("red")
# axs2.spines["right"].set_color("orange");


# colors = {'Total Runtime':'orange'}
# labels = list(colors.keys())
# handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
# axs2.legend(handles, labels,bbox_to_anchor=(0.0078,0.79),fontsize=15,labelspacing=0.25,handlelength=1,borderaxespad=0.2,handletextpad=0.4,borderpad=0.2);
# plt.grid(color='lightgrey', linestyle=':')
axs.set_xticks([0,25,50,75,100])
axs.set_yticks([0,25,50,75,100])
plt.grid(color='darkgrey', linestyle=':')
plt.savefig('./plots/time_series/'+'plot1'+'.pdf',bbox_inches='tight');


active_df=gpu_ts_rst_idx[gpu_ts_rst_idx['utilization_gpu_pct']>=1]
active_df_mean=active_df.groupby('id_job').mean()
active_df_std=active_df.groupby('id_job').std()
cov_df=active_df_std/active_df_mean*100
cov_df.columns=[col+"_cov" for col in cov_df.columns]
cov_mean_df=cov_df.join(active_df_mean)

thrshold=5
cov_mean_df[['utilization_gpu_pct','utilization_gpu_pct_cov']].dropna().sort_values('utilization_gpu_pct_cov')
cov_df=cov_mean_df[cov_mean_df['utilization_gpu_pct']>cov_mean_df['utilization_gpu_pct'].mean()//thrshold]
df1=cov_df['utilization_gpu_pct_cov']
cov_mean_df[['utilization_gpu_pct','utilization_gpu_pct_cov']].dropna().sort_values('utilization_gpu_pct_cov')
cov_df=cov_mean_df[cov_mean_df['utilization_gpu_pct']>cov_mean_df['utilization_gpu_pct'].mean()//thrshold]
df1=cov_df['utilization_gpu_pct_cov']
df1=df1.sort_values()
x=(np.arange(len(df1))/(len(df1)-1))*100
plt.plot(df1.values,x,linestyle='-',linewidth=2,label="SM Util.",color='red')


cov_mean_df[['utilization_memory_pct','utilization_memory_pct_cov']].dropna().sort_values('utilization_memory_pct_cov')
cov_df=cov_mean_df[cov_mean_df['utilization_memory_pct']>cov_mean_df['utilization_memory_pct'].mean()//thrshold]
df1=cov_df['utilization_memory_pct_cov']
df1=df1.sort_values()
x=(np.arange(len(df1))/(len(df1)-1))*100
plt.plot(df1.values,x,linestyle='--',linewidth=2,label="Mem. Util.",color='black')


cov_mean_df[['memory_used_MiB','memory_used_MiB_cov']].dropna().sort_values('memory_used_MiB_cov')
cov_df=cov_mean_df[cov_mean_df['memory_used_MiB']>cov_mean_df['memory_used_MiB'].mean()//thrshold]
df1=cov_df['memory_used_MiB_cov']
df1=df1.sort_values()
x=(np.arange(len(df1))/(len(df1)-1))*100
plt.plot(df1.values,x,linestyle=':',linewidth=3,label="Mem. Size",color='blue')

plt.rcParams["figure.figsize"] = (3.1,2.6)
fig, axs = plt.subplots()

thrshold=5

cov_mean_df[['utilization_gpu_pct','utilization_gpu_pct_cov']].dropna().sort_values('utilization_gpu_pct_cov')
cov_df=cov_mean_df[cov_mean_df['utilization_gpu_pct']>cov_mean_df['utilization_gpu_pct'].mean()//thrshold]
df1=cov_df['utilization_gpu_pct_cov']
df1=df1.sort_values()
x=(np.arange(len(df1))/(len(df1)-1))*100
plt.plot(df1.values,x,linestyle='-',linewidth=2,label="SM Util.",color='red')


cov_mean_df[['utilization_memory_pct','utilization_memory_pct_cov']].dropna().sort_values('utilization_memory_pct_cov')
cov_df=cov_mean_df[cov_mean_df['utilization_memory_pct']>cov_mean_df['utilization_memory_pct'].mean()//thrshold]
df1=cov_df['utilization_memory_pct_cov']
df1=df1.sort_values()
x=(np.arange(len(df1))/(len(df1)-1))*100
plt.plot(df1.values,x,linestyle='--',linewidth=2,label="Mem. Util.",color='black')


cov_mean_df[['memory_used_MiB','memory_used_MiB_cov']].dropna().sort_values('memory_used_MiB_cov')
cov_df=cov_mean_df[cov_mean_df['memory_used_MiB']>cov_mean_df['memory_used_MiB'].mean()//thrshold]
df1=cov_df['memory_used_MiB_cov']
df1=df1.sort_values()
x=(np.arange(len(df1))/(len(df1)-1))*100
plt.plot(df1.values,x,linestyle=':',linewidth=3,label="Mem. Size",color='blue')






# cov_mean_df[['power_draw_W','power_draw_W_cov']].dropna().sort_values('power_draw_W_cov')
# cov_df=cov_mean_df[cov_mean_df['power_draw_W']>cov_mean_df['power_draw_W'].mean()//thrshold]
# df1=cov_df['power_draw_W_cov']
# df1=df1.sort_values()
# x=(np.arange(len(df1))/(len(df1)-1))*100
# plt.plot(df1.values,x,linestyle='--',linewidth=3,label="Power Drawn",color='darkorchid')



plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# plt.legend(fontsize=14);
axs.legend(fontsize=13,labelspacing=0.25,handlelength=1,borderaxespad=0.2,handletextpad=0.4,borderpad=0.2)

axs.set_xlabel("CoV when GPUs\nare Active (%)",fontdict={'fontsize':18})
axs.set_ylabel("Empirical CDF \n(% of Jobs)",fontdict={'fontsize':18});

plt.xlim(0,200)
plt.ylim(0,100)

# plt.axhspan(0, 30, 0, 0.8, facecolor='green', alpha=0.05)
# plt.axvspan(0, 80, 0,0.15,facecolor='green', alpha=0.05)
# plt.text(70,36,"80,30",fontsize=14)
# plt.stem([80],[30],linefmt='g')
# plt.hlines([30], 1, [80],colors='g')
# Cov is quite low. 80 % of the jobs have cov less than 30%
plt.grid(color='darkgrey', linestyle=':')
axs.set_yticks([0,25,50,75,100])
plt.legend(fontsize=13,edgecolor='black',borderpad=0.3,handletextpad=0.3,handlelength=1.5,labelspacing=0.2,borderaxespad=0.2)
plt.savefig('./plots/time_series/'+'plot2'+'.pdf',bbox_inches='tight');

df_ck1['Idle_steps_rm_first_step']=df_ck1['Idle_steps'].apply(lambda x : x[1:])
df_ck1['Active_steps_rm_first_step']=df_ck1['Active_steps_list'].apply(lambda x : x[1:])

df_ck1_two_steps=df_ck1[df_ck1['Idle_steps_rm_first_step'].apply(lambda x: len(x)>1)].copy() #only consider steps length two or more
df_ck1_two_steps=df_ck1[df_ck1['Active_steps_rm_first_step'].apply(lambda x: len(x)>1)].copy() #only consider steps length two or more

# # df_ck1_two_steps=df_ck1[df_ck1['count_intervals']>2].copy() #later
df_ck1_two_steps['Idle_steps_rm1_mean']=df_ck1_two_steps['Idle_steps_rm1'].apply(lambda x: np.mean(x))
df_ck1_two_steps['Idle_steps_rm1_median']=df_ck1_two_steps['Idle_steps_rm1'].apply(lambda x: np.median(x))
df_ck1_two_steps['Idle_steps_rm1_std']=df_ck1_two_steps['Idle_steps_rm1'].apply(lambda x: np.std(x))

df_ck1_two_steps['Active_steps_rm1_mean']=df_ck1_two_steps['Active_steps_rm_first_step'].apply(lambda x: np.mean(x))
df_ck1_two_steps['Active_steps_rm1_median']=df_ck1_two_steps['Active_steps_rm_first_step'].apply(lambda x: np.median(x))
df_ck1_two_steps['Active_steps_rm1_std']=df_ck1_two_steps['Active_steps_rm_first_step'].apply(lambda x: np.std(x))

# df_ck1_two_steps=df_ck1_two_steps.sort_values('Idle_steps_rm1_median')

df_ck1_two_steps['Idle_steps_mean']=df_ck1_two_steps['Idle_steps'].apply(lambda x: np.mean(x))
df_ck1_two_steps['Idle_steps_std']=df_ck1_two_steps['Idle_steps'].apply(lambda x: np.std(x))

df_ck1_two_steps['Active_steps_mean']=df_ck1_two_steps['Active_steps_list'].apply(lambda x: np.mean(x))
df_ck1_two_steps['Active_steps_std']=df_ck1_two_steps['Active_steps_list'].apply(lambda x: np.std(x))

df_ck1_two_steps['cov_idle']=df_ck1_two_steps['Idle_steps_std']/df_ck1_two_steps['Idle_steps_mean']*100
df_ck1_two_steps['cov_active']=df_ck1_two_steps['Active_steps_std']/df_ck1_two_steps['Active_steps_mean']*100

fig, axs = plt.subplots(figsize=(4, 2.6))


df1=df_ck1_two_steps['cov_idle'].sort_values()
x=(np.arange(len(df1))/(len(df1)-1))*100
plt.plot(df1.values,x,linestyle='-',linewidth=3,label="Idle Phase",color='gold')
plt.grid(color='lightgrey', linestyle=':')

df1=df_ck1_two_steps['cov_active'].sort_values()
x=(np.arange(len(df1))/(len(df1)-1))*100
plt.plot(df1.values,x,linestyle='--',linewidth=3,label="Active Phase",color='navy')
plt.grid(color='lightgrey', linestyle=':')


plt.ylim(0,100)
plt.xlim(0,500)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)



plt.grid(color='lightgrey', linestyle=':')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=15,edgecolor='black',borderpad=0.3,handletextpad=0.3,handlelength=1.5,labelspacing=0.2,loc="lower right")
# axs.set_xticks([0,125,250,375,500])
axs.set_yticks([0,25,50,75,100])

axs.set_xlabel("CoV of Interval Length (%)",fontdict={'fontsize':18})
axs.set_ylabel("Empirical CDF (%)",fontdict={'fontsize':18});

plt.savefig('./plots/time_series/'+'plot4'+'.pdf',bbox_inches='tight');
