import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import plot_functions
import numpy as np
from collections import Counter
import matplotlib as mp

def plot_cdf_utilization(gpu_join):

    fig, axs = plt.subplots(1, 2, figsize=(6.2,2.6), sharey=False, gridspec_kw={'hspace': 0.2, 'wspace': 0.19, 'bottom':0.28, 'left':0.12, 'right':0.99, 'top':0.96})
    metric='smutilization_pct_avg'
    sm_util = gpu_join[metric].sort_values().to_frame()
    sm_cdf = sm_util.rank(pct=True)*100
    metric='memoryutilization_pct_avg'
    mem_util = gpu_join[metric].sort_values().to_frame()
    mem_cdf = mem_util.rank(pct=True)*100
    metric='maxgpumemoryused_bytes'
    mem_used = gpu_join[metric].sort_values().to_frame()
    mem_used = mem_used / mem_used.max() * 100
    used_cdf = mem_used.rank(pct=True)*100
    
    axs[0].set_xlabel('GPU Usage (%)', fontsize=13)
    axs[0].set_ylabel('Empirical CDF (%)', fontsize=13)
    axs[0].plot(sm_util, sm_cdf, label='SM Util.', color='orangered', ls='solid', lw=2.5)
    axs[0].plot(mem_util, mem_cdf, label='Mem. Util.', color='navy', ls='dashed', lw=2.5)
    axs[0].plot(mem_used, used_cdf, label='Mem. Used', color='lime', ls='dashdot', lw=2.5)
    
    axs[0].legend(loc=4)
    axs[0].set_ylim(0,100)
    axs[0].set_xlim(0,100)
    
    metric='pcietxbandwidth_megabytes_avg'
    Tx = gpu_join[metric].sort_values().to_frame()
    Tx_cdf = Tx.rank(pct=True)*100
    metric='pcierxbandwidth_megabytes_avg'
    Rx = gpu_join[metric].sort_values().to_frame()
    Rx_cdf = Rx.rank(pct=True)*100
    
    axs[1].set_xlabel('PCIe Bandwidth (MB)', fontsize=13)
    axs[1].plot(Tx, Tx_cdf, label='Tx BW.', color='orangered', ls='solid', lw=2.5)
    axs[1].plot(Rx, Rx_cdf, label='Rx BW.', color='navy', ls='dashed', lw=2.5)
    axs[1].set_ylabel('', fontsize=13)
    axs[1].legend(loc=4)
    axs[1].set_ylim(0,100)
    axs[1].set_xlim(0,)
    axs[1].xaxis.set_major_locator(MultipleLocator(500))
    tick_label = ['0', '0.5k', '1k', '1.5k', '2k']
    
    for ax in axs:
        ax.grid(which='major', axis='both', color='darkgrey', ls=':')
        ax.tick_params(labelsize=13)
    
    fig.text(0.3, 0.04, '(a)', ha='center', va='center', fontsize=12)
    fig.text(0.78, 0.04, '(b)', ha='center', va='center', fontsize=12)
    plt.show()

def plot_job_types_pie():
    sizes=np.array([28174,8349,8965,1632])
    sizes=sizes/sum(sizes)*100
    sizes=np.round(sizes,1)
    labels=["Mature", "Exploratory", "Development", "IDE"]
    sizes_hours=np.array([824288434.4700054,728193112.2700019,177956408.22999993,384975397.70000017])
    sizes_hours=sizes_hours/sum(sizes_hours)*100
    sizes_hours=np.round(sizes_hours,1)
    fig, ax = plt.subplots(1,2,figsize=(6.2, 3), subplot_kw=dict(aspect="equal"))
    plt.rcParams['font.size'] = 15
    colors=["maroon","orangered","salmon","mistyrose"]
    wedgeprops={"edgecolor":"black",'linewidth': 1.5, 'linestyle': '-', 'antialiased': True,"width":0.5}
    ax[0].pie(sizes, colors=colors, wedgeprops=wedgeprops, startangle=0, labels=[str(b)+" %" for b in sizes])
    # fig.subplots_adjust(left=0.085, right=0.995, top=0.955, bottom=0.25, wspace=0.42)
    fig.subplots_adjust(wspace=1)
    ax[0].set_xlabel('(a) Num. Jobs',fontsize=18)
    ax[1].pie(sizes_hours, colors=colors, wedgeprops=wedgeprops, startangle=50, labels=[str(b)+" %" for b in sizes_hours])
    ax[1].set_xlabel('(b) Num. GPU Hours',fontsize=18)
    ax[0].legend(labels ,ncol=4, loc='right', bbox_to_anchor=(3.5, 1.2),
               fontsize=15,edgecolor='black',borderpad=0.3,handletextpad=0.3,handlelength=1,columnspacing=1.5)
    plt.show()

def plot_users_and_job_types(gpu_user_join, gpu_user_join_rm_small_runtime):

    gpu_user_join_rm_small_runtime['GPU_hours']=gpu_user_join_rm_small_runtime['totalexecutiontime_sec']*gpu_user_join_rm_small_runtime['GPU_Count']
    
    gpu_user_join_rm_small_runtime['fail_type']=gpu_user_join_rm_small_runtime['state'].map({3:'Mature',4: 'Exploratory', 5: 'Development', 6: 'IDE',7: 'IDE',11: 'IDE',1024: 'IDE'})
    
    '''
    slurm "state" id -> slurm decoding -> % of jobs -> name used in paper
    3 -> JOB_COMPLETE -> 59.8% -> Completed -> Mature
    4 -> JOB_CANCELLED -> user cancel -> 17.72 % -> Cancelled -> Exploratory
    5 -> JOB_FAILED -> 19.02% -> Failed (prev runtime error) -> Development
    6 -> JOB_TIMEOUT -> 3.27 % -> Timeout+ -> IDE
    7 -> JOB_NODE_FAIL -> 0.01 % -> Timeout+ -> IDE
    11, 1024 -> No decodings  0.18% -> Timeout+ -> IDE
    ''';
    
    gpu_user_join_rm_small_runtime_JOB_COMPLETE=gpu_user_join_rm_small_runtime[gpu_user_join_rm_small_runtime['state']==3]
    gpu_user_join_rm_small_runtime_EXIT=gpu_user_join_rm_small_runtime[gpu_user_join_rm_small_runtime['state']!=3]
    gpu_user_join_JOB_COMPLETE=gpu_user_join[gpu_user_join['state']==3]
    gpu_user_join_EXIT=gpu_user_join[gpu_user_join['state']!=3]
    
    gpu_user_join_rm_small_runtime_EXIT['fail_type']=gpu_user_join_rm_small_runtime_EXIT['state'].map({4: 'Canceled', 5: 'Failed', 6: 'Timeout+',7: 'Timeout+',11: 'Timeout+',1024: 'Timeout+'})
    col_for_tk_exit=['totalexecutiontime_sec','avgmemoryutilization_pct','smutilization_pct_avg','maxgpumemoryused_bytes','GPU_Count','GPU_hours','fail_type']
    col_for_tk=['totalexecutiontime_sec','avgmemoryutilization_pct','smutilization_pct_avg','maxgpumemoryused_bytes','GPU_Count','GPU_hours']
#    gpu_user_join_rm_small_runtime_EXIT[col_for_tk_exit].to_csv('fail_job_stats.csv')
#    gpu_user_join_rm_small_runtime_JOB_COMPLETE[col_for_tk].to_csv('complete_job_stats.csv')
    
    df_all=gpu_user_join_rm_small_runtime[['id_user','GPU_hours']].groupby('id_user').sum()
    df_any_fail=gpu_user_join_rm_small_runtime_EXIT[['id_user','GPU_hours']].groupby('id_user').sum()
    df_Complete=gpu_user_join_rm_small_runtime_JOB_COMPLETE[['id_user','GPU_hours']].groupby('id_user').sum()
    df_Canceled=gpu_user_join_rm_small_runtime_EXIT[gpu_user_join_rm_small_runtime_EXIT['fail_type']=='Canceled'][['id_user','GPU_hours']].groupby('id_user').sum()
    df_Failed=gpu_user_join_rm_small_runtime_EXIT[gpu_user_join_rm_small_runtime_EXIT['fail_type']=='Failed'][['id_user','GPU_hours']].groupby('id_user').sum()
    df_Timeout=gpu_user_join_rm_small_runtime_EXIT[gpu_user_join_rm_small_runtime_EXIT['fail_type']=='Timeout+'][['id_user','GPU_hours']].groupby('id_user').sum()
    
    hash_users_Complete=set(df_Complete.index)
    hash_users_Canceled=set(df_Canceled.index)
    hash_users_Failed=set(df_Failed.index)
    hash_users_Timeout=set(df_Timeout.index)
    hash_users_any_fail=set(df_any_fail.index)
    
    df_all['GPU_hours_Complete']=0
    df_all['GPU_hours_Canceled']=0
    df_all['GPU_hours_Failed']=0
    df_all['GPU_hours_Timeout']=0
    df_all['GPU_hours_any_fail']=0
    for job in df_all.index:
        if job in hash_users_Complete:
            df_all.loc[job,'GPU_hours_Complete']=df_Complete.loc[job,'GPU_hours']
        if job in hash_users_Canceled:
            df_all.loc[job,'GPU_hours_Canceled']=df_Canceled.loc[job,'GPU_hours']
        if job in hash_users_Failed:
            df_all.loc[job,'GPU_hours_Failed']=df_Failed.loc[job,'GPU_hours']
        if job in hash_users_Timeout:
            df_all.loc[job,'GPU_hours_Timeout']=df_Timeout.loc[job,'GPU_hours']
        if job in hash_users_any_fail:
            df_all.loc[job,'GPU_hours_any_fail']=df_any_fail.loc[job,'GPU_hours']
    
    df_all2=gpu_user_join_rm_small_runtime[['id_user','GPU_hours']].groupby('id_user').count().rename(columns={'GPU_hours':'Jobs'})
    df_Complete=gpu_user_join_rm_small_runtime_JOB_COMPLETE[['id_user','GPU_hours']].groupby('id_user').count().rename(columns={'GPU_hours':'Jobs'})
    df_Canceled=gpu_user_join_rm_small_runtime_EXIT[gpu_user_join_rm_small_runtime_EXIT['fail_type']=='Canceled'][['id_user','GPU_hours']].groupby('id_user').count().rename(columns={'GPU_hours':'Jobs'})
    df_Failed=gpu_user_join_rm_small_runtime_EXIT[gpu_user_join_rm_small_runtime_EXIT['fail_type']=='Failed'][['id_user','GPU_hours']].groupby('id_user').count().rename(columns={'GPU_hours':'Jobs'})
    df_Timeout=gpu_user_join_rm_small_runtime_EXIT[gpu_user_join_rm_small_runtime_EXIT['fail_type']=='Timeout+'][['id_user','GPU_hours']].groupby('id_user').count().rename(columns={'GPU_hours':'Jobs'})
    
    hash_users_Complete=set(df_Complete.index)
    hash_users_Canceled=set(df_Canceled.index)
    hash_users_Failed=set(df_Failed.index)
    hash_users_Timeout=set(df_Timeout.index)
    
    df_all2['jobs_Complete']=0
    df_all2['jobs_Canceled']=0
    df_all2['jobs_Failed']=0
    df_all2['jobs_Timeout']=0
    for job in df_all2.index:
        if job in hash_users_Complete:
            df_all2.loc[job,'jobs_Complete']=df_Complete.loc[job,'Jobs']
        if job in hash_users_Canceled:
            df_all2.loc[job,'jobs_Canceled']=df_Canceled.loc[job,'Jobs']
        if job in hash_users_Failed:
            df_all2.loc[job,'jobs_Failed']=df_Failed.loc[job,'Jobs']
        if job in hash_users_Timeout:
            df_all2.loc[job,'jobs_Timeout']=df_Timeout.loc[job,'Jobs']
    
    gpu_user_join_rm_small_runtime['time_wait']=gpu_user_join_rm_small_runtime['time_start']-gpu_user_join_rm_small_runtime['time_submit']
    metric='GPU_hours_Complete_pct'
    df_all[metric]=(df_all['GPU_hours_Complete']/df_all['GPU_hours'])*100
    metric='GPU_hours_Canceled_pct'
    df_all[metric]=(df_all['GPU_hours_Canceled']/df_all['GPU_hours'])*100
    metric='GPU_hours_Failed_pct'
    df_all[metric]=(df_all['GPU_hours_Failed']/df_all['GPU_hours'])*100
    metric='GPU_hours_Timeout_pct'
    df_all[metric]=(df_all['GPU_hours_Timeout']/df_all['GPU_hours'])*100
    metric='jobs_Complete_pct'
    df_all2[metric]=(df_all2['jobs_Complete']/df_all2['Jobs'])*100
    metric='jobs_Canceled_pct'
    df_all2[metric]=(df_all2['jobs_Canceled']/df_all2['Jobs'])*100
    metric='jobs_Failed_pct'
    df_all2[metric]=(df_all2['jobs_Failed']/df_all2['Jobs'])*100
    metric='jobs_Timeout_pct'
    df_all2[metric]=(df_all2['jobs_Timeout']/df_all2['Jobs'])*100
    
    
    df_tmp=df_all[['GPU_hours_Complete_pct','GPU_hours_Canceled_pct','GPU_hours_Failed_pct','GPU_hours_Timeout_pct']]
    df_tmp['dummy_col']=df_tmp['GPU_hours_Complete_pct']+df_tmp['GPU_hours_Canceled_pct']
    df_tmp=df_tmp.sort_values('GPU_hours_Complete_pct')
    gpuh_data=(df_tmp.transpose().to_numpy().tolist())
    
    df_tmp=df_all2[['jobs_Complete_pct','jobs_Canceled_pct','jobs_Failed_pct','jobs_Timeout_pct']]
    df_tmp['dummy_col']=df_tmp['jobs_Complete_pct']+df_tmp['jobs_Canceled_pct']
    df_tmp=df_tmp.sort_values('jobs_Complete_pct')
    jobs_data=(df_tmp.transpose().to_numpy().tolist())

    jobs_data = jobs_data
    gpuh_data = gpuh_data
    fig = plt.figure(figsize=(5.5, 2.5))
    fig.subplots_adjust(left=0.125, top=0.83, right=0.965, bottom=0.205, wspace=0.55)
    axl = fig.add_subplot(111, frameon=False)
    axl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    axl.set_xticks([])
    axl.set_yticks([])
    ax = fig.add_subplot(121)
    d0 = jobs_data[0]
    d1 = [jobs_data[0][j] + jobs_data[1][j] for j in range(len(d0))]
    d2 = [jobs_data[0][j] + jobs_data[1][j] + jobs_data[2][j] for j in range(len(d0))]
    d3 = [jobs_data[0][j] + jobs_data[1][j] + jobs_data[2][j] + jobs_data[3][j] for j in range(len(d0))]
    plots = []
    plots.append(ax.fill_between(range(len(d0)),  0, d0, color='maroon'))
    plots.append(ax.fill_between(range(len(d0)), d0, d1, color='orangered'))
    plots.append(ax.fill_between(range(len(d0)), d1, d2, color='salmon'))
    plots.append(ax.fill_between(range(len(d0)), d2, d3, color='mistyrose'))
    ax.set_xticks(np.linspace(0, len(d0)-1, 5))
    ax.set_xticklabels([str(y) for y in range(0, 101, 25)])
    ax.set_yticks(range(0, 101, 25))
    ax.grid(color='white', linestyle=':',linewidth=1)
    ax.set_xlim([0, len(d0)-1])
    ax.set_ylim([0, 100])
    plt.setp(ax.get_xticklabels(), fontsize=13)
    plt.setp(ax.get_yticklabels(), fontsize=13)
    ax.set_xlabel('Num. Users (%)', fontsize=13)
    ax.set_ylabel('Num. Jobs (%)', fontsize=13)
    ax = fig.add_subplot(122)
    d0 = gpuh_data[0]
    d1 = [gpuh_data[0][j] + gpuh_data[1][j] for j in range(len(d0))]
    d2 = [gpuh_data[0][j] + gpuh_data[1][j] + gpuh_data[2][j] for j in range(len(d0))]
    d3 = [gpuh_data[0][j] + gpuh_data[1][j] + gpuh_data[2][j] + gpuh_data[3][j] for j in range(len(d0))]
    plots = []
    plots.append(ax.fill_between(range(len(d0)),  0, d0, color='maroon'))
    plots.append(ax.fill_between(range(len(d0)), d0, d1, color='orangered'))
    plots.append(ax.fill_between(range(len(d0)), d1, d2, color='salmon'))
    plots.append(ax.fill_between(range(len(d0)), d2, d3, color='mistyrose'))
    ax.set_xticks(np.linspace(0, len(d0)-1, 5))
    ax.set_xticklabels([str(y) for y in range(0, 101, 25)])
    ax.set_yticks(range(0, 101, 25))
    ax.set_xlim([0, len(d0)-1])
    ax.set_ylim([0, 100])
    ax.grid(color='white', linestyle=':',linewidth=1)
    plt.setp(ax.get_xticklabels(), fontsize=13)
    plt.setp(ax.get_yticklabels(), fontsize=13)
    ax.set_xlabel('Num. Users (%)', fontsize=13)
    ax.set_ylabel('Num. GPU\nHours (%)', fontsize=13)
    axl.legend(plots, ['Mature', 'Exploratory', 'Development', 'IDE'],
	bbox_to_anchor=(0.0, 1.075, 1.0, 0.102), loc='lower left', ncol=4,
	borderaxespad=0., fontsize=13, edgecolor='black', mode='expand',
	handletextpad=0.2, borderpad=0.3, handlelength=1.2)
    plt.show()

def plot_multi_gpu_jobs(gpu_user_join_rm_small_runtime):
    df=gpu_user_join_rm_small_runtime #filter small runtime jobs for this
    dict_gpu_job_count=Counter(df['GPU_Range'])
    sorted_list=sorted(dict_gpu_job_count.items(), key=lambda k: k[0])
    print ("GPU Count Range","\t","Jobs","\t","Percentage_of_jobs")
    for code, jobs in sorted_list:
        print (code,"\t",jobs,"\t",round(jobs/df.shape[0]*100,2))
    
    pct_list=[f"{np.round(lst[1]/gpu_user_join_rm_small_runtime.shape[0]*100,1)}" for lst in sorted_list]
    label_list=[f"{lst[0]}" for lst in sorted_list]
    label_list=['1 GPU','2 GPUs','3-8 GPUs','>8 GPUs']
    gpu_user_join_rm_small_runtime['GPU_hours']=gpu_user_join_rm_small_runtime.GPU_Count*gpu_user_join_rm_small_runtime.totalexecutiontime_sec
    df1=gpu_user_join_rm_small_runtime.groupby('GPU_Range').sum()['GPU_hours']
    pct_list2=(df1/sum(df1)*100).values
    pct_list2=pct_list2.round(1)
    
    fig, ax = plt.subplots(1,2,figsize=(6.2, 3), subplot_kw=dict(aspect="equal"))
    fig.subplots_adjust(wspace=1)
    plt.rcParams['font.size'] = 15
    wedges, texts = ax[0].pie(pct_list, colors=["azure","aqua","teal","navy"],\
                    wedgeprops={"edgecolor":"black",'linewidth': 1.5, 'linestyle': '-',\
                    'antialiased': True,"width":0.5}, startangle=0, labels=[str(pct_list[0])+'%',\
                    str(pct_list[1])+'%','',''],textprops={'fontsize': 15})
    
    kw = dict(arrowprops=dict(arrowstyle="-"),
              zorder=0, va="center")
    
    for i, p in enumerate(wedges):
        if i == 2:
            ang = (p.theta2 - p.theta1)/2. + p.theta1 -3
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax[0].annotate(str(pct_list[i])+'%', xy=(x, y), xytext=(1.7*np.sign(x), 1.7*y),
                    horizontalalignment=horizontalalignment, **kw,fontsize=15)
        if i == 3:
            ang = (p.theta2 - p.theta1)/2. + p.theta1  + 3
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax[0].annotate(str(pct_list[i])+'%', xy=(x, y), xytext=(1.7*np.sign(x), 1.7*y),
                    horizontalalignment=horizontalalignment, **kw,fontsize=15)
    
    ax[0].legend(label_list ,ncol=4, loc='right', bbox_to_anchor=(3.2, 1.2),
               fontsize=15,edgecolor='black',borderpad=0.3,handletextpad=0.3,handlelength=1,columnspacing=1.5)
    ax[1].pie(pct_list2, colors=["azure","aqua","teal","navy"],wedgeprops={"edgecolor":"black",'linewidth': 1.5,\
                                'linestyle': '-', 'antialiased': True,"width":0.5}, startangle=0, \
              labels=[str(pct_list2[0])+'%', str(pct_list2[1])+'%',str(pct_list2[2])+'%',str(pct_list2[3])+'%'],\
              textprops={'fontsize': 15})
    ax[0].set_xlabel('(a) Num. Jobs',fontsize=18)
    ax[1].set_xlabel('(b) Num. GPU Hours',fontsize=18)
    plt.show()

def plot_jobs_by_same_user(gpu_user_join_rm_small_runtime):
    gpu_user_join_rm_small_runtime['maxgpumemoryused_bytes_pct']=gpu_user_join_rm_small_runtime['maxgpumemoryused_bytes']/3.2e+10*100
    fig = plt.figure(figsize=(7, 2.2))
    fig.subplots_adjust(left=0.125, right=0.99, top=0.895, bottom=0.245, wspace=0.1)
    axl = fig.add_subplot(111, frameon=False)
    axl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    axl.set_xticks([])
    axl.set_yticks([])
    gs = fig.add_gridspec(1, 4)
    ax = []
    
    metric=['totalexecutiontime_sec', 'smutilization_pct_avg','avgmemoryutilization_pct','maxgpumemoryused_bytes_pct']
    xlabel=['Avg. Run Time\n (Minute)', 'Avg. SM \nUtil. (%)','Avg. Mem \nUtil. (%)','Avg. Mem \nSize (%)']
    color=['pink','hotpink','mediumvioletred','indigo'][::-1]
    style=['-','--','-.','-']
    # xlabel=['Average SM Utilization %','']
    for i in range(4):
        ax.append(fig.add_subplot(gs[0, i]))
        ax[i].set_axisbelow(True)
        ax[i].grid(color='darkgrey', linestyle=':')
        tmp_df=gpu_user_join_rm_small_runtime[['id_user',metric[i]]].groupby('id_user')[metric[i]]
        mean_series=tmp_df.mean()
        mean_df=mean_series.sort_values()
        if i==0:
            mean_df=mean_df/60 #convert to minutes
        val = mean_df.values.tolist()
        x_pos = (np.arange(len(mean_df))/(len(mean_df)-1))*100
        ax[i].plot(val , x_pos ,linestyle=style[i],linewidth=2,color=color[i])
    #     ax[i].legend(fontsize=15,edgecolor='black',borderpad=0.3,handletextpad=0.3,handlelength=1.5,labelspacing=0.2,frameon=False)
        md = 0
        ax[i].set_xlim((0, 100))
        ax[i].set_ylim((0, 100))
        if i == 0:
            ax[i].set_ylabel('Empirical CDF\n(% of Users)', fontsize=14)
        else:
            ax[i].set_yticklabels([])
    
        plt.setp(ax[i].get_xticklabels(), fontsize=14)
        plt.setp(ax[i].get_yticklabels(), fontsize=14)
        ax[i].set_xlabel(xlabel[i], fontsize=14)
        ax[i].set_xticks([0, 25, 50, 75])
        if i == 0:
            ax[i].set_xscale('log')
            ax[i].set_xlim((10**-1, 10**4))
        plt.show()

def plot_cdf_types(gpu_join):
    types = ['map_reduce', 'batch', 'iteractive', 'other']
    jobs = {}
    jobs['map_reduce'] = gpu_join[gpu_join.job_type == 'LLMAPREDUCE:MAP']
    jobs['batch'] = gpu_join[gpu_join.job_type == 'LLSUB:BATCH']
    jobs['interactive'] = gpu_join[gpu_join.job_type == 'LLSUB:INTERACTIVE']
    jobs['other'] = gpu_join[gpu_join.job_type == 'OTHER']
    
    fig, axs = plt.subplots(1, 2, figsize=(6,2.5), sharey=False, gridspec_kw={'hspace': 0.2, 'wspace': 0.19, 'bottom':0.28, 'left':0.12, 'right':0.98, 'top':0.96})
    
    colors = ['orangered', 'navy', 'lime', 'orchid', 'gold']
    styles = ['solid', 'dashed', 'dashdot', 'dotted']
    
    for key, val in jobs.items():
        index = list(jobs.keys()).index(key)
        x = val.smutilization_pct_avg.sort_values().to_frame()
        y = x.rank(pct=True)*100
        axs[0].plot(x, y, label=key, ls=styles[index], color=colors[index], lw=2)
        axs[0].set_xlabel('SM Utilization (%)', fontsize=13)
        axs[0].set_ylabel('Empirical CDF (%)', fontsize=13)
        axs[0].set_xlim(-1,80)
    
    for key, val in jobs.items():
        index = list(jobs.keys()).index(key)
        x = val.memoryutilization_pct_avg.sort_values().to_frame()
        y = x.rank(pct=True)*100
        axs[1].plot(x, y, label=key, ls=styles[index], color=colors[index], lw=2)
        axs[1].set_xlabel('Memory Utilization (%)', fontsize=13)
        axs[1].set_xlim(-1,50)
        axs[1].xaxis.set_major_locator(MultipleLocator(10))
    
    for ax in axs:
        ax.set_ylim(-1,100)
        ax.grid(which='major', axis='both', color='darkgrey', ls=':')
        ax.tick_params(labelsize=13)
        ax.yaxis.set_major_locator(MultipleLocator(20))
    
    axs[1].legend(loc=4, handlelength=3, fontsize=11)
    
    fig.text(0.3, 0.04, '(a)', ha='center', va='center', fontsize=12)
    fig.text(0.78, 0.04, '(b)', ha='center', va='center', fontsize=12)
    plt.show()

def plot_power_cap(gpu_join):
    fig, axs = plt.subplots(1, 2, figsize=(6.2,2.8), sharey=False, gridspec_kw={'hspace': 0.3, 'wspace': 0.3, 'bottom':0.25, 'left':0.11, 'right':0.98, 'top':0.8})

    metric='powerusage_watts_avg'
    avg_pwr = gpu_join[metric].sort_values().to_frame()
    avg_cdf = avg_pwr.rank(pct=True)*100
    avg_pwr_list = gpu_join[metric].tolist()
    
    metric='powerusage_watts_max'
    max_pwr = gpu_join[metric].sort_values().to_frame()
    max_cdf = max_pwr.rank(pct=True)*100
    max_pwr_list = gpu_join[metric].tolist()
    
    axs[0].set_xlabel('GPU Power (Watt)', fontsize=12.5)
    axs[0].set_ylabel('Empirical CDF (%)', fontsize=12.5)
    axs[0].plot(avg_pwr, avg_cdf, label='Avg. Power', color='orangered', ls='solid', lw=2.5)
    axs[0].plot(max_pwr, max_cdf, label='Max. Power', color='navy', ls='dashed', lw=2.5)
    
    axs[0].legend(loc='upper left', ncol=1, mode="expand",
                  borderaxespad=0.1, edgecolor='black', handletextpad=0.5, handlelength=2.5, fontsize=10, bbox_to_anchor=(-0.,0.85,.6,0.5))
    axs[0].set_ylim(0,100)
    axs[0].set_xlim(25,300)
    axs[0].xaxis.set_major_locator(MultipleLocator(50))
    axs[0].xaxis.set_minor_locator(MultipleLocator(25))
    axs[0].grid(which='both', axis='both', color='darkgrey', ls=':')
    
    ########## power capping #################
    
    caps = [150, 200, 250]
    total = len(avg_pwr)
    x = np.arange(len(caps))
    width = 0.3
    
    max_pct = []
    avg_pct = []
    none_pct = []
    
    for cap in caps:
        pwr_thr = cap
        unaffected = 0
        avg_affect = 0
        max_affect = 0
        for i in range(total):
            if max_pwr_list[i] <= pwr_thr:
                unaffected += 1
            elif avg_pwr_list[i] <= pwr_thr:
                max_affect += 1
            elif avg_pwr_list[i] > pwr_thr:
                avg_affect += 1
            else:
                print('error: should not exist')
                sys.exit()
    
        none_pct.append(round(unaffected / total * 100,2))
        avg_pct.append(round(avg_affect / total * 100,2))
        max_pct.append(round(max_affect / total * 100,2))
    
    labels = [str(cap) for cap in caps]
    axs[1].bar(x-width/2, none_pct, width, label='No Impact', color='orangered', alpha=0.5, edgecolor='black')#'darksalmon')
    axs[1].bar(x+width/2, avg_pct, width, label='Avg. Pwr.', color='navy', alpha=0.8, edgecolor='black')#'darkturquoise')
    axs[1].bar(x+width/2, max_pct, width, label='Max. Pwr.', bottom=np.asarray(avg_pct), color='lime', alpha=0.5, edgecolor='black')#'darkseagreen')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels)
    axs[1].set_xlabel('Power Cap (Watt)', fontsize=12.5)
    axs[1].set_ylabel('Impacted Jobs (%)', fontsize=12.5)
    axs[1].set_ylim(0,95)
    
    handles, labels = axs[1].get_legend_handles_labels()
    #axs[1].legend(loc=2)
    #plt.figlegend(handles, labels, loc='upper left', ncol=3, mode="expand",
    #              borderaxespad=0.1, edgecolor='black', handletextpad=0.5, handlelength=2, fontsize=10, bbox_to_anchor=(0.4,0.5,0.58,0.5))
    axs[1].legend(loc='upper left', ncol=2, mode="expand",
                  borderaxespad=0.1, edgecolor='black', handletextpad=0.5, handlelength=2., fontsize=10, bbox_to_anchor=(-0.,0.85,1,0.5))
    
    
    axs[1].yaxis.set_major_locator(MultipleLocator(20))
    axs[1].yaxis.set_minor_locator(MultipleLocator(10))
    
    axs[1].grid(which='both', axis='y', ls=':', color='darkgrey')
    
    for ax in axs:
        ax.tick_params(labelsize=12.5)
    
    fig.text(0.3, 0.04, '(a)', ha='center', va='center', fontsize=12)
    fig.text(0.78, 0.04, '(b)', ha='center', va='center', fontsize=12)
    plt.show()

def plot_bottlenecks(gpu_join):

    bottlenecks = {'Mem. Util.': 'memoryutilization_pct_max', 'SM Util.': 'smutilization_pct_max', 'PCIe Rx': 'pcierxbandwidth_megabytes_max', 'PCIe Tx': 'pcietxbandwidth_megabytes_max', 'Mem. Size': 'maxgpumemoryused_bytes'}
    bound = {'SM Util.': 100, 'Mem. Util.': 100, 'Mem. Size': gpu_join.maxgpumemoryused_bytes.max(), 'PCIe Tx': gpu_join.pcietxbandwidth_megabytes_max.max(), 'PCIe Rx': gpu_join.pcierxbandwidth_megabytes_max.max()}
    
    bounded = {}
    
    total = len(gpu_join)
    for key, val in bottlenecks.items():
        num_bounded = len(gpu_join[gpu_join[val] >= bound[key] * 0.99])
        bounded[key] = num_bounded / total * 100
    
    y = list(bounded.values())
    x = list(bounded.keys())
    data = [x, ('title', y)]
    N = len(data[0])
    theta = plot_functions.radar_factory(N, frame='polygon')
    fig, axs = plt.subplots(1,1,figsize=(4, 4), subplot_kw=dict(projection='radar'))
    spoke_labels = data.pop(0)
    title, case_data = data[0]
    
    #axs[0].set_title(title,  position=(0.5, 1.1), ha='center')
    axs.set_ylim(0,25)
    axs.set_yticks(np.arange(5,26,5))
    axs.tick_params(axis = 'y', which = 'major', labelsize = 12)
    
    color = 'orangered'
    
    line = axs.plot(theta, y, color=color)
    axs.fill(theta, y,  alpha=0.25, facecolor=color)
    
    axs.set_varlabels(spoke_labels)
    plt.show()

def plot_res_bound(gpu_user_join_rm_small_runtime):
    sm_max='smutilization_pct_max'
    mem_util_max='memoryutilization_pct_max'
    mem_size_max='maxgpumemoryused_bytes'
    pcie_rx_max='pcierxbandwidth_megabytes_max'
    pcie_tx_max='pcietxbandwidth_megabytes_max'
    
    sm_bound=sum(gpu_user_join_rm_small_runtime[sm_max]>=99)/47120*100
    mem_util_bound=sum(gpu_user_join_rm_small_runtime[mem_util_max]>=99)/47120*100
    mem_size_bound=sum(gpu_user_join_rm_small_runtime[mem_size_max]>=.99*34083962880.0)/47120*100
    pcie_rx_bound=sum(gpu_user_join_rm_small_runtime[pcie_rx_max]>=2147*.99)/47120*100
    pcie_tx_bound=sum(gpu_user_join_rm_small_runtime[pcie_tx_max]>=2147*.99)/47120*100
    
    plt.rcParams["figure.figsize"] = (4,3)
    
    fig, axs = plt.subplots(1, 1)
    axs.grid(color='darkgrey', linestyle=':',axis='y')
    bar_names=["SM. Util.","Mem. Util.","Mem. Size","PCIe Rx","PCIe Tx"]
    x_pos=np.arange(len(bar_names))
    plt.xticks(x_pos,bar_names)
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(rotation = 45);
    axs.set_xlabel('Bounded Resource', fontsize=18)
    axs.set_ylabel('Jobs (%)',fontsize=18)
    
    plt.bar(x_pos,[sm_bound,mem_util_bound,mem_size_bound,pcie_rx_bound,pcie_tx_bound]);
    
    axs.set_axisbelow(True)   
    plt.show()

#def plot_double_res_bound(gpu_user_join_rm_small_runtime):
    sm_bound_and_mem_util=sum((gpu_user_join_rm_small_runtime[mem_util_max]>=99) & (gpu_user_join_rm_small_runtime[sm_max]>=99))/47120*100
    sm_bound_and_mem_size=sum((gpu_user_join_rm_small_runtime[sm_max]>=99) & (gpu_user_join_rm_small_runtime[mem_size_max]>=.99*34083962880.0))/47120*100
    sm_bound_and_pcie_rx=sum((gpu_user_join_rm_small_runtime[sm_max]>=99) & (gpu_user_join_rm_small_runtime[pcie_rx_max]>=2147*.99))/47120*100
    sm_bound_and_pcie_tx=sum((gpu_user_join_rm_small_runtime[sm_max]>=99) & (gpu_user_join_rm_small_runtime[pcie_tx_max]>=2147*.99))/47120*100
    
    mem_util_and_mem_size=sum((gpu_user_join_rm_small_runtime[mem_util_max]>=99) & (gpu_user_join_rm_small_runtime[mem_size_max]>=.99*34083962880.0))/47120*100
    mem_util_and_pcie_rx=sum((gpu_user_join_rm_small_runtime[mem_util_max]>=99) & (gpu_user_join_rm_small_runtime[pcie_rx_max]>=2147*.99))/47120*100
    mem_util_and_pcie_tx=sum((gpu_user_join_rm_small_runtime[mem_util_max]>=99) & (gpu_user_join_rm_small_runtime[pcie_tx_max]>=2147*.99))/47120*100
    
    mem_size_and_pcie_rx=sum((gpu_user_join_rm_small_runtime[mem_size_max]>=.99*34083962880.0) & (gpu_user_join_rm_small_runtime[pcie_rx_max]>=2147*.99))/47120*100
    mem_size_and_pcie_tx=sum((gpu_user_join_rm_small_runtime[mem_size_max]>=.99*34083962880.0) & (gpu_user_join_rm_small_runtime[pcie_tx_max]>=2147*.99))/47120*100
    
    pcie_rx_and_pcie_tx=sum((gpu_user_join_rm_small_runtime[pcie_rx_max]>=2147*.99) & (gpu_user_join_rm_small_runtime[pcie_tx_max]>=2147*.99))/47120*100
    
    plt.rcParams["figure.figsize"] = (4,4)
    fig, axs = plt.subplots(1, 1)
    H = np.array([
              [sm_bound,sm_bound_and_mem_util,sm_bound_and_mem_size,sm_bound_and_pcie_rx,sm_bound_and_pcie_tx],
              [sm_bound_and_mem_util,mem_util_bound,mem_util_and_mem_size,mem_util_and_pcie_rx,mem_util_and_pcie_tx],
              [sm_bound_and_mem_size,mem_util_and_mem_size,mem_size_bound,mem_size_and_pcie_rx,mem_size_and_pcie_tx],
              [sm_bound_and_pcie_rx,mem_util_and_pcie_rx,mem_size_and_pcie_rx,pcie_rx_bound,pcie_rx_and_pcie_tx],
              [sm_bound_and_pcie_tx,mem_util_and_pcie_tx,mem_size_and_pcie_tx,pcie_rx_and_pcie_tx,pcie_tx_bound]])
    
    x_start = 0
    x_end = 4
    y_start = 0
    y_end = 4
    extent = [x_start, x_end, y_start, y_end]
    plt.xticks(x_pos,bar_names)
    plt.yticks(x_pos,bar_names)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15);
    plt.xticks(rotation = 45);
    
    size = 5
    # Add the text
    jump_x = (x_end - x_start) / (2.0 * size)
    jump_y = (y_end - y_start) / (2.0 * size)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=True)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=True)
    
    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = round(H[y_index, x_index],1)
            text_x = x + jump_x-0.37
            text_y = y + jump_y-0.37
            axs.text(text_x, text_y, label, color='black', ha='center', va='center')
    axs.set_title('Jobs (%)', fontdict={'fontsize': 18},pad=10)
    pos=axs.imshow(H,  interpolation='none',cmap='cool')
    fig.colorbar(pos, ax=axs)
    plt.show()

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
