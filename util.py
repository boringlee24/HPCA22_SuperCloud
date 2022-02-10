import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

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
