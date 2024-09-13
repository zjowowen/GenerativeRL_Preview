import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

# def plot_distribution(data, save_path):
#     """
#     绘制给定数据的分布图并保存为图片。

#     参数:
#     data (torch.Tensor or np.ndarray): 数据，形状为 (batch, dim)。
#     save_path (str): 图片保存的路径。
#     """
#     if hasattr(data, 'detach') and callable(data.detach):
#         data = data.detach().cpu().numpy()
#     assert len(data.shape) == 2, "data 需要是一个二维张量或数组"
#     dim = data.shape[1]
#     df = pd.DataFrame(data, columns=[f'Dim {i+1}' for i in range(dim)])
#     g = sns.PairGrid(df)
#     g.map_diag(sns.histplot, kde=False, color='skyblue', edgecolor='black')
#     g.map_offdiag(sns.kdeplot, cmap='Blues', fill=True, thresh=0)
#     sns.set_style('whitegrid')
#     plt.suptitle('Analyse Data Distribution', fontsize=16, y=1.02)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=500, bbox_inches='tight')
#     plt.close()
    
    
def plot_distribution(data, save_path, plot_type='kde'):
    """
    绘制给定数据的分布图并保存为图片

    参数:
    data (torch.Tensor or np.ndarray): 数据，形状为 (batch, dim)。
    save_path (str): 图片保存的路径。
    plot_type (str): 指定二维图的类型，可选值为 'kde'、'hexbin' 或 'hist2d'。
    wandb_log (bool): 如果为 True，则将图片上传到 wandb。
    """
    # 如果 data 是 torch.Tensor，将其转为 numpy
    if hasattr(data, 'detach') and callable(data.detach):
        data = data.detach().cpu().numpy()
    
    # 确保数据是二维的
    assert len(data.shape) == 2, "data 需要是一个二维张量或数组"
    dim = data.shape[1]
    
    # 将数据转换为 DataFrame
    df = pd.DataFrame(data, columns=[f'Dim {i+1}' for i in range(dim)])

    # 创建 PairGrid
    g = sns.PairGrid(df)
    g.map_diag(sns.histplot, kde=False, color='skyblue', edgecolor='black')

    # 根据指定的 plot_type 绘图
    if plot_type == 'kde':
        g.map_offdiag(sns.kdeplot, cmap='Blues', fill=True, thresh=0)
    elif plot_type == 'hexbin':
        def hexbin_func(x, y, **kwargs):
            plt.hexbin(x, y, gridsize=30, cmap='Blues')
        g.map_offdiag(hexbin_func)
    elif plot_type == 'hist2d':
        def hist2d_func(x, y, **kwargs):
            plt.hist2d(x, y, bins=30, cmap='Blues')
        g.map_offdiag(hist2d_func)
    else:
        raise ValueError("plot_type 参数必须是 'kde'、'hexbin' 或 'hist2d'")
    
    # 设置风格和布局
    sns.set_style('whitegrid')
    plt.suptitle('Analyse Data Distribution', fontsize=16, y=1.02)
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()
    
    # # 将图片上传到 wandb
    # if wandb_log:
    #     wandb_image = wandb.Image(save_path, caption="Data Distribution")
    #     wandb.log({"distribution_plot": wandb_image})