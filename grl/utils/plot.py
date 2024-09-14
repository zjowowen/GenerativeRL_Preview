def plot_distribution(data, save_path, size=None, plot_type='hist2d',dpi=500):
    """
    Plot the distribution of the given data and save it as an image.

    Parameters:
    data (torch.Tensor or np.ndarray): Data with shape (batch, dim).
    save_path (str): Path to save the image.
    size (int):size of picture
    plot_type (str): Specify the type of 2D plot; options are 'kde', 'hexbin', or 'hist2d'.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    # If data is a torch.Tensor, convert it to numpy
    if hasattr(data, 'detach') and callable(data.detach):
        data = data.detach().cpu().numpy()
    
    # Ensure the data is 2-dimensional
    assert len(data.shape) == 2, "data needs to be a 2-dimensional tensor or array"
    dim = data.shape[1]
    
    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=[f'Dim {i+1}' for i in range(dim)])
    
    # Create PairGrid
    g = sns.PairGrid(df)

    # Plot according to the specified plot_type
    if plot_type == 'kde':
        g.map_diag(sns.histplot, kde=False, color='skyblue', edgecolor='black')
        g.map_offdiag(sns.kdeplot, cmap='Blues', fill=True, thresh=0)
    
    elif plot_type == 'hexbin':
        sns.set_style('whitegrid')
        g.map_diag(sns.histplot, kde=False, color='skyblue', edgecolor='black')
        # Save the first mappable object
        mappables = []
        def hexbin_func(x, y, **kwargs):
            ax = plt.gca()
            hb = ax.hexbin(x, y, gridsize=30, cmap='Blues')
            mappables.append(hb)
        g.map_offdiag(hexbin_func)
    
    elif plot_type == 'hist2d':
        sns.set_style('whitegrid')
        g.map_diag(sns.histplot, kde=False, color='magenta', edgecolor='black')
        # Save the first mappable object
        mappables = []
        def hist2d_func(x, y, **kwargs):
            ax = plt.gca()
            counts, xedges, yedges, im = ax.hist2d(x, y, bins=30, cmap='Purples')
            mappables.append(im)
        g.map_offdiag(hist2d_func)
    else:
        raise ValueError("plot_type parameter must be 'kde', 'hexbin', or 'hist2d'")
    
    if plot_type != "kde" and mappables:
        # Add a colorbar at the bottom of the plot
        cbar = g.fig.colorbar(
            mappables[0],
            ax=g.axes,
            orientation='horizontal',
            fraction=0.05,  # Adjust as needed
            pad=0.1         # Adjust as needed
        )
        cbar.ax.set_xlabel('Density')
    
    if size is not None:
        for i in range(dim):
            for j in range(dim):
                ax = g.axes[i, j]
                ax.set_xlim(-size, size)
                if i != j:  # Only set y-axis limits for off-diagonal subplots
                    ax.set_ylim(-size, size)
    # Set title and layout
    plt.suptitle('Analyse Data Distribution', fontsize=16, y=1.02)
    # Save image
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()