import numpy as np
from varname.helpers import debug
import csv
import os
from tools_dl.tools import (
    boolean_string,
    ensure_folder,
    curDateTime,
    list2chunks
)

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt


def corr_form_no_leading_zero(x, pos):
    return '{:.2f}'.format(x).replace('0.', '.').replace('1.00', '')

def corr_form(x, pos):
    return '{:.2f}'.format(x)

def heatmap(
    data, 
    row_labels, 
    col_labels, 
    ax=None,
    cbar_kw=None, 
    cbarlabel='', 
    **kwargs
):
    '''
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    '''

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va='bottom')

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha='right',
             rotation_mode='anchor')

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im, 
    data=None, 
    valfmt='{x:.2f}',
    textcolors=('black', 'white'),
    threshold=None, 
    **textkw
):
    '''
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. '$ {x:.2f}', or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    '''

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        # threshold = im.norm(data.max())/2.
        threshold = im.norm(data.max())*3/4.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment='center',
              verticalalignment='center')
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each 'pixel'.
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def matrix2heatmap(
    matrix,
    v_header,
    h_header,
    cmap,
    cbarlabel,
    valfmt=matplotlib.ticker.FuncFormatter(
        corr_form
    ),
    out_dir='/home/ilpech/datasets/test'
):
    create_time = curDateTime()
    if not os.path.isdir(os.path.dirname(out_dir)):
        ensure_folder(out_dir)
    fig, ax = plt.subplots()
    im, cbar = heatmap(
        matrix, 
        v_header, 
        h_header, 
        ax=ax,
        cmap=cmap, 
        cbarlabel=cbarlabel
    )
    if valfmt is not None:
        texts = annotate_heatmap(
            im, 
            valfmt=valfmt
        )
    fig.tight_layout()
    opath = out_dir
    if os.path.isdir(opath):
        opath = os.path.join(
            out_dir,
            'heatmap_{}.png'.format(create_time)
        )
    # plt.savefig(opath, dpi=228)
    plt.savefig(opath, dpi=1024)
    print(f'heatmap table saved to {opath}')

def violin(
    data,
    header,
    out_dir='/home/ilpech/datasets/test/violin_plot',
    means=True,
    medians=True,
    facecolor='#D43F3A',
    ylabel=None,
    out_name=None
):
    create_time = curDateTime()
    if not os.path.isdir(out_dir):
        ensure_folder(out_dir)
    fig, ax = plt.subplots()
    parts = ax.violinplot(
        data,
        showmeans=means,
        showmedians=medians,
        showextrema=False,
        widths=0.55
    )
    for pc in parts['bodies']:
        # print(pc)
        # print(type(pc))
        # exit()
        pc.set_facecolor(facecolor)
        # pc.set_facecolor('#D43F3A')
        # pc.set_facecolor('#D91C9B')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    parts['cmedians'].set_color('green')
    ax.yaxis.grid(True)
    ax.set_xticks(
        [
            y + 1 for y in range(len(header))
        ],
        labels=list(header)
    )
    ax.set_ylabel('Tissue29::log(RNA_value + 1)')
    plt.setp(
        ax.get_xticklabels(), 
        rotation=-30, 
        ha='left',
        rotation_mode='anchor'
    )
    # ax.spines[:].set_visible(False)
    # ax.set_xticks(np.arange(len(header)+1)-.5, minor=True)
    # ax.set_yticks(np.arange(len(header)+1)-.5, minor=True)
    # ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    # ax.tick_params(which='minor', bottom=False, left=False)
    fig.tight_layout()
    opath = out_dir
    if os.path.isdir(opath):
        opath = os.path.join(
            out_dir,
            'violin_{}.png'.format(create_time)
        )
        if out_name:
            opath = os.path.join(
                out_dir,
                'violin_{}_{}.png'.format(create_time, out_name)
            )            
    plt.savefig(opath, dpi=1024)
    print(f'violin viz saved to {opath}')

def violin_plot_exps_gt(
    data,
    dataset_name,
    max_in_one=10,
    facecolor=None,
    key2process='_gt_',
    out_dir='/home/ilpech/datasets/test/violin_plot',
    sum2rna_prot=False
):
    if sum2rna_prot:
        violin_plot_exps_gt_sum(
            data=data,
            dataset_name=dataset_name,
            facecolor=facecolor,
            key2process=key2process,
            out_dir=out_dir
        )
    net_names = list(data.keys())
    exps = []
    for net_name in net_names:
        net_data = data[net_name][dataset_name]
        keys2process = [
            x for x in net_data.keys() 
            if key2process in x
        ]
        for key in keys2process:
            if key not in exps:
                exps.append(key)
        exps_patches = list2chunks(exps, max_in_one)
        for patch in exps_patches:
            matrix = [0] * len(patch)
            v_header = [0] * len(patch) 
            for j, net_name in enumerate(net_names):
                gt_data = data[net_name][dataset_name]
                for i, exp_name in enumerate(patch):
                    matrix[i] = gt_data[exp_name]
                    v_header[i] = '{} ({})'.format(
                        exp_name.split('_')[-1],
                        len(matrix[i])
                    )
            y_label_type = 'protein'
            if 'rna' in key2process:
                y_label_type = 'RNA'
            violin(
                matrix,
                v_header,
                facecolor=facecolor,
                out_dir=out_dir,
                ylabel='{}(log({} + 1)'.format(
                    dataset_name,
                    y_label_type
                )
            )
        
    
def violin_plot_exps_gt_sum(
    data,
    dataset_name,
    facecolor=None,
    key2process='_gt_',
    out_dir='/home/ilpech/datasets/test/violin_plot',
):
    net_names = list(data.keys())
    for net_name in net_names:
        net_data = data[net_name][dataset_name]
        keys2process = [
            x for x in net_data.keys() 
            if key2process in x
        ]
        for patch in keys2process:
            matrix = [0] * 2
            v_header = [None] * 2
            for j, net_name in enumerate(net_names):
                gt_data = data[net_name][dataset_name]
                for l, exp_name in enumerate(patch):
                    i = 0
                    if 'rna' in exp_name:
                        i = 1
                    matrix[i] = gt_data[exp_name]
                    v_header[i] = '{} ({})'.format(
                        exp_name.split('_')[-1],
                        exp_name.split('_')[0]
                    )
            y_label_type = 'protein'
            if 'rna' in key2process:
                y_label_type = 'RNA'
            violin(
                matrix,
                v_header,
                facecolor=facecolor,
                out_dir=out_dir,
                ylabel='{}(log({} + 1)'.format(
                    dataset_name,
                    y_label_type
                )
            )
        
    