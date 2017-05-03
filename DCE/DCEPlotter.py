import math
import numpy as np
import matplotlib.pyplot as pyplot
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.ticker import FormatStrFormatter


# noinspection PyTypeChecker
def plot_dce(subplot, in_file_name):
    # print 'plotting dce...'
    dce_data = np.loadtxt(in_file_name)
    x = dce_data[:,0]
    y = dce_data[:,1]
    subplot.scatter(x, y, color='black', s=.1)
    # subplot.set_aspect('equal')
    subplot.set(adjustable='box-forced', aspect='equal')
    subplot.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    subplot.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    xlims = subplot.get_xlim()
    subplot.set_xticks([0, xlims[1]])

    ylims = subplot.get_ylim()
    subplot.set_yticks([0, ylims[1]])

    #
    # max_label = AnchoredText('%.3f' %ylims[1],
    #                          prop=dict(size=8), frameon=True,
    #                          loc=2)
    # min_label = AnchoredText('%.3f' %ylims[0],
    #                          prop=dict(size=8), frameon=True,
    #                          loc=3)
    # # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    # subplot.add_artist(max_label)
    # subplot.add_artist(min_label)


# noinspection PyTypeChecker
def plot_waveform(subplot, waveform_data, embed_crop):
    # print 'plotting waveform...'
    y = waveform_data
    x = np.linspace(0, len(y)/44100., len(y))


    subplot.plot(x, y, color='k', zorder=0, lw= .5)
    subplot.axis('tight')
    if math.fabs(embed_crop[0] - embed_crop[1]) < .01:   # how to un-hardcode?
        subplot.axvline(embed_crop[0], color='r', alpha=0.7, zorder=1)
    subplot.axvspan(embed_crop[0], embed_crop[1], facecolor='r', alpha=0.5, zorder=1)


    subplot.set_xlabel('time (s)')

    subplot.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ylims = subplot.get_ylim()
    subplot.set_yticks([ylims[0], 0, ylims[1]])

    # ymax_norm = subplot.get_ylim()[1]
    # ymax = 20 * math.log(ymax_norm, 10) # convert normalized -> dB
    # subplot.set_yticks(subplot.get_ylim())  # no ticks
    # max_label = AnchoredText('%.3f' % ymax,
    #                          prop={'size': 8},
    #                          frameon=True,
    #                          loc=2)
    # subplot.add_artist(max_label)
    # subplot.add_artist(min_label)


def plot_title(subplot, in_file_name, tau):
    subplot.axis('off')
    subplot.set_xlim([0,1])
    subplot.set_ylim([0,1])

    tau_str = r'$\tau = %d$' % tau
    subplot.text(.5,.5 , tau_str,
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=14,
                 bbox=dict(facecolor='none')
                 )

    # subplot.text(0, .1, in_file_name,
    #              horizontalalignment='left',
    #              verticalalignment='bottom',
    #              size=8,
    #              bbox=dict(facecolor='none')
    #              )


def make_window_frame(coords_file_name, wave_file_name, out_file_name, embed_crop, tau, frame_num):
    fig = pyplot.figure(figsize=(9, 9), tight_layout=False)
    fig.subplots_adjust(hspace=.5)
    fig.suptitle(wave_file_name)
    title_subplot = pyplot.subplot2grid((4, 4), (0, 3), rowspan=3)
    dce_subplot = pyplot.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
    wavform_subplot = pyplot.subplot2grid((4, 4), (3, 0), colspan=4)

    wave_data = np.loadtxt(wave_file_name)

    plot_dce(dce_subplot, coords_file_name)
    plot_waveform(wavform_subplot, wave_data, embed_crop)
    plot_title(title_subplot, wave_file_name, tau)
    pyplot.savefig(out_file_name)
    pyplot.close(fig)


def compare_vary_tau_frame(out_file_name, wave_file_name1, wave_file_name2, frame_num, tau, embed_crop):
    fig = pyplot.figure(figsize=(12, 9), tight_layout=True)
    subplot1 = pyplot.subplot2grid((5, 2), (0,0), rowspan=4)
    subplot2 = pyplot.subplot2grid((5, 2), (0,1), rowspan=4)
    subplot3 = pyplot.subplot2grid((5, 2), (4,0))
    subplot4 = pyplot.subplot2grid((5, 2), (4, 1), sharey=subplot3)
    pyplot.setp(subplot4.get_yticklabels(), visible=False)

    plot_dce(subplot1, 'DCE/temp_data/embedded_coords_comp1.txt')
    plot_dce(subplot2, 'DCE/temp_data/embedded_coords_comp2.txt')

    wave_data1, wave_data2 = np.loadtxt(wave_file_name1), np.loadtxt(wave_file_name2)
    plot_waveform(subplot3, wave_data1, embed_crop)
    plot_waveform(subplot4, wave_data2, embed_crop)

    subplot1.set_title(wave_file_name1)
    subplot2.set_title(wave_file_name2)
    fig.suptitle('$tau = %d$' % tau, bbox={'pad':5}, fontsize=14)

    pyplot.savefig(out_file_name)
    pyplot.close(fig)


def compare_multi_frame(out_file_name, wave_file_name1, wave_file_name2, frame_num, tau, embed_crop):
    fig = pyplot.figure(figsize=(12, 9), tight_layout=True)
    subplot1 = pyplot.subplot2grid((5, 2), (0, 0), rowspan=4)
    subplot2 = pyplot.subplot2grid((5, 2), (0, 1), rowspan=4)
    subplot3 = pyplot.subplot2grid((5, 2), (4, 0))
    subplot4 = pyplot.subplot2grid((5, 2), (4, 1), sharey=subplot3)
    pyplot.setp(subplot4.get_yticklabels(), visible=False)

    plot_dce(subplot1, 'DCE/temp_data/embedded_coords_comp1.txt')
    plot_dce(subplot2, 'DCE/temp_data/embedded_coords_comp2.txt')

    wave_data1, wave_data2 = np.loadtxt(wave_file_name1), np.loadtxt(wave_file_name2)
    plot_waveform(subplot3, wave_data1, embed_crop)
    plot_waveform(subplot4, wave_data2, embed_crop)

    subplot1.set_title(wave_file_name1.split('/')[-1])
    subplot2.set_title(wave_file_name2.split('/')[-1])
    fig.suptitle('$tau =  %d$' % tau, bbox={'pad': 5}, fontsize=14)

    pyplot.savefig(out_file_name)
    pyplot.close(fig)


def plot_titlebox(subplot, info):

    # subplot.axis('tight')
    subplot.axis('off')
    # subplot.xaxis.set_ticks([])
    # subplot.yaxis.set_ticks([])
    subplot.set_xlim([0, 1])
    subplot.set_ylim([0, 1])

    info_main = info['title_main']
    info1 = info['title_1']
    info2 = info['title_2']

    col_widths = [1, .75]
    col_heights = .05
    spacing_h = .05
    header_h = .1

    h2 = col_heights * (len(info2)-1)
    h1 = col_heights * (len(info1) - 1)
    h_main = col_heights * len(info_main)

    pos2 = 0
    pos1 = pos2 + h2 + header_h + spacing_h
    pos_main = pos1 + h1 + header_h + spacing_h

    tables = []

    table_main = subplot.table(
        cellText=info_main,
        colWidths=col_widths,
        bbox=[0, pos_main, 1, h_main],  # x0, y0, width, height
    )


    header_1 = subplot.table(   # higher
        cellText=[[info1[0].split('/')[-1]]],
        bbox=[0, pos1 + h1, 1, header_h],
        cellLoc='center'
    )

    try:
        table_1 = subplot.table(    # higher
            cellText=info1[1:],
            colWidths=col_widths,
            bbox=[0, pos1, 1, h1],
        )
        tables.append(table_1)
        pass
    except IndexError:
        pass


    header_2 = subplot.table(   # lower
        cellText=[[info2[0].split('/')[-1]]],
        bbox=[0, h2, 1, header_h],
        cellLoc='center'

    )
    try:
        table_2 = subplot.table(    # lower
            cellText=info2[1:],
            colWidths=col_widths,
            bbox=[0, pos2, 1, h2],
        )
        tables.append(table_2)
        pass
    except IndexError:
        pass

    for table in [header_1, header_2]:
        table.auto_set_font_size(False)
        table.set_fontsize(10)

    for table in tables:
        table.auto_set_font_size(False)
        table.set_fontsize(8)



def compare_multi_frame_new(out_file_name, wave_file_name1, wave_file_name2, frame_num, crop_1, crop_2, info, dpi):
    fig = pyplot.figure(figsize=(11, 5), tight_layout=True, dpi=dpi)
    titlebox = pyplot.subplot2grid((5, 11), (0, 0), rowspan=5, colspan=3)
    dce1 = pyplot.subplot2grid((5, 11), (0, 3), rowspan=4, colspan=4)
    dce2 = pyplot.subplot2grid((5, 11), (0, 7), rowspan=4, colspan=4)
    wave1 = pyplot.subplot2grid((5, 11), (4, 3), colspan=4)
    wave2 = pyplot.subplot2grid((5, 11), (4, 7), colspan=4, sharey=wave1)
    pyplot.setp(wave2.get_yticklabels(), visible=False)

    plot_titlebox(titlebox, info)

    plot_dce(dce1, 'DCE/temp_data/embedded_coords_comp1.txt')
    plot_dce(dce2, 'DCE/temp_data/embedded_coords_comp2.txt')

    wave_data1, wave_data2 = np.loadtxt(wave_file_name1), np.loadtxt(wave_file_name2)
    plot_waveform(wave1, wave_data1, crop_1)
    plot_waveform(wave2, wave_data2, crop_2)

    dce1.set_title(wave_file_name1.split('/')[-1])
    dce2.set_title(wave_file_name2.split('/')[-1])


    pyplot.savefig(out_file_name)
    pyplot.close(fig)

